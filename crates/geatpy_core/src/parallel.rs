//! Thread pool for the operators that run in parallel.
//!
//! Parallel paths only split work whose result does not depend on how it is split (one output row
//! per task, one candidate per task), so in the default floating-point environment the result is
//! bit-identical to the serial one for every thread count. (Worker threads keep the
//! floating-point control bits of the thread that built the pool, so a caller that turns on
//! flush-to-zero afterwards may see subnormal results differ.)
//!
//! Every parallel call has a hand-off cost. Between two calls in an algorithm the workers have
//! fallen asleep (they do after a few tens of microseconds idle), and waking a pool costs more the
//! more threads it has, however little of it the work uses. How much depends on the machine, above
//! all on how deep its idle cores sleep: on an M2 Pro about 16 us for two threads, 30 us for four
//! and 90 us for ten; on an i9-13900H laptop under Linux (C3, 1 ms exit latency) 20-150 us for two
//! and 130-280 us for four or more; on a Xeon Silver 4310 server (C6, 170 us) 25-65 us for up to
//! eight and 55-160 us for 24. So besides the pool of every thread there are pools of 2, 4, 8, ...
//! threads, and `pool_sized` gives a call the one expected to finish first, or none when no pool
//! beats the serial path clearly. Outside macOS the hand-off of each pool is measured once, when a
//! call first could go parallel.

use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Parallel calls made so far (read by the tests to check that their inputs go parallel).
static PARALLEL_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Threads set by `set_num_threads`; 0 means the default.
static REQUESTED: AtomicUsize = AtomicUsize::new(0);

/// The pools built so far and their thread counts (`None` as a pool remembers a failed build;
/// powers of two are kept, other sizes, all threads, only until another one is built), and the
/// hand-off cost in ns measured for each pool size.
struct Slot {
    pools: Vec<(usize, Option<Arc<ThreadPool>>)>,
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    handoffs: Vec<(usize, f64)>,
}
static POOL: Mutex<Slot> = Mutex::new(Slot {
    pools: Vec::new(),
    handoffs: Vec::new(),
});

fn lock() -> MutexGuard<'static, Slot> {
    POOL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Logical CPUs available to this process (respecting CPU affinity): more threads than this only
/// slow the work down, so it caps every request.
fn cpus() -> usize {
    static CPUS: OnceLock<usize> = OnceLock::new();
    *CPUS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// GEATPY_NUM_THREADS when it is a positive integer, otherwise the number of logical CPUs.
fn default_threads() -> usize {
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        std::env::var("GEATPY_NUM_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(cpus)
    })
}

/// Threads the operators use: the request (or the default), at most the number of logical CPUs.
pub fn num_threads() -> usize {
    let n = match REQUESTED.load(Ordering::Relaxed) {
        0 => default_threads(),
        n => n,
    };
    n.clamp(1, cpus())
}

/// The pool of every thread to run `work` units on, or None to run serially below `min_work`.
pub fn pool_for(work: usize, min_work: usize) -> Option<Arc<ThreadPool>> {
    let threads = num_threads();
    if threads <= 1 || work < min_work {
        return None;
    }
    pool(threads)
}

/// Below this serial time (ns) no machine measured gains from going parallel, so `pool_sized`
/// decides without measuring any hand-off.
const MIN_PARALLEL_NS: f64 = 20_000.0;

/// Speed-up of the work itself on a pool of `threads`. Outside macOS logical CPUs often include
/// hyper-threads and efficiency cores (an i9-13900H has 20 for 6 performance cores): there GD
/// 800 x 3 ran 5.6 times faster on 8 threads and 7.6 on 20, after the hand-off, about p^0.7.
#[cfg(target_os = "macos")]
fn speedup(threads: usize) -> f64 {
    threads as f64
}
#[cfg(not(target_os = "macos"))]
fn speedup(threads: usize) -> f64 {
    (threads as f64).powf(0.7)
}

/// The pool to run `work` units on, each taking about `unit_ns` serially, or None to run
/// serially. With p threads a call is expected to take serial / speedup(p) plus the pool's
/// hand-off; the pool (2, 4, 8, ... or every thread) expected to finish first is taken if that
/// beats 85% of the serial time, a margin for uneven cores and lower all-core clocks.
pub fn pool_sized(work: usize, unit_ns: f64) -> Option<Arc<ThreadPool>> {
    let threads = num_threads();
    let serial = work as f64 * unit_ns;
    if threads <= 1 || serial < MIN_PARALLEL_NS {
        return None;
    }
    let (best, expected) = std::iter::successors(Some(2usize), |&p| p.checked_mul(2))
        .take_while(|&p| p < threads)
        .chain(std::iter::once(threads))
        .filter_map(|p| {
            let (pool, ns) = with_handoff(p)?;
            Some((pool, serial / speedup(p) + ns))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    (expected < 0.85 * serial).then_some(best)
}

/// The pool of `threads` workers and its hand-off cost in ns, None if it cannot be built.
#[cfg(target_os = "macos")]
fn with_handoff(threads: usize) -> Option<(Arc<ThreadPool>, f64)> {
    pool(threads).map(|p| (p, 8_000.0 * threads as f64))
}

/// The pool of `threads` workers and its hand-off cost in ns, None if it cannot be built. The
/// cost is measured on the pool's first use, without holding the lock: nine calls that each give
/// every worker 20 us of work after 2, 5 or 10 ms idle (how deep idle cores sleep, and so how long
/// they take to wake, varies from call to call and with the idle time), and the second largest
/// excess over 20 us, about 55 ms in all. Threads measuring the same pool at once slow each other
/// down, which only errs towards staying serial. The measurement is kept across forks.
#[cfg(not(target_os = "macos"))]
fn with_handoff(threads: usize) -> Option<(Arc<ThreadPool>, f64)> {
    use std::time::{Duration, Instant};
    let known = |slot: &Slot| {
        slot.handoffs
            .iter()
            .find(|(t, _)| *t == threads)
            .map(|&(_, ns)| ns)
    };
    let pool = pool(threads)?;
    if let Some(ns) = known(&lock()) {
        return Some((pool, ns));
    }
    const WORK: Duration = Duration::from_micros(20);
    let call = || {
        pool.install(|| {
            (0..threads).into_par_iter().for_each(|_| {
                let start = Instant::now();
                while start.elapsed() < WORK {
                    std::hint::spin_loop();
                }
            })
        })
    };
    call();
    let mut excess: Vec<f64> = [2, 5, 10, 2, 5, 10, 2, 5, 10]
        .iter()
        .map(|&ms| {
            std::thread::sleep(Duration::from_millis(ms));
            let start = Instant::now();
            call();
            start.elapsed().saturating_sub(WORK).as_nanos() as f64
        })
        .collect();
    excess.sort_by(f64::total_cmp);
    let ns = excess[excess.len() - 2];
    let mut slot = lock();
    // another thread measured it meanwhile: keep that one
    let ns = known(&slot).unwrap_or_else(|| {
        slot.handoffs.push((threads, ns));
        ns
    });
    drop(slot);
    Some((pool, ns))
}

/// The pool of `threads` workers. It is built without holding the lock (building starts OS
/// threads), so a fork in another thread never inherits it locked.
fn pool(threads: usize) -> Option<Arc<ThreadPool>> {
    let find = |slot: &Slot| {
        slot.pools
            .iter()
            .find(|(t, _)| *t == threads)
            .map(|(_, pool)| pool.clone())
    };
    if let Some(pool) = find(&lock()) {
        return pool;
    }
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("geatpy-{i}"))
        .build()
        .ok()
        .map(Arc::new);
    let mut slot = lock();
    // another thread installed a pool of this size meanwhile: use that one
    if let Some(pool) = find(&slot) {
        return pool;
    }
    if !threads.is_power_of_two() {
        slot.pools.retain(|(t, _)| t.is_power_of_two());
    }
    slot.pools.push((threads, built.clone()));
    built
}

thread_local! {
    /// The pool lock held by the thread calling `os.fork`, from just before the fork until after it.
    static FORK_GUARD: RefCell<Option<MutexGuard<'static, Slot>>> = const { RefCell::new(None) };
}

/// Registers `os.register_at_fork` handlers: the pool lock is taken before a fork and released in
/// both processes afterwards, and the child forgets the pools, whose worker threads exist only in
/// the parent (they are leaked, as tearing them down would wait for those threads).
pub fn register_fork_handlers(py: Python<'_>) -> PyResult<()> {
    use pyo3::types::{PyCFunction, PyDict, PyTuple};
    let os = py.import("os")?;
    if !os.hasattr("register_at_fork")? {
        return Ok(()); // Windows: no fork
    }
    let before = PyCFunction::new_closure(
        py,
        None,
        None,
        |_: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| {
            FORK_GUARD.with(|g| *g.borrow_mut() = Some(lock()));
        },
    )?;
    let parent = PyCFunction::new_closure(
        py,
        None,
        None,
        |_: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| {
            FORK_GUARD.with(|g| g.borrow_mut().take());
        },
    )?;
    let child = PyCFunction::new_closure(
        py,
        None,
        None,
        |_: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| {
            FORK_GUARD.with(|g| {
                if let Some(mut slot) = g.borrow_mut().take() {
                    std::mem::forget(std::mem::take(&mut slot.pools));
                }
            });
        },
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("before", before)?;
    kwargs.set_item("after_in_parent", parent)?;
    kwargs.set_item("after_in_child", child)?;
    os.getattr("register_at_fork")?.call((), Some(&kwargs))?;
    Ok(())
}

/// (0..n).map(f).collect(), on `pool` when there is one.
pub fn map_collect<T, F>(pool: Option<&ThreadPool>, n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    match pool {
        Some(p) => {
            PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
            p.install(|| (0..n).into_par_iter().map(f).collect())
        }
        None => (0..n).map(f).collect(),
    }
}

/// Like `map_collect` with a scratch value made by `init` once per piece of work a thread takes
/// (once in all when serial) instead of once per item.
pub fn map_collect_init<T, S, I, F>(pool: Option<&ThreadPool>, n: usize, init: I, f: F) -> Vec<T>
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> T + Sync + Send,
{
    match pool {
        Some(p) => {
            PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
            p.install(|| (0..n).into_par_iter().map_init(&init, &f).collect())
        }
        None => {
            let mut scratch = init();
            (0..n).map(|i| f(&mut scratch, i)).collect()
        }
    }
}

/// (0..n).fold(init(), f), on `pool` when there is one: pieces of the range are folded from their
/// own init() and combined by `merge`, which must give the same result however the range is split.
pub fn fold_merge<A, I, F, M>(pool: Option<&ThreadPool>, n: usize, init: I, f: F, merge: M) -> A
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(A, usize) -> A + Sync + Send,
    M: Fn(A, A) -> A + Sync + Send,
{
    match pool {
        Some(p) => {
            PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
            p.install(|| (0..n).into_par_iter().fold(&init, &f).reduce(&init, &merge))
        }
        None => (0..n).fold(init(), f),
    }
}

/// Calls f(row index, row) for every `width`-long row of `data`, on `pool` when there is one.
pub fn for_each_row<T, F>(pool: Option<&ThreadPool>, data: &mut [T], width: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if width == 0 {
        return;
    }
    match pool {
        Some(p) => {
            PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
            p.install(|| {
                data.par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(i, row)| f(i, row))
            })
        }
        None => data
            .chunks_mut(width)
            .enumerate()
            .for_each(|(i, row)| f(i, row)),
    }
}

/// Sets the number of threads the operators may use; None or 0 restores the default
/// (GEATPY_NUM_THREADS, or the number of logical CPUs). 1 runs everything serially. More threads
/// than logical CPUs are not used.
#[pyfunction]
#[pyo3(signature = (n=None))]
pub fn set_num_threads(n: Option<usize>) {
    REQUESTED.store(n.unwrap_or(0), Ordering::Relaxed);
}

/// The number of threads the operators use.
#[pyfunction]
pub fn get_num_threads() -> usize {
    num_threads()
}

/// Number of parallel calls so far; for the tests only.
#[pyfunction]
pub fn _parallel_calls() -> usize {
    PARALLEL_CALLS.load(Ordering::Relaxed)
}
