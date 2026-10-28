use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::seq::SliceRandom;
use rand::Rng;

type Opt<'a, 'py> = Option<&'a Bound<'py, PyAny>>;

/// FitnV_N is either a fitness column vector or the population size (all fitness equal to 1).
fn fitness_or_n(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if !obj.hasattr("__len__")? {
        if let Ok(n) = obj.extract::<f64>() {
            let n = n.max(0.0) as usize;
            crate::utils::check_alloc(&[n], 8, "float64")?;
            return Ok(vec![1.0; n]);
        }
    }
    let a = crate::utils::to_f64_array2(obj)?;
    Ok(a.column(0).to_vec())
}

/// Number of individuals to select: None => n, (0, 1) => ceil(n * x), otherwise ceil(x).
fn sel_count(nsel: Opt, n: usize) -> PyResult<usize> {
    let count = match nsel {
        Some(o) if !o.is_none() => {
            let x: f64 = o.extract()?;
            if x > 0.0 && x < 1.0 {
                (n as f64 * x).ceil() as usize
            } else {
                x.ceil().max(0.0) as usize
            }
        }
        _ => n,
    };
    // the indices are collected as 8-byte integers before the int32 output
    crate::utils::check_alloc(&[count], 8, "int64")?;
    Ok(count)
}

/// Selected indices as int32, like geatpy 2.7.0 (its C kernels fill np.int32 buffers).
fn idx_out(py: Python<'_>, v: Vec<usize>) -> Bound<'_, PyArray1<i32>> {
    Array1::from_iter(v.into_iter().map(|x| x as i32)).into_pyarray(py)
}

/// int64 indices for the operators geatpy 2.7.0 implements with numpy (urs, ecs, otos with Nsel = N).
fn idx_out64(py: Python<'_>, v: Vec<usize>) -> Bound<'_, PyArray1<i64>> {
    Array1::from_iter(v.into_iter().map(|x| x as i64)).into_pyarray(py)
}

/// Whether `name` returns int64 indices for this call (see `idx_out64`).
fn wide_output(name: &str, n: usize, nsel: usize) -> bool {
    matches!(name, "urs" | "ecs") || (name == "otos" && nsel == n)
}

/// Indices by descending fitness, stable, NaN last (geatpy 2.7.0:
/// `np.argsort(-FitnV, kind='mergesort')`).
fn order_desc(fit: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..fit.len()).collect();
    o.sort_by(|&a, &b| crate::utils::cmp_nan_last(-fit[a], -fit[b]));
    o
}

fn dup_idx(fit: &[f64], nsel: usize) -> Vec<usize> {
    let n = fit.len();
    if n == 0 {
        return vec![];
    }
    let order = order_desc(fit);
    let mut out: Vec<usize> = (0..nsel).map(|i| order[i % n]).collect();
    out.shuffle(&mut crate::utils::rng());
    out
}

fn tour_idx(fit: &[f64], nsel: usize, k: usize, elitist: bool) -> Vec<usize> {
    let n = fit.len();
    if n == 0 {
        return vec![];
    }
    let k = if k >= 1 && k <= n { k } else { 2.min(n) };
    let mut rng = crate::utils::rng();
    // A later contestant wins only when strictly fitter (ties keep the first one drawn).
    let mut out: Vec<usize> = (0..nsel)
        .map(|_| {
            let mut win = rng.gen_range(0..n);
            for _ in 1..k {
                let c = rng.gen_range(0..n);
                if fit[c] > fit[win] {
                    win = c;
                }
            }
            win
        })
        .collect();
    if elitist && nsel > 0 {
        // The (first) best individual survives: if no tournament picked it, it replaces the
        // fittest pick (geatpy 2.7.0 behaviour).
        let best = (1..n).fold(0, |b, i| if fit[i] > fit[b] { i } else { b });
        if !out.contains(&best) {
            let slot = (1..nsel).fold(0, |s, t| if fit[out[t]] > fit[out[s]] { t } else { s });
            out[slot] = best;
        }
    }
    out
}

fn cumulative(fit: &[f64]) -> (Vec<f64>, f64) {
    let shift = fit.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
    let mut total = 0.0;
    let cum = fit
        .iter()
        .map(|&v| {
            total += v - shift;
            total
        })
        .collect();
    (cum, total)
}

fn rws_idx(fit: &[f64], nsel: usize) -> Vec<usize> {
    let n = fit.len();
    let mut rng = crate::utils::rng();
    let (cum, total) = cumulative(fit);
    (0..nsel)
        .map(|_| {
            if n == 0 {
                return 0;
            }
            if total <= 0.0 {
                return rng.gen_range(0..n);
            }
            let r = rng.gen::<f64>() * total;
            cum.partition_point(|&c| c <= r).min(n - 1)
        })
        .collect()
}

fn sus_idx(fit: &[f64], nsel: usize) -> Vec<usize> {
    let n = fit.len();
    let mut rng = crate::utils::rng();
    let (cum, total) = cumulative(fit);
    if nsel == 0 {
        return vec![];
    }
    if n == 0 {
        // geatpy 2.7.0 returns zeros for an empty population (like rws)
        return vec![0; nsel];
    }
    if total <= 0.0 {
        return (0..nsel).map(|_| rng.gen_range(0..n)).collect();
    }
    let step = total / nsel as f64;
    let start = rng.gen::<f64>() * step;
    // Pointers are returned in ascending order, as in geatpy 2.7.0 (no shuffle).
    (0..nsel)
        .map(|i| {
            cum.partition_point(|&c| c <= start + i as f64 * step)
                .min(n - 1)
        })
        .collect()
}

fn urs_idx(n: usize, nsel: usize) -> PyResult<Vec<usize>> {
    let mut rng = crate::utils::rng();
    if n == 0 {
        // geatpy 2.7.0 calls np.random.randint(0, 0, ...)
        return Err(pyo3::exceptions::PyValueError::new_err("high <= 0"));
    }
    Ok((0..nsel).map(|_| rng.gen_range(0..n)).collect())
}

/// Nsel distinct indices of a random permutation; geatpy 2.7.0 requires 1 <= Nsel <= N.
fn rps_idx(n: usize, nsel: usize) -> PyResult<Vec<usize>> {
    if nsel == 0 || nsel > n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in rps: Nsel must be an integer bigger than 0 and not bigger than N. (Nsel必须是一个大于0且不大于N的正整数。)",
        ));
    }
    let mut v: Vec<usize> = (0..n).collect();
    v.shuffle(&mut crate::utils::rng());
    v.truncate(nsel);
    Ok(v)
}

fn rcs_idx(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let rg = if n > 1 {
        crate::utils::rng().gen_range(1..n)
    } else {
        0
    };
    (0..n).map(|i| (i + rg) % n).collect()
}

/// One-to-one survivor selection: the population is cut into chunks of `nsel`; position j of the
/// result is the fittest among the j-th members of all chunks.
fn otos_idx(fit: &[f64], nsel: usize) -> PyResult<Vec<usize>> {
    let n = fit.len();
    if nsel == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in otos: Nsel must be an integer and be or bigger than 1. (要选择的个体数Nsel必须是int类型而且要不小于1。)",
        ));
    }
    if !n.is_multiple_of(nsel) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in otos: Nsel must be exactly divisible by the number of individuals. (要选择的个体数Nsel必须能够被种群的个体数整除。)",
        ));
    }
    if nsel == n {
        // a single chunk: geatpy 2.7.0 returns np.arange(N) unchanged
        return Ok((0..n).collect());
    }
    let mut out: Vec<usize> = (0..nsel)
        .map(|j| {
            (0..n / nsel)
                .map(|s| s * nsel + j)
                .fold(j, |best, i| if fit[i] > fit[best] { i } else { best })
        })
        .collect();
    out.shuffle(&mut crate::utils::rng());
    Ok(out)
}

fn ecs_idx(fit: &[f64], nsel: usize) -> Vec<usize> {
    if fit.is_empty() {
        return vec![];
    }
    // np.argmax: the first maximum, or the first NaN if there is one
    let best = fit
        .iter()
        .position(|v| v.is_nan())
        .unwrap_or_else(|| (1..fit.len()).fold(0, |b, i| if fit[i] > fit[b] { i } else { b }));
    vec![best; nsel]
}

fn select_by_name(name: &str, fit: &[f64], nsel: usize, tour: usize) -> PyResult<Vec<usize>> {
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    Ok(match name {
        "dup" => dup_idx(fit, nsel),
        "ecs" => ecs_idx(fit, nsel),
        "etour" => tour_idx(fit, nsel, tour, true),
        "tour" => tour_idx(fit, nsel, tour, false),
        "otos" => otos_idx(fit, nsel)?,
        "rcs" => rcs_idx(fit.len()),
        "rps" => rps_idx(fit.len(), nsel)?,
        "rws" => rws_idx(fit, nsel),
        "sus" => sus_idx(fit, nsel),
        "urs" => urs_idx(fit.len(), nsel)?,
        other => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "error in selecting: unknown selection operator '{}'.",
                other
            )))
        }
    })
}

/// Truncation selection: the Nsel fittest individuals (cycling when Nsel > N), in random order.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, params3=None))]
pub fn dup<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    params3: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let _ = (params2, params3);
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    Ok(idx_out(py, dup_idx(&fitness_or_n(fitn_v_n)?, nsel)))
}

/// Elite copy selection: the best individual Nsel times.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, params3=None))]
pub fn ecs<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    params3: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, params3);
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    Ok(idx_out64(py, ecs_idx(&fitness_or_n(fitn_v_n)?, nsel)))
}

/// Tournament selection with replacement.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, tour=None, parallel=None))]
pub fn tour<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    tour: Option<usize>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let _ = parallel;
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    Ok(idx_out(
        py,
        tour_idx(&fitness_or_n(fitn_v_n)?, nsel, tour.unwrap_or(2), false),
    ))
}

/// Tournament selection in which the best individual always takes part (and therefore survives).
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, tour=None, parallel=None))]
pub fn etour<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    tour: Option<usize>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let _ = parallel;
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    Ok(idx_out(
        py,
        tour_idx(&fitness_or_n(fitn_v_n)?, nsel, tour.unwrap_or(2), true),
    ))
}

/// Roulette-wheel selection.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, parallel=None))]
pub fn rws<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let _ = (params2, parallel);
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    Ok(idx_out(py, rws_idx(&fitness_or_n(fitn_v_n)?, nsel)))
}

/// Stochastic universal sampling.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, parallel=None))]
pub fn sus<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let _ = (params2, parallel);
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    Ok(idx_out(py, sus_idx(&fitness_or_n(fitn_v_n)?, nsel)))
}

/// Uniform random selection with replacement.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, params3=None))]
pub fn urs<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    params3: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, params3);
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    Ok(idx_out64(py, urs_idx(fitness_or_n(fitn_v_n)?.len(), nsel)?))
}

/// Random compensation selection: r0 = (i + rg) % N with one random rg in [1, N).
#[pyfunction]
#[pyo3(signature = (fitn_v_n, params1=None, params2=None, parallel=None))]
pub fn rcs<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    params1: Opt<'_, 'py>,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let _ = (params1, params2, parallel);
    Ok(idx_out(py, rcs_idx(fitness_or_n(fitn_v_n)?.len())))
}

/// Random permutation selection without replacement.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel=None, params2=None, params3=None))]
pub fn rps<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: Opt<'_, 'py>,
    params2: Opt<'_, 'py>,
    params3: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let _ = (params2, params3);
    let n = fitness_or_n(fitn_v_n)?.len();
    let k = match nsel {
        Some(o) if !o.is_none() => o.extract::<usize>()?,
        _ => n,
    };
    Ok(idx_out(py, rps_idx(n, k)?))
}

/// One-to-one survivor selection.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, parallel=None))]
pub fn otos<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (params2, parallel);
    crate::utils::check_alloc(&[nsel], 8, "int64")?;
    let fit = fitness_or_n(fitn_v_n)?;
    let idx = otos_idx(&fit, nsel)?;
    Ok(if wide_output("otos", fit.len(), nsel) {
        idx_out64(py, idx).into_any().unbind()
    } else {
        idx_out(py, idx).into_any().unbind()
    })
}

/// Dispatcher: selecting(SEL_F, FitnV_N, NSel, params2, Parallel); params2 is passed on to the
/// operator (the tournament size of tour/etour).
#[pyfunction]
#[pyo3(signature = (sel_f, fitn_v_n, nsel=None, params2=None, parallel=None))]
pub fn selecting<'py>(
    py: Python<'py>,
    sel_f: &str,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: Opt<'_, 'py>,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let fit = fitness_or_n(fitn_v_n)?;
    if let Some(o) = nsel.filter(|o| !o.is_none()) {
        if o.extract::<f64>()? <= 0.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "error in selecting: NSel must be bigger than 0. (待选择的个体数NSel必须不小于0。)",
            ));
        }
    }
    let k = sel_count(nsel, fit.len())?;
    let tour = match params2 {
        Some(p) if !p.is_none() && matches!(sel_f, "tour" | "etour") => p.extract::<usize>()?,
        _ => 2,
    };
    let idx = select_by_name(sel_f, &fit, k, tour)?;
    Ok(if wide_output(sel_f, fit.len(), k) {
        idx_out64(py, idx).into_any().unbind()
    } else {
        idx_out(py, idx).into_any().unbind()
    })
}

/// Multi-population selection. Selection runs over the merged populations; a population that gets
/// nothing then receives its minimum quota (MSel) chosen inside it.
#[pyfunction]
#[pyo3(signature = (sel_f, fitn_vs, nsel=None, msel=None, separate=None))]
pub fn mselecting<'py>(
    py: Python<'py>,
    sel_f: &str,
    fitn_vs: &Bound<'py, PyAny>,
    nsel: Opt<'_, 'py>,
    msel: Option<f64>,
    separate: Option<bool>,
) -> PyResult<Bound<'py, PyList>> {
    let fits: Vec<Vec<f64>> = fitn_vs
        .try_iter()?
        .map(|f| fitness_or_n(&f?))
        .collect::<PyResult<_>>()?;
    if fits.is_empty() {
        // geatpy 2.7.0 concatenates the FitnVs with numpy first
        return Err(pyo3::exceptions::PyValueError::new_err(
            "need at least one array to concatenate",
        ));
    }
    let sizes: Vec<usize> = fits.iter().map(|f| f.len()).collect();
    let total: usize = sizes.iter().sum();
    let msel = msel.unwrap_or(0.1);
    let mins: Vec<usize> = sizes
        .iter()
        .map(|&s| {
            if msel < 1.0 {
                (s as f64 * msel).ceil() as usize
            } else {
                (msel.ceil() as usize).min(s)
            }
        })
        .collect();
    let k = sel_count(nsel, total)?.max(mins.iter().sum());
    let all: Vec<f64> = fits.iter().flatten().cloned().collect();
    let offsets: Vec<usize> = sizes
        .iter()
        .scan(0, |acc, &s| {
            let o = *acc;
            *acc += s;
            Some(o)
        })
        .collect();
    let mut per_pop: Vec<Vec<usize>> = vec![Vec::new(); fits.len()];
    for g in select_by_name(sel_f, &all, k, 2)? {
        let p = offsets.iter().rposition(|&o| o <= g).unwrap();
        per_pop[p].push(g - offsets[p]);
    }
    // Every population gets at least its documented minimum. geatpy 2.7.0 only refilled the
    // populations that got none, so a small one could keep a single individual
    // (geatpy-dev/geatpy#330); topping up the short ones is the same where nothing is short.
    for (p, chosen) in per_pop.iter_mut().enumerate() {
        if chosen.len() < mins[p] {
            let extra = select_by_name(sel_f, &fits[p], mins[p] - chosen.len(), 2)?;
            chosen.extend(extra);
        }
    }
    let separate = separate.unwrap_or(true);
    let out = PyList::empty(py);
    for (p, chosen) in per_pop.into_iter().enumerate() {
        let shift = if separate { 0 } else { offsets[p] };
        // Python lists of indices, like geatpy 2.7.0.
        out.append(PyList::new(
            py,
            chosen.into_iter().map(|i| (i + shift) as i64),
        )?)?;
    }
    Ok(out)
}

/// Single-objective column prepared for fitness assignment (minimisation, feasibility rule applied).
/// geatpy 2.7.0 only accepts an n*1 ndarray ObjV here.
fn prepared_column(
    obj_v: &Bound<'_, PyAny>,
    cv: Opt,
    maxormins: Opt,
    shape_error: &str,
) -> PyResult<Vec<f64>> {
    let shape: Option<Vec<usize>> = obj_v.getattr("shape").and_then(|s| s.extract()).ok();
    if !matches!(shape.as_deref(), Some([_, 1])) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            shape_error.to_string(),
        ));
    }
    // the operator name, as in "error in ranking: ..."
    let name = shape_error
        .strip_prefix("error in ")
        .and_then(|r| r.split(':').next())
        .unwrap_or("");
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins, name)?;
    Ok(obj.column(0).to_vec())
}

fn runtime_err(msg: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(msg.to_string())
}

fn column(v: Vec<f64>) -> Array2<f64> {
    let n = v.len();
    Array2::from_shape_vec((n, 1), v).unwrap()
}

/// Positive real root of (SP - N) x^(N-1) + SP x^(N-2) + ... + SP x + SP = 0 (Pohlheim).
fn nonlinear_root(sp: f64, n: usize) -> f64 {
    let f = |x: f64| {
        (sp - n as f64) * x.powi(n as i32 - 1)
            + (0..n - 1)
                .map(|i| sp * x.powi(i as i32))
                .fold(0.0, |s, x| s + x)
    };
    let (mut lo, mut hi) = (1.0, 2.0);
    while f(hi) > 0.0 && hi < 1e6 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

/// `nonlinear_root` and the fitness of rank p computed from powers of 1/x (every term divided by
/// x^(N-1)), which cannot overflow: the direct form reaches x^(N-1) = inf once N and SP are large
/// (N = 500, SP = 400 has x near 5), where geatpy 2.7.0 returns NaN.
fn nonlinear_fitness_scaled(sp: f64, n: usize, pos: &[usize]) -> Vec<f64> {
    // g(x) = f(x) / x^(N-1) = (SP - N) + SP * sum_{k=1}^{N-1} x^-k, decreasing in x
    let g = |x: f64| {
        let inv = 1.0 / x;
        let (mut term, mut sum) = (1.0, 0.0);
        for _ in 1..n {
            term *= inv;
            sum += term;
        }
        (sp - n as f64) + sp * sum
    };
    let (mut lo, mut hi) = (1.0, 2.0);
    while g(hi) > 0.0 && hi < 1e6 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if g(mid) > 0.0 {
            lo = mid
        } else {
            hi = mid
        }
    }
    let x = 0.5 * (lo + hi);
    let denom: f64 = (0..n).map(|k| x.powi(-(k as i32))).fold(0.0, |s, v| s + v);
    pos.iter()
        .map(|&p| n as f64 * x.powi(p as i32 - (n as i32 - 1)) / denom)
        .collect()
}

/// Rank-based fitness. Ties share the rank of the worst member of the tie group.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, rm=None, sp=None, mask=None))]
pub fn ranking<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Opt<'_, 'py>,
    maxormins: Opt<'_, 'py>,
    rm: Opt<'_, 'py>,
    sp: Option<f64>,
    mask: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // RM is compared as a number (1.0 and True are 1); anything else is refused below
    let rm: Option<i64> = match rm.filter(|o| !o.is_none()) {
        None => None,
        Some(o) => match o.extract::<f64>() {
            Ok(v) if v == 0.0 || v == 1.0 => Some(v as i64),
            _ => Some(-1),
        },
    };
    let f = prepared_column(
        obj_v,
        cv,
        maxormins,
        "error in ranking: ObjV must be a n*1 Numpy ndarray. (目标函数矩阵ObjV必须是n行1列的Numpy ndarray类型矩阵。)",
    )?;
    let n = f.len();
    // RM (called PM in the original messages) must be 0 or 1 whenever SP is given
    if (sp.is_some() && rm.is_none()) || rm.is_some_and(|r| r != 0 && r != 1) {
        return Err(runtime_err(
            "error in ranking: PM must be in 0 or 1. (PM必须是0或1。)",
        ));
    }
    if let (Some(r), Some(v)) = (rm, sp) {
        if r == 0 && !(1.0..=2.0).contains(&v) {
            return Err(runtime_err("error in ranking: SP must be in [1,2] when PM = 0. (当PM为0时，SP必须在[1,2]范围内，详见ranking帮助文档。)"));
        }
        if r == 1 && !(v >= 1.0 && v <= n as f64 - 2.0) {
            return Err(runtime_err("error in ranking: SP must be in [1,Nind] when PM = 1, Nind is the number of individuals. (当PM为1时，SP必须在[1,Nind]范围内，Nind为种群规模，详见ranking帮助文档。)"));
        }
    }
    // position counted from the worst individual (0) to the best (n - 1): the number of values
    // greater than x (none for NaN, and NaN is greater than nothing), found in a sorted copy
    let mut sorted: Vec<f64> = f.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| crate::utils::cmp_nan_last(*a, *b));
    let pos: Vec<usize> = f
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0
            } else {
                sorted.len() - sorted.partition_point(|&y| y <= x)
            }
        })
        .collect();
    let fit: Vec<f64> = if let Some(m) = mask.filter(|m| !m.is_none()) {
        let m = crate::utils::to_f64_array1(m)?;
        if m.len() != n {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "error in ranking: The length of Mask must equal Nind.",
            ));
        }
        pos.iter().map(|&p| m[p]).collect()
    } else if n <= 1 {
        vec![1.0; n]
    } else if rm.unwrap_or(0) == 1 {
        let sp = sp.unwrap_or(2.0);
        let x = nonlinear_root(sp, n);
        let denom: f64 = (0..n).map(|i| x.powi(i as i32)).fold(0.0, |s, x| s + x);
        let fit: Vec<f64> = pos
            .iter()
            .map(|&p| n as f64 * x.powi(p as i32) / denom)
            .collect();
        if fit.iter().all(|v| v.is_finite()) {
            fit
        } else {
            nonlinear_fitness_scaled(sp, n, &pos)
        }
    } else {
        let sp = sp.unwrap_or(2.0);
        pos.iter()
            .map(|&p| 2.0 - sp + 2.0 * (sp - 1.0) * p as f64 / (n - 1) as f64)
            .collect()
    };
    Ok(column(fit).into_pyarray(py))
}

/// Linear scaling (Goldberg) of max(F) - F; the minimum fitness is kept non-negative.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, smul=None))]
pub fn scaling<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Opt<'_, 'py>,
    maxormins: Opt<'_, 'py>,
    smul: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let f = prepared_column(
        obj_v,
        cv,
        maxormins,
        "error in scaling: ObjV must be a n*1 Numpy ndarray. (种群的目标函数值矩阵必须是n行1列的Numpy ndarray类型矩阵。)",
    )?;
    let n = f.len();
    let smul = smul.unwrap_or(2.0);
    if smul < 2.0 {
        return Err(runtime_err(
            "error in scaling: Smul must be or more than 2. (传入本函数的参数Smul必须不小于2。)",
        ));
    }
    let fmax = f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let g: Vec<f64> = f.iter().map(|&x| fmax - x).collect();
    if n == 0 || g.iter().all(|&x| x == g[0]) {
        return Ok(column(vec![1.0; n]).into_pyarray(py));
    }
    let avg = g.iter().fold(0.0, |s, x| s + x) / n as f64;
    let gmin = g.iter().cloned().fold(f64::INFINITY, f64::min);
    let gmax = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (a, b) = if gmin < (avg * smul - gmax) / (smul - 1.0) {
        let a = avg / (avg - gmin);
        (a, -gmin * a)
    } else {
        (
            (smul - 1.0) * avg / (gmax - avg),
            (gmax - avg * smul) * avg / (gmax - avg),
        )
    };
    Ok(column(g.iter().map(|&x| a * x + b).collect()).into_pyarray(py))
}

/// Power scaling: ((max - F) / (max - min)) ^ k.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, k=None))]
pub fn powing<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Opt<'_, 'py>,
    maxormins: Opt<'_, 'py>,
    k: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let f = prepared_column(
        obj_v,
        cv,
        maxormins,
        "error in powing: ObjV must be an n*1 array.  (目标函数矩阵ObjV必须是n行1列的Numpy ndarray类型矩阵。)",
    )?;
    let k = k.unwrap_or(1.0);
    if k <= 0.0 {
        return Err(runtime_err(
            "error in powing: The k must be more than 0. (传入本函数的参数k必须大于0。)",
        ));
    }
    let (mn, mx) = f
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let fit = if mx - mn <= 0.0 {
        vec![1.0; f.len()]
    } else {
        f.iter().map(|&x| ((mx - x) / (mx - mn)).powf(k)).collect()
    };
    Ok(column(fit).into_pyarray(py))
}

/// Exponential scaling: exp(-Beta * (F - min) / (max - min)).
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, beta=None))]
pub fn indexing<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Opt<'_, 'py>,
    maxormins: Opt<'_, 'py>,
    beta: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let f = prepared_column(
        obj_v,
        cv,
        maxormins,
        "error in indexing: ObjV must be a n*1 Numpy ndarray. (目标函数矩阵ObjV必须是n行1列的Numpy ndarray类型矩阵。)",
    )?;
    let beta = beta.unwrap_or(1.0);
    if beta <= 0.0 {
        return Err(runtime_err(
            "error in indexing: Beta must be more than 0. (Beta必须大于0。)",
        ));
    }
    let (mn, mx) = f
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let fit = if mx - mn <= 0.0 {
        vec![1.0; f.len()]
    } else {
        f.iter()
            .map(|&x| (-beta * (x - mn) / (mx - mn)).exp())
            .collect()
    };
    Ok(column(fit).into_pyarray(py))
}
