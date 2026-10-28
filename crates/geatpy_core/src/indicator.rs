use numpy::ndarray::Array2;
use pyo3::prelude::*;
use rand::Rng;

type Opt<'a, 'py> = Option<&'a Bound<'py, PyAny>>;

/// Parallel must be None, True or False, as geatpy 2.7.0 checks (the value itself is ignored).
fn check_parallel(parallel: Opt, name: &str) -> PyResult<()> {
    match parallel {
        Some(p) if !p.is_none() && !p.is_instance_of::<pyo3::types::PyBool>() => {
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "error in indicator.{name}: Parallel must be None, True or False. (传入参数Parallel必须是None,True或者False。)"
            )))
        }
        _ => Ok(()),
    }
}

fn pair(
    obj_v: &Bound<'_, PyAny>,
    pf: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Array2<f64>, Array2<f64>)> {
    let p = crate::utils::to_f64_array2(obj_v)?;
    let r = crate::utils::to_f64_array2(pf)?;
    if p.shape()[1] != r.shape()[1] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in indicator.{name}: ObjV and PF disagree. (ObjV和PF的列数必须相同。)"
        )));
    }
    Ok((p, r))
}

/// Euclidean distance summed left to right like geatpy 2.7.0's `_cdist_eu` (0.0 for no columns).
fn dist(a: &Array2<f64>, i: usize, b: &Array2<f64>, k: usize) -> f64 {
    (0..a.shape()[1])
        .fold(0.0, |s, j| s + (a[[i, j]] - b[[k, j]]).powi(2))
        .sqrt()
}

/// Smallest distance from row `i` of `a` to the rows of `b`: NaN distances are skipped unless all
/// of them are NaN (then NaN, as geatpy 2.7.0 gives), and an empty `b` is infinitely far away.
fn nearest(a: &Array2<f64>, i: usize, b: &Array2<f64>) -> f64 {
    if b.shape()[0] == 0 {
        return f64::INFINITY;
    }
    (0..b.shape()[0])
        .map(|k| dist(a, i, b, k))
        .fold(f64::NAN, f64::min)
}

/// Generational distance as geatpy 2.7.0 computes it: the norm of the nearest distances (squared
/// back after the square root, summed in order) divided by N.
#[pyfunction]
#[pyo3(signature = (ObjV, PF, params2=None, Parallel=None))]
#[allow(non_snake_case)]
pub fn GD(
    ObjV: &Bound<'_, PyAny>,
    PF: &Bound<'_, PyAny>,
    params2: Opt,
    Parallel: Opt,
) -> PyResult<f64> {
    let _ = params2;
    check_parallel(Parallel, "GD")?;
    let (p, r) = pair(ObjV, PF, "GD")?;
    let n = p.shape()[0];
    if n == 0 {
        return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
            "float division by zero",
        ));
    }
    let s = (0..n).fold(0.0, |s, i| {
        let d = nearest(&p, i, &r);
        s + d * d
    });
    Ok(s.sqrt() / n as f64)
}

/// Inverted generational distance: mean distance from each PF point to its nearest solution.
#[pyfunction]
#[pyo3(signature = (ObjV, PF, Parallel=None))]
#[allow(non_snake_case)]
pub fn IGD(ObjV: &Bound<'_, PyAny>, PF: &Bound<'_, PyAny>, Parallel: Opt) -> PyResult<f64> {
    check_parallel(Parallel, "IGD")?;
    let (p, r) = pair(ObjV, PF, "IGD")?;
    let nr = r.shape()[0];
    // the mean over no PF points is NaN, as in the original
    let s = (0..nr).fold(0.0, |s, k| s + nearest(&r, k, &p));
    Ok(s / nr as f64)
}

/// Schott's spacing as geatpy 2.7.0 computes it: `np.sqrt(np.var(d) * N / (N - 1))` of the
/// nearest-neighbour Manhattan distances d, returned as numpy.float64 (NaN for one point).
#[pyfunction]
#[pyo3(signature = (ObjV, params1=None, Parallel=None))]
#[allow(non_snake_case)]
pub fn Spacing<'py>(
    py: Python<'py>,
    ObjV: &Bound<'py, PyAny>,
    params1: Opt,
    Parallel: Opt,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::IntoPyArray;
    let _ = params1;
    check_parallel(Parallel, "Spacing")?;
    let p = crate::utils::to_f64_array2(ObjV)?;
    let (n, m) = (p.shape()[0], p.shape()[1]);
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "zero-size array to reduction operation minimum which has no identity",
        ));
    }
    // np.min over each row of the distance matrix with an infinite diagonal (NaN propagates); a row
    // of L1 distances is accumulated for all k at once from a column-major copy, so the loop over k
    // vectorises (every distance is still the sum from 0.0 in order of j)
    let pt = crate::utils::transpose(&p);
    let mut row = vec![0.0; n];
    let d: Vec<f64> = (0..n)
        .map(|i| {
            row.fill(0.0);
            for (j, col) in pt.chunks_exact(n).enumerate().take(m) {
                let x = p[[i, j]];
                for (o, &y) in row.iter_mut().zip(col) {
                    *o += (x - y).abs();
                }
            }
            row[i] = f64::INFINITY;
            if row.iter().any(|x| x.is_nan()) {
                f64::NAN
            } else {
                row.iter().copied().fold(f64::INFINITY, f64::min)
            }
        })
        .collect();
    let np = py.import("numpy")?;
    let var = np.call_method1("var", (numpy::ndarray::Array1::from(d).into_pyarray(py),))?;
    let ratio = var.mul(n)?.div(n as i64 - 1)?;
    np.call_method1("sqrt", (ratio,))
}

/// Exact hypervolume (minimisation) of `pts` against `refp` by slicing along the last objective.
fn hv_exact(mut pts: Vec<Vec<f64>>, refp: &[f64]) -> f64 {
    let m = refp.len();
    pts.retain(|p| p.iter().zip(refp).all(|(a, r)| a < r));
    if pts.is_empty() {
        return 0.0;
    }
    if m == 1 {
        return refp[0] - pts.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    }
    if m == 2 {
        pts.sort_by(|a, b| crate::utils::cmp_nan_last(a[0], b[0]));
        let mut vol = 0.0;
        let mut y = refp[1];
        for p in &pts {
            if p[1] < y {
                vol += (refp[0] - p[0]) * (y - p[1]);
                y = p[1];
            }
        }
        return vol;
    }
    pts.sort_by(|a, b| crate::utils::cmp_nan_last(a[m - 1], b[m - 1]));
    let mut vol = 0.0;
    for i in 0..pts.len() {
        let upper = if i + 1 < pts.len() {
            pts[i + 1][m - 1]
        } else {
            refp[m - 1]
        };
        let depth = upper - pts[i][m - 1];
        if depth > 0.0 {
            let slice: Vec<Vec<f64>> = pts[..=i].iter().map(|p| p[..m - 1].to_vec()).collect();
            vol += depth * hv_exact(slice, &refp[..m - 1]);
        }
    }
    vol
}

/// Total order on f64 for the staircase keys (no NaN reaches it).
#[derive(Clone, Copy, PartialEq)]
struct Key(f64);

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Exact 3-D hypervolume by a sweep along the third objective: the points are inserted in order
/// of z into the 2-D staircase of the first two objectives, whose dominated area is kept up to
/// date, so each slab between consecutive z values costs O(log n) instead of a new 2-D sweep.
fn hv3_sweep(pts: &mut [[f64; 3]], refp: &[f64]) -> f64 {
    use std::collections::BTreeMap;
    use std::ops::Bound::{Excluded, Included, Unbounded};
    pts.sort_by(|a, b| crate::utils::cmp_nan_last(a[2], b[2]));
    let (rx, ry) = (refp[0], refp[1]);
    // non-dominated (x, y) so far: x increasing, y decreasing
    let mut stairs: BTreeMap<Key, f64> = BTreeMap::new();
    let (mut area, mut vol) = (0.0, 0.0);
    for (i, p) in pts.iter().enumerate() {
        let (x, y) = (p[0], p[1]);
        // the point with the largest x <= x has the smallest y among them
        let covered = stairs
            .range(..=Key(x))
            .next_back()
            .is_some_and(|(_, &qy)| qy <= y);
        if !covered {
            let mut cur_y = stairs.range(..Key(x)).next_back().map_or(ry, |(_, &qy)| qy);
            let mut cur_x = x;
            // remove the points p dominates (x' >= x, y' >= y), adding the strips p now covers
            let dominated: Vec<(Key, f64)> = stairs
                .range((Included(Key(x)), Unbounded))
                .take_while(|(_, &qy)| qy >= y)
                .map(|(&k, &v)| (k, v))
                .collect();
            for (k, qy) in &dominated {
                area += (k.0 - cur_x) * (cur_y - y);
                cur_x = k.0;
                cur_y = *qy;
                stairs.remove(k);
            }
            let end = stairs
                .range((Excluded(Key(x)), Unbounded))
                .next()
                .map_or(rx, |(k, _)| k.0);
            area += (end - cur_x) * (cur_y - y);
            stairs.insert(Key(x), y);
        }
        let upper = pts.get(i + 1).map_or(refp[2], |q| q[2]);
        vol += area * (upper - p[2]);
    }
    vol
}

/// Exact 4-D hypervolume: slices along the fourth objective, each the 3-D hypervolume of the points
/// below it (O(n^2 log n) in all; slicing down to two dimensions was O(n^3 log n)).
fn hv4(mut pts: Vec<Vec<f64>>, refp: &[f64]) -> f64 {
    pts.retain(|p| p.iter().zip(refp).all(|(a, r)| a < r));
    pts.sort_by(|a, b| crate::utils::cmp_nan_last(a[3], b[3]));
    let mut slice: Vec<[f64; 3]> = Vec::with_capacity(pts.len());
    let mut vol = 0.0;
    for i in 0..pts.len() {
        let upper = pts.get(i + 1).map_or(refp[3], |q| q[3]);
        let depth = upper - pts[i][3];
        if depth > 0.0 {
            slice.clear();
            slice.extend(pts[..=i].iter().map(|p| [p[0], p[1], p[2]]));
            vol += depth * hv3_sweep(&mut slice, &refp[..3]);
        }
    }
    vol
}

/// Fraction of `samples` uniform points of the box [min(pts), refp] that some point weakly
/// dominates, times the box volume. A sample is drawn as m consecutive uniforms (one per objective).
/// Which point dominates a sample does not matter, so points dominated by another point are dropped
/// first and the rest are sorted by the first objective: only the prefix with p[0] <= s[0] can
/// dominate s. The estimate is the same as testing every point against every sample.
fn hv_monte_carlo(pts: &[Vec<f64>], refp: &[f64], samples: usize) -> f64 {
    let m = refp.len();
    let lo: Vec<f64> = (0..m)
        .map(|j| pts.iter().map(|p| p[j]).fold(f64::INFINITY, f64::min))
        .collect();
    let box_vol: f64 = (0..m).map(|j| refp[j] - lo[j]).product();
    let weakly = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x <= y);
    // drop points another point weakly dominates (of equal points the first one stays)
    let mut kept: Vec<&[f64]> = pts
        .iter()
        .enumerate()
        .filter(|&(i, p)| {
            !pts.iter()
                .enumerate()
                .any(|(k, q)| k != i && weakly(q, p) && (k < i || !weakly(p, q)))
        })
        .map(|(_, p)| p.as_slice())
        .collect();
    kept.sort_by(|a, b| crate::utils::cmp_nan_last(a[0], b[0]));
    let first: Vec<f64> = kept.iter().map(|p| p[0]).collect();
    let rest: Vec<f64> = kept.iter().flat_map(|p| p[1..].iter().copied()).collect();
    let mut rng = crate::utils::rng();
    let mut s = vec![0.0; m];
    let mut hits = 0usize;
    for _ in 0..samples {
        for (j, x) in s.iter_mut().enumerate() {
            *x = lo[j] + rng.gen::<f64>() * (refp[j] - lo[j]);
        }
        let k = first.partition_point(|&a| a <= s[0]);
        if rest[..k * (m - 1)]
            .chunks_exact(m - 1)
            .any(|p| weakly(p, &s[1..]))
        {
            hits += 1;
        }
    }
    box_vol * hits as f64 / samples as f64
}

/// Hypervolume of ObjV normalised like geatpy 2.7.0: each objective is mapped by
/// (x - lo) / (hi - lo) with lo = min(0, min of the ObjV column) and hi = 1.1 * max of the PF
/// column (PF defaults to ObjV); the reference point is 1 on every axis and points beyond it are
/// dropped. Exact for M <= 4 (M = 4 by 3-D sweeps), Monte Carlo with 100000 samples above (as
/// geatpy 2.7.0).
#[pyfunction]
#[pyo3(signature = (ObjV, PF=None, Parallel=None))]
#[allow(non_snake_case)]
pub fn HV(ObjV: &Bound<'_, PyAny>, PF: Opt, Parallel: Opt) -> PyResult<f64> {
    check_parallel(Parallel, "HV")?;
    let shape_error = || {
        pyo3::exceptions::PyRuntimeError::new_err(
            "error in indicator.HV: PF must be a matrix and has the same number of columns as ObjV which is more than 0. (PF必须是一个列数至少为1的矩阵且拥有和ObjV一样的列数。)",
        )
    };
    let ndim = |o: &Bound<'_, PyAny>| {
        o.getattr("ndim")
            .and_then(|v| v.extract::<usize>())
            .unwrap_or(2)
    };
    if ndim(ObjV) > 2 || PF.is_some_and(|x| !x.is_none() && ndim(x) > 2) {
        return Err(shape_error());
    }
    let p = crate::utils::to_f64_array2(ObjV)?;
    let r = match PF {
        Some(x) if !x.is_none() => crate::utils::to_f64_array2(x)?,
        _ => p.clone(),
    };
    let (n, m) = (p.shape()[0], p.shape()[1]);
    if m == 0 || r.shape()[1] != m {
        return Err(shape_error());
    }
    // no point, or no PF to scale by (the original reads uninitialised memory then)
    if n == 0 || r.shape()[0] == 0 {
        return Ok(0.0);
    }
    let lo: Vec<f64> = (0..m)
        .map(|j| p.column(j).iter().cloned().fold(0.0, f64::min))
        .collect();
    // NaN entries are skipped; a PF column with nothing but NaN leaves no scale (NaN, like the original)
    let hi: Vec<f64> = (0..m)
        .map(|j| 1.1 * r.column(j).iter().cloned().fold(f64::NAN, f64::max))
        .collect();
    if hi.iter().any(|h| h.is_nan()) {
        return Ok(f64::NAN);
    }
    let inside: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..m)
                .map(|j| (p[[i, j]] - lo[j]) / (hi[j] - lo[j]))
                .collect::<Vec<f64>>()
        })
        .filter(|q| q.iter().all(|&x| x < 1.0))
        .collect();
    if inside.is_empty() {
        return Ok(0.0);
    }
    let refp = vec![1.0; m];
    Ok(if m == 4 {
        hv4(inside, &refp)
    } else if m < 4 {
        hv_exact(inside, &refp)
    } else {
        // geatpy 2.7.0 draws 100000 samples
        hv_monte_carlo(&inside, &refp, 100_000)
    })
}
