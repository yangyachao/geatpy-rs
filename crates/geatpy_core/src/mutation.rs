use numpy::ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rand::Rng;

use crate::utils::{bounds, chrom_out, fix_value, is_integer_like, knob, per_gene, Knob};

type Opt<'a, 'py> = Option<&'a Bound<'py, PyAny>>;

fn require_ri(encoding: &str, name: &str) -> PyResult<()> {
    if encoding != "RI" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: The encoding must be 'RI'. (编码方式必须为'RI'。)",
            name
        )));
    }
    Ok(())
}

fn err(msg: String) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(msg)
}

/// Pm as a scalar or a per-gene array, every value in [0, 1] (geatpy 2.7.0 messages).
fn pm_vec(pm: Opt, d: usize, default: f64, name: &str) -> PyResult<Vec<f64>> {
    let is_array = matches!(knob(pm)?, Some(Knob::Array(_)));
    let v = per_gene(pm, d, default, "Pm")?;
    if v.iter().any(|&x| !(0.0..=1.0).contains(&x)) {
        return Err(err(if is_array {
            format!(
                "error in {}: Every element in Pm must be in [0, 1] if Pm is an array. (各变异概率Pm必须在[0, 1]之间。)",
                name
            )
        } else {
            format!(
                "error in {}: Pm must be in [0, 1]. (变异概率Pm必须在[0, 1]之间。)",
                name
            )
        }));
    }
    Ok(v)
}

fn default_pm(pm: Opt, d: usize, name: &str) -> PyResult<Vec<f64>> {
    pm_vec(pm, d, 1.0 / (d.max(1) as f64), name)
}

/// A Python/numpy integer (geatpy 2.7.0 rejects floats such as 3.0 for integer arguments).
fn int_value(o: &Bound<'_, PyAny>) -> Option<i64> {
    if o.is_instance_of::<pyo3::types::PyFloat>() || o.is_instance_of::<pyo3::types::PyBool>() {
        return None;
    }
    o.extract::<i64>().ok()
}

/// Polynomial mutation (Deb). Inputs are repaired first; the mutated value stays in range by construction.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, pm=None, dis_i=None, fix_type=None, parallel=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutpolyn<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    pm: Opt<'_, 'py>,
    dis_i: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params7);
    require_ri(encoding, "mutpolyn")?;
    // A 1-D OldChrom is a single chromosome (the original segfaults on it).
    let one_d = old_chrom
        .getattr("ndim")
        .and_then(|v| v.extract::<usize>())
        .map(|nd| nd == 1)
        .unwrap_or(false);
    let mut chrom = if one_d {
        crate::utils::to_f64_array1(old_chrom)?.insert_axis(numpy::ndarray::Axis(0))
    } else {
        crate::utils::to_f64_array2(old_chrom)?
    };
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let pm = default_pm(pm, d, "mutpolyn")?;
    let eta = per_gene(dis_i, d, 20.0, "DisI")?;
    let fix = crate::utils::fix_type(fix_type, "mutpolyn")?;
    let mut rng = crate::utils::rng();
    for i in 0..n {
        for j in 0..d {
            let mut x = if b.span[j] > 1e-15 {
                fix_value(chrom[[i, j]], b.lb[j], b.ub[j], b.span[j], fix, &mut rng)
            } else {
                b.lb[j]
            };
            if b.span[j] > 1e-15 && rng.gen::<f64>() < pm[j] {
                let e1 = eta[j] + 1.0;
                let u: f64 = rng.gen();
                if u > 0.5 {
                    let dv = 2.0 * u - 1.0;
                    let t = (1.0 - (b.ub[j] - x) / b.span[j]).powf(e1);
                    x += (1.0 - ((1.0 - dv) + dv * t).powf(1.0 / e1)) * b.span[j];
                } else {
                    let t = (1.0 - (x - b.lb[j]) / b.span[j]).powf(e1);
                    x += (((1.0 - 2.0 * u) * t + 2.0 * u).powf(1.0 / e1) - 1.0) * b.span[j];
                }
            }
            chrom[[i, j]] = if b.discrete[j] { x.round() } else { x };
        }
    }
    if one_d {
        let out = crate::utils::ri_out(py, chrom, &b.discrete)?;
        return Ok(out.bind(py).call_method0("ravel")?.unbind());
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    rng.sample(rand_distr::StandardNormal)
}

/// Gaussian mutation. Sigma3 is three standard deviations: scalar, per-gene array,
/// True (min distance to a bound) or False/None (0.5 * (ub - lb)).
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, pm=None, sigma3=None, middle=None, fix_type=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutgau<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    pm: Opt<'_, 'py>,
    sigma3: Opt<'_, 'py>,
    middle: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    require_ri(encoding, "mutgau")?;
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let pm = default_pm(pm, d, "mutgau")?;
    let fix = crate::utils::fix_type(fix_type, "mutgau")?;
    let middle = matches!(knob(middle)?, Some(Knob::Bool(true)));
    // None => per-gene sigma derived from bounds; Some(true) => adaptive; per-gene fixed sigma otherwise.
    let (adaptive, sigma): (bool, Option<Vec<f64>>) = match knob(sigma3)? {
        None | Some(Knob::Bool(false)) => (false, None),
        Some(Knob::Bool(true)) => (true, None),
        Some(Knob::Scalar(s)) => (
            false,
            Some(
                (0..d)
                    .map(|j| {
                        if b.discrete[j] {
                            (s + 0.499999) / 3.0
                        } else {
                            s / 3.0
                        }
                    })
                    .collect(),
            ),
        ),
        Some(Knob::Array(v)) => {
            if v.len() != d {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in mutgau: The length of Sigma3 must equal the chromosome length.",
                ));
            }
            (
                false,
                Some(
                    (0..d)
                        .map(|j| {
                            if b.discrete[j] {
                                (v[j] + 0.499999) / 3.0
                            } else {
                                v[j] / 3.0
                            }
                        })
                        .collect(),
                ),
            )
        }
    };
    let mut rng = crate::utils::rng();
    for i in 0..n {
        for j in 0..d {
            let mut x;
            if b.span[j] <= 1e-15 {
                x = b.lb[j];
            } else {
                x = chrom[[i, j]];
                if rng.gen::<f64>() < pm[j] {
                    let s = if adaptive {
                        (b.ub[j] - x).abs().min((x - b.lb[j]).abs()) / 3.0
                    } else {
                        match &sigma {
                            Some(v) => v[j],
                            None => (b.ub[j] - b.lb[j]) / 6.0,
                        }
                    };
                    if middle {
                        x = (b.lb[j] + b.ub[j]) * 0.5;
                    }
                    if s > 1e-15 {
                        x += s * standard_normal(&mut rng);
                    }
                }
                x = fix_value(x, b.lb[j], b.ub[j], b.span[j], fix, &mut rng);
            }
            chrom[[i, j]] = if b.discrete[j] { x.round() } else { x };
        }
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

/// Breeder GA mutation (Mühlenbein & Schlierkamp-Voosen).
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, pm=None, mut_shrink=None, gradient=None, fix_type=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutbga<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    pm: Opt<'_, 'py>,
    mut_shrink: Opt<'_, 'py>,
    gradient: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    require_ri(encoding, "mutbga")?;
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let pm = default_pm(pm, d, "mutbga")?;
    let shrink = per_gene(mut_shrink, d, 0.5, "MutShrink")?;
    if shrink.iter().any(|&x| !(0.0..=1.0).contains(&x)) {
        return Err(err(if matches!(knob(mut_shrink)?, Some(Knob::Array(_))) {
            "error in mutbga: Every element of MutShrink must be or bigger than 0 if MutShrink is an array. (当输入参数MutShrink是一个Numpy ndarray一维数组时，它的各个元素必须在[0, 1]之间。)".to_string()
        } else {
            "error in mutbga: MutShrink must be in [0, 1]. (当输入参数MutShrink为一个数时，它必须在[0, 1]之间。)".to_string()
        }));
    }
    // Gradient: a non-negative int, or an array of positive integers.
    let grad: Vec<i64> = match gradient.filter(|g| !g.is_none()) {
        None => vec![20; d],
        Some(g) => match int_value(g) {
            Some(v) if v < 0 => {
                return Err(err("error in mutbga: Gradient must be or bigger than 0. (当输入参数Gradient为一个数时，它必须不小于0。)".to_string()))
            }
            Some(v) => vec![v; d],
            None if g.hasattr("__len__")? => {
                let v = per_gene(Some(g), d, 20.0, "Gradient")?;
                if v.iter().any(|&x| x <= 0.0) {
                    return Err(err("error in mutbga: Every element of Gradient must be or bigger than 0 if Gradient is an array. (当输入参数Gradient是一个Numpy ndarray一维数组时，它的各个元素必须不小于0。)".to_string()));
                }
                v.iter().map(|&x| x as i64).collect()
            }
            None => {
                return Err(err("error in mutbga: The type of Gradient must be int or Numpy ndarray. (输入参数Gradient必须是int或者Numpy ndarray类型。)".to_string()))
            }
        },
    };
    let fix = crate::utils::fix_type(fix_type, "mutbga")?;
    // The mutation range doubles as the repair span, as in the reference implementation.
    let range: Vec<f64> = (0..d).map(|j| b.span[j] * shrink[j]).collect();
    let mut rng = crate::utils::rng();
    for i in 0..n {
        for j in 0..d {
            let mut x;
            if range[j] <= 1e-15 {
                x = b.lb[j];
            } else {
                x = chrom[[i, j]];
                if rng.gen::<f64>() < pm[j] {
                    let g = grad[j].max(1);
                    let p = 1.0 / g as f64;
                    let mut delta: f64 = (0..g)
                        .filter(|_| rng.gen::<f64>() < p)
                        .map(|k| 0.5f64.powi(k as i32))
                        .fold(0.0, |s, x| s + x);
                    delta = delta.max(0.5f64.powi((g - 1) as i32));
                    if rng.gen::<f64>() < 0.5 {
                        delta = -delta;
                    }
                    x += delta * range[j];
                }
                x = fix_value(x, b.lb[j], b.ub[j], range[j], fix, &mut rng);
            }
            chrom[[i, j]] = if b.discrete[j] { x.round() } else { x };
        }
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

/// Uniform mutation within a radius Alpha around the current value (or the domain centre).
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, pm=None, alpha=None, middle=None, fix_type=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutuni<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    pm: Opt<'_, 'py>,
    alpha: Opt<'_, 'py>,
    middle: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    require_ri(encoding, "mutuni")?;
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let pm = default_pm(pm, d, "mutuni")?;
    let fix = crate::utils::fix_type(fix_type, "mutuni")?;
    let middle = matches!(knob(middle)?, Some(Knob::Bool(true)));
    let (adaptive, radius): (bool, Option<Vec<f64>>) = match knob(alpha)? {
        None | Some(Knob::Bool(false)) => (false, None),
        Some(Knob::Bool(true)) => (true, None),
        Some(Knob::Scalar(a)) => (
            false,
            Some(
                (0..d)
                    .map(|j| if b.discrete[j] { a + 0.5 } else { a })
                    .collect(),
            ),
        ),
        Some(Knob::Array(v)) => {
            if v.len() != d {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in mutuni: The length of Alpha must equal the chromosome length.",
                ));
            }
            (
                false,
                Some(
                    (0..d)
                        .map(|j| if b.discrete[j] { v[j] + 0.5 } else { v[j] })
                        .collect(),
                ),
            )
        }
    };
    let mut rng = crate::utils::rng();
    for i in 0..n {
        for j in 0..d {
            let mut x;
            if b.span[j] <= 1e-15 {
                x = b.lb[j];
            } else {
                x = chrom[[i, j]];
                if rng.gen::<f64>() < pm[j] {
                    let r = if adaptive {
                        (b.ub[j] - x).abs().min((x - b.lb[j]).abs())
                    } else {
                        match &radius {
                            Some(v) => v[j],
                            None => (b.ub[j] - b.lb[j]) * 0.5,
                        }
                    };
                    if middle {
                        x = (b.lb[j] + b.ub[j]) * 0.5;
                    }
                    x = x - r + 2.0 * r * rng.gen::<f64>();
                }
                x = fix_value(x, b.lb[j], b.ub[j], b.span[j], fix, &mut rng);
            }
            chrom[[i, j]] = if b.discrete[j] { x.round() } else { x };
        }
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

/// Bit-flip mutation for binary chromosomes.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, params2=None, pm=None, parallel=None, params5=None, params6=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutbin<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    params2: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params5: Opt<'_, 'py>,
    params6: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (params2, parallel, params5, params6, params7);
    if encoding != "BG" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in mutbin: The encoding must be 'BG'. (编码方式必须为'BG'。)",
        ));
    }
    // A 1-D OldChrom is a single chromosome, as in mutpolyn (the original raises IndexError; read
    // as a column, every gene would be a chromosome of its own with Pm = 1 and flip).
    let one_d = old_chrom
        .getattr("ndim")
        .and_then(|v| v.extract::<usize>())
        .is_ok_and(|nd| nd == 1);
    if one_d {
        let row = old_chrom.call_method1("reshape", (1, -1))?;
        let out = mutbin(py, encoding, &row, None, pm, None, None, None, None)?;
        return Ok(out.bind(py).call_method0("ravel")?.unbind());
    }
    // integer chromosomes (the usual case) are flipped in place without a round trip through f64
    if let Some(mut chrom) = crate::utils::to_i32_array2(old_chrom) {
        let d = chrom.shape()[1];
        let pm = default_pm(pm, d, "mutbin")?;
        let flat = chrom.as_slice_mut().expect("owned arrays are contiguous");
        for k in flip_positions(flat.len(), d, &pm) {
            flat[k] = 1 - flat[k];
        }
        return Ok(chrom.into_pyarray(py).into_any().unbind());
    }
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let d = chrom.shape()[1];
    let pm = default_pm(pm, d, "mutbin")?;
    let flat = chrom.as_slice_mut().expect("owned arrays are contiguous");
    for k in flip_positions(flat.len(), d, &pm) {
        flat[k] = 1.0 - flat[k];
    }
    chrom_out(py, chrom, true)
}

/// Row-major indices of the genes that flip, each gene independently with probability pm[column].
fn flip_positions(total: usize, d: usize, pm: &[f64]) -> Vec<usize> {
    let mut rng = crate::utils::rng();
    let mut hits = Vec::new();
    if d == 0 {
        return hits;
    }
    if pm.iter().all(|&p| p == pm[0]) {
        crate::utils::for_each_hit(total, pm[0], &mut rng, |k| hits.push(k));
    } else {
        for k in 0..total {
            if rng.gen::<f64>() < pm[k % d] {
                hits.push(k);
            }
        }
    }
    hits
}

/// Draws a donor index from 0..n uniformly without replacement (`taken` holds the draws since the
/// pool was last filled; `skip`, the target row, is excluded until then). Like geatpy 2.7.0, once
/// every candidate is taken the pool is refilled with every row, the target included, and drawing
/// goes on without replacement. Rejection sampling keeps each draw O(1) (a candidate pool of length
/// n per individual made mutde O(n^2)) with the same distribution as drawing from that pool. `n` > 0.
fn draw<R: Rng>(taken: &mut Vec<usize>, skip: &mut Option<usize>, n: usize, rng: &mut R) -> usize {
    if taken.len() + usize::from(skip.is_some()) >= n {
        taken.clear();
        *skip = None;
    }
    loop {
        let k = rng.gen_range(0..n);
        if Some(k) != *skip && !taken.contains(&k) {
            taken.push(k);
            return k;
        }
    }
}

enum Xr {
    Random,
    Index(Vec<usize>),
    Vector(Array2<f64>),
}

/// Differential mutation: Xr0 + F1 (Xr1 - Xr2) [+ F2 (Xr3 - Xr4)].
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, xr_list=None, f=None, fix_type=None, mask_n=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutde<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    xr_list: Opt<'_, 'py>,
    f: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    mask_n: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    require_ri(encoding, "mutde")?;
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let fix = crate::utils::fix_type(fix_type, "mutde")?;

    // Output rows, and an optional element mask.
    let mut rows = n;
    let mut mask: Option<Array2<bool>> = None;
    if let Some(m) = mask_n.filter(|m| !m.is_none()) {
        if m.is_instance_of::<pyo3::types::PyBool>() && m.extract::<bool>()? {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "an integer is required",
            ));
        }
        if let Ok(arr) = m.downcast::<numpy::PyUntypedArray>() {
            use numpy::{PyArrayDescrMethods, PyUntypedArrayMethods};
            if arr.ndim() != 2 {
                return Err(err("error in mutde: The dimension of Mask_N must be 2 when it is a Numpy ndarray. (当传入参数Mask_N是Numpy ndarray类型时，它必须是2维的。)".to_string()));
            }
            if !arr.dtype().is_equiv_to(&numpy::dtype::<bool>(py)) {
                return Err(err("error in mutde: The element of Mask_N must be True or False when Mask_N is a Numpy ndarray. (当传入参数Mask_N是Numpy ndarray类型时，它的元素必须是True或False。)".to_string()));
            }
            let arr = crate::utils::to_f64_array2(m)?;
            if arr.shape()[1] != d {
                return Err(err(
                    "error in mutde: The number of columns of Mask_N must equal that of OldChrom."
                        .to_string(),
                ));
            }
            rows = arr.shape()[0];
            mask = Some(arr.mapv(|x| x != 0.0));
        } else if let Ok(k) = m.extract::<i64>() {
            if k <= 0 {
                return Err(err("error in mutde: Mask_N must be bigger than 0 when it is a integer. (当传入参数Mask_N是int类型时，它必须是正整数。)".to_string()));
            }
            rows = k as usize;
        } else {
            return Err(err(
                "error in mutde: Mask_N type error. (传入参数Mask_N的类型错误，详见API文档。)"
                    .to_string(),
            ));
        }
    }

    // XrList: a list of at most 5 entries, each None, a 2-D vector matrix Xr or a 1-D index array r
    // (geatpy 2.7.0's checks and messages; its out-of-range reads on short Xr are refused instead).
    const ROLE: [&str; 5] = ["基向量", "差分量", "差分量", "差分量", "差分量"];
    const SEQ: [&str; 5] = ["基向量", "差分向量", "差分向量", "差分向量", "差分向量"];
    let mut xrs: Vec<Xr> = Vec::new();
    if let Some(list) = xr_list.filter(|l| !l.is_none()) {
        let list = list.downcast::<pyo3::types::PyList>().map_err(|_| {
            err("error in mutde: XrList must be a list. (XrList必须是list类型。)".to_string())
        })?;
        if list.len() > 5 {
            return Err(err("error in mutde: The length of XrList must not be over than 5. (输入参数XrList所包含的元素个数必须不超过5个)".to_string()));
        }
        for (k, item) in list.iter().enumerate() {
            if item.is_none() {
                xrs.push(Xr::Random);
                continue;
            }
            let ndim = item
                .downcast::<numpy::PyUntypedArray>()
                .map(|a| {
                    use numpy::PyUntypedArrayMethods;
                    a.ndim()
                })
                .unwrap_or(0);
            if ndim != 1 && ndim != 2 {
                return Err(err(format!("error in mutde: Data error of the parameter r{k}(or Xr{k}). ({}索引r{k}或{}矩阵Xr{k}的数据格式不正确，详见help(mutde)帮助文档。)", ROLE[k], if k == 0 { "基向量" } else { "差分向量" })));
            }
            if ndim == 2 {
                let v = crate::utils::to_f64_array2(&item)?;
                if v.shape()[0] < rows || v.shape()[1] != d {
                    return Err(err(
                        "error in mutde: The shape of a vector in XrList does not match."
                            .to_string(),
                    ));
                }
                xrs.push(Xr::Vector(v));
                continue;
            }
            let v = crate::utils::to_f64_array1(&item)?;
            if v.len() != rows {
                return Err(err(if k == 1 {
                    "error in mutde: r1 and OldChrom disagree. (差分向量索引序列r1的长度必须和种群规模相等。)".to_string()
                } else {
                    format!("error in mutde: The length of r{k} must be equal to others. (向量索引序列长度不一致，如果Mask_N为int，则传入的索引序列长度必须等于Mask_N。)")
                }));
            }
            let idx: Vec<usize> = v
                .iter()
                .map(|&x| x as i64)
                .map(|x| {
                    if x < 0 || x as usize >= n {
                        usize::MAX
                    } else {
                        x as usize
                    }
                })
                .collect();
            if idx.contains(&usize::MAX) {
                // the original has no message of its own for r1 and reuses the length one
                return Err(err(if k == 1 {
                    "error in mutde: The length of r1 must be equal to others. (向量索引序列长度不一致，如果Mask_N为int，则传入的索引序列长度必须等于Mask_N。)".to_string()
                } else {
                    format!("error in mutde: Some elements of r{k} is illegal. ({}索引序列r{k}的某些元素不合法。)", SEQ[k])
                }));
            }
            xrs.push(Xr::Index(idx));
        }
    }
    let pairs = if xrs.len() > 3 { 2 } else { 1 };
    while xrs.len() < 1 + 2 * pairs {
        xrs.push(Xr::Random);
    }
    // geatpy 2.7.0 crashes (division by zero) when it has to draw donors from no individuals.
    let draws = xrs
        .iter()
        .take(1 + 2 * pairs)
        .any(|x| matches!(x, Xr::Random));
    let produces = mask.as_ref().map_or(rows > 0, |m| m.iter().any(|&on| on));
    if n == 0 && draws && produces {
        return Err(err(
            "error in mutde: OldChrom must not be empty when donors are drawn at random."
                .to_string(),
        ));
    }

    // F: scalar, [F1, F2] with None meaning "random in (0, 1) per individual", or None.
    let parse_f = |o: &Bound<'py, PyAny>| -> PyResult<Option<f64>> {
        if o.is_none() {
            Ok(None)
        } else {
            Ok(Some(o.extract::<f64>()?))
        }
    };
    let (fa_opt, fb_opt) = match f {
        None => (None, None),
        Some(o) if o.is_none() => (None, None),
        Some(o) => {
            if let Ok(v) = o.extract::<f64>() {
                (Some(v), Some(v))
            } else if o.is_instance_of::<pyo3::types::PyList>() || o.is_instance_of::<PyTuple>() {
                let items: Vec<Bound<'py, PyAny>> = o.try_iter()?.collect::<PyResult<_>>()?;
                let a = items.first().map(parse_f).transpose()?.flatten();
                let c = match items.get(1) {
                    Some(x) => parse_f(x)?,
                    None => a,
                };
                (a, c)
            } else {
                return Err(err("error in mutde: F must be a scalar or a list. (输入参数F必须是一个标量或者list列表。)".to_string()));
            }
        }
    };
    if fa_opt.is_some_and(|x| x < 0.0) || fb_opt.is_some_and(|x| x < 0.0) {
        return Err(err(
            "error in mutde: All the F must be a bigger than 0. (所有的缩放因子F必须大于0。)"
                .to_string(),
        ));
    }
    // a negative value marks "draw F at random"
    let (f1, f2) = (fa_opt.unwrap_or(-1.0), fb_opt.unwrap_or(-1.0));

    let mut rng = crate::utils::rng();
    // Rows as slices (the converted matrices are row-major).
    let flat = chrom.as_slice().expect("converted arrays are row-major");
    let row_of = |k: usize| &flat[k * d..(k + 1) * d];
    let k_vec = 1 + 2 * pairs;
    // Donor rows and scale factors of row i, freshly drawn (the target row is excluded from the
    // donors when one offspring is produced per individual).
    let donors = |i: usize, rng: &mut rand::rngs::SmallRng| {
        let mut skip = (rows == n).then_some(i);
        let mut taken = Vec::with_capacity(k_vec);
        let mut vecs: [&[f64]; 5] = [&[]; 5];
        for (slot, x) in vecs.iter_mut().zip(xrs.iter().take(k_vec)) {
            *slot = match x {
                Xr::Random => row_of(draw(&mut taken, &mut skip, n, rng)),
                Xr::Index(idx) => row_of(idx[i]),
                Xr::Vector(v) => {
                    let w = v.ncols();
                    &v.as_slice().expect("converted arrays are row-major")[i * w..(i + 1) * w]
                }
            };
        }
        let fa = if f1 < 0.0 { rng.gen::<f64>() } else { f1 };
        let fb = if f2 < 0.0 { rng.gen::<f64>() } else { f2 };
        (vecs, fa, fb)
    };
    let repair = |v: f64, j: usize, rng: &mut rand::rngs::SmallRng| {
        let x = if b.span[j] > 1e-15 {
            fix_value(v, b.lb[j], b.ub[j], b.span[j], fix, rng)
        } else {
            b.lb[j]
        };
        if b.discrete[j] {
            x.round()
        } else {
            x
        }
    };
    if let Some(m) = mask {
        // Mask_N matrix: like geatpy 2.7.0, every selected element gets its own donors and F.
        let vals: Vec<f64> = m
            .indexed_iter()
            .filter(|(_, &on)| on)
            .map(|((i, j), _)| {
                let (v, fa, fb) = donors(i, &mut rng);
                let mut x = v[0][j] + fa * (v[1][j] - v[2][j]);
                if pairs == 2 {
                    x += fb * (v[3][j] - v[4][j]);
                }
                repair(x, j, &mut rng)
            })
            .collect();
        // int32 when every variable is an integer, like the matrix output
        if !b.discrete.is_empty() && b.discrete.iter().all(|&x| x) {
            let ints: Vec<i32> = vals.iter().map(|&x| x.round() as i32).collect();
            return Ok(Array1::from(ints).into_pyarray(py).into_any().unbind());
        }
        return Ok(Array1::from(vals).into_pyarray(py).into_any().unbind());
    }
    crate::utils::check_alloc(&[rows, d], 8, "float64")?;
    let mut out = Array2::<f64>::zeros((rows, d));
    // Truncation (the default) draws no random numbers, so its repair is written without calls and
    // the compiler vectorises it; f64::clamp's lb <= ub assertion is checked once for all genes and
    // the comparisons below equal clamp when it holds. The other repairs go gene by gene.
    let truncate = fix == 1 && b.lb.iter().zip(&b.ub).all(|(l, u)| l <= u);
    // With no genes there is nothing to compute (the generator is seeded per call, so skipping
    // the draws is not observable), and a huge integer Mask_N must not loop for nothing.
    if d > 0 {
        let flat_out = out.as_slice_mut().expect("fresh arrays are row-major");
        for (i, row) in flat_out.chunks_exact_mut(d).enumerate() {
            let (v, fa, fb) = donors(i, &mut rng);
            for (((o, &x0), &x1), &x2) in row.iter_mut().zip(v[0]).zip(v[1]).zip(v[2]) {
                *o = x0 + fa * (x1 - x2);
            }
            if pairs == 2 {
                for ((o, &x3), &x4) in row.iter_mut().zip(v[3]).zip(v[4]) {
                    *o += fb * (x3 - x4);
                }
            }
            if truncate {
                let genes = b.lb.iter().zip(&b.ub).zip(&b.span).zip(&b.discrete);
                for (o, (((&l, &u), &s), &disc)) in row.iter_mut().zip(genes) {
                    // a degenerate range collapses to lb, like `repair`
                    let x = if s <= 1e-15 || *o < l {
                        l
                    } else if *o > u {
                        u
                    } else {
                        *o
                    };
                    *o = if disc { x.round() } else { x };
                }
            } else {
                for (j, o) in row.iter_mut().enumerate() {
                    *o = repair(*o, j, &mut rng);
                }
            }
        }
    }
    crate::utils::ri_out(py, out, &b.discrete)
}

/// Segment length in [lo, max_len] such that every (start, length) pair is equally likely.
fn segment_len<R: Rng>(l: usize, max_len: usize, lo: usize, rng: &mut R) -> usize {
    let max_len = max_len.clamp(lo, l.max(lo));
    let total = segment_total(l, max_len, lo);
    if total == 0 {
        return lo;
    }
    segment_pick(l, max_len, lo, rng.gen_range(0..total))
}

/// Weight of length k is l + 1 - k (the number of places a segment of that length fits).
fn segment_weight(l: usize, k: usize) -> usize {
    l + 1 - k.min(l)
}

/// Sum of the weights of lengths lo..=max_len (an arithmetic series when lo <= l).
fn segment_total(l: usize, max_len: usize, lo: usize) -> usize {
    if lo <= l {
        let t = max_len - lo + 1;
        t * segment_weight(l, lo) - t * (t - 1) / 2
    } else {
        (lo..=max_len).map(|k| segment_weight(l, k)).sum()
    }
}

/// The length whose cumulative weight first exceeds r: the smallest count t with
/// S(t) = a*t - t*(t-1)/2 > r (a = weight of lo), from the quadratic formula and then stepped to
/// the exact integer, so it equals scanning the weights one by one.
fn segment_pick(l: usize, max_len: usize, lo: usize, r: usize) -> usize {
    if lo > l {
        let mut r = r;
        for k in lo..=max_len {
            let w = segment_weight(l, k);
            if r < w {
                return k;
            }
            r -= w;
        }
        return max_len;
    }
    let a = segment_weight(l, lo);
    let count = max_len - lo + 1;
    let s = |t: usize| t * a - t * t.saturating_sub(1) / 2;
    let b = 2.0 * a as f64 + 1.0;
    let disc = (b * b - 8.0 * r as f64).max(0.0);
    let mut t = (((b - disc.sqrt()) / 2.0).floor().max(0.0) as usize + 1).min(count);
    while t < count && s(t) <= r {
        t += 1;
    }
    while t > 1 && s(t - 1) > r {
        t -= 1;
    }
    lo + t - 1
}

/// Parses InvertLen / MoveLen: int => maximum length, [k] => fixed length, None => default. Both forms
/// must hold an integer of at least `min` (geatpy 2.7.0 checks and messages).
fn parse_len(obj: Opt, default: usize, op: &str, arg: &str, min: i64) -> PyResult<(usize, bool)> {
    let Some(o) = obj.filter(|o| !o.is_none()) else {
        return Ok((default, false));
    };
    if let Some(v) = int_value(o) {
        if v < min {
            return Err(err(format!(
                "error in {0}: When {1} is a scalar, it must be a integer bigger than {2}. (当{1}是一个标量时，它必须是大于{2}的整数。)",
                op, arg, min - 1
            )));
        }
        return Ok((v as usize, false));
    }
    if let Ok(list) = o.downcast::<pyo3::types::PyList>() {
        let k = if list.len() == 1 {
            int_value(&list.get_item(0)?)
        } else {
            None
        };
        return match k {
            Some(k) if k >= min => Ok((k as usize, true)),
            _ => Err(err(format!(
                "error in {0}: When {1} is a list, it must have only one element which is a integer that bigger than {2}. (当{1}是个list列表时，它必须只包含一个元素且该元素是个大于{2}的整数。)",
                op, arg, min - 1
            ))),
        };
    }
    Err(err(format!(
        "error in {0}: {1} illegal. (输入参数{1}不合法，详见help({0})帮助文档。)",
        op, arg
    )))
}

/// Permutation chromosomes shorter than 2 genes are returned unchanged, in their own dtype.
fn short_permutation<'py>(
    encoding: &str,
    old: &Bound<'py, PyAny>,
    l: usize,
) -> PyResult<Option<PyObject>> {
    if encoding == "P" && l < 2 {
        let np = old.py().import("numpy")?;
        return Ok(Some(np.call_method1("array", (old,))?.unbind()));
    }
    Ok(None)
}

/// "Random repair" of the order mutations on 'RI' chromosomes (geatpy 2.7.0 semantics): every gene is
/// checked against the bounds (integer variables widened by 0.499999), an out-of-range gene is redrawn
/// uniformly in the range, a degenerate range collapses to lb, and integer genes are rounded.
/// Returns the varTypes flags of FieldDR, or None when no FieldDR was given.
fn random_repair_rows<R: Rng>(
    chrom: &mut Array2<f64>,
    field: Opt,
    rng: &mut R,
) -> PyResult<Option<Vec<bool>>> {
    let Some(f) = field.filter(|f| !f.is_none()) else {
        return Ok(None);
    };
    let d = chrom.shape()[1];
    let b = bounds(&crate::utils::to_f64_array2(f)?, d, true)?;
    for mut row in chrom.rows_mut() {
        for j in 0..d {
            let x = row[j];
            let v = if b.span[j] <= 1e-15 {
                b.lb[j]
            } else if x < b.lb[j] || x > b.ub[j] {
                b.lb[j] + rng.gen::<f64>() * b.span[j]
            } else {
                x
            };
            row[j] = if b.discrete[j] { v.round() } else { v };
        }
    }
    Ok(Some(b.discrete))
}

/// Output of the order mutations: 'RI' follows varTypes (int32 when all integer), 'P' keeps the input type.
fn order_out(
    py: Python<'_>,
    chrom: Array2<f64>,
    discrete: Option<Vec<bool>>,
    as_int: bool,
) -> PyResult<PyObject> {
    match discrete {
        Some(d) => crate::utils::ri_out(py, chrom, &d),
        None => chrom_out(py, chrom, as_int),
    }
}

fn check_order_encoding(encoding: &str, name: &str) -> PyResult<()> {
    if encoding != "P" && encoding != "RI" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: The encoding must be 'P' or 'RI'.",
            name
        )));
    }
    Ok(())
}

/// Inversion mutation: reverse one segment of each chromosome with probability Pm.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr=None, pm=None, invert_len=None, parallel=None, params6=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutinv<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    invert_len: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params6: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params6, params7);
    check_order_encoding(encoding, "mutinv")?;
    let as_int = is_integer_like(old_chrom);
    // integer 'P' chromosomes are reversed in place as i32 (no round trip through f64)
    let int_chrom = if as_int && encoding == "P" {
        crate::utils::to_i32_array2(old_chrom)
    } else {
        None
    };
    let float_chrom = match int_chrom {
        Some(_) => None,
        None => Some(crate::utils::to_f64_array2(old_chrom)?),
    };
    let l = match (&int_chrom, &float_chrom) {
        (Some(c), _) => c.ncols(),
        (_, Some(c)) => c.ncols(),
        _ => unreachable!(),
    };
    let pm = pm_vec(pm, 1, 1.0, "mutinv")?[0];
    let (len, fixed) = parse_len(invert_len, l, "mutinv", "InvertLen", 2)?;
    if let Some(out) = short_permutation(encoding, old_chrom, l)? {
        return Ok(out);
    }
    let len = len.min(l);
    let mut rng = crate::utils::rng();
    if let Some(mut chrom) = int_chrom {
        invert_rows(&mut chrom, pm, len, fixed, &mut rng);
        return Ok(chrom.into_pyarray(py).into_any().unbind());
    }
    let mut chrom = float_chrom.expect("set when not integer");
    invert_rows(&mut chrom, pm, len, fixed, &mut rng);
    let discrete = if encoding == "RI" {
        random_repair_rows(&mut chrom, field_dr, &mut rng)?
    } else {
        None
    };
    order_out(py, chrom, discrete, as_int)
}

/// Reverses one segment of each row with probability pm (segment length: fixed, or drawn with
/// weights proportional to the number of places it fits).
fn invert_rows<T, R: Rng>(chrom: &mut Array2<T>, pm: f64, len: usize, fixed: bool, rng: &mut R) {
    let l = chrom.ncols();
    if l < 2 {
        return;
    }
    for mut row in chrom.rows_mut() {
        if rng.gen::<f64>() >= pm {
            continue;
        }
        let k = if fixed {
            len.max(2)
        } else {
            segment_len(l, len, 2, rng)
        };
        let start = rng.gen_range(0..=(l - k));
        for t in 0..k / 2 {
            row.swap(start + t, start + k - 1 - t);
        }
    }
}

/// Shift mutation: move one segment to another position, optionally reversing it with probability Pr.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr=None, pm=None, move_len=None, pr=None, parallel=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutmove<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    move_len: Opt<'_, 'py>,
    pr: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params7);
    check_order_encoding(encoding, "mutmove")?;
    let as_int = is_integer_like(old_chrom);
    // integer 'P' chromosomes are moved in place as i32 (no round trip through f64)
    let int_chrom = if as_int && encoding == "P" {
        crate::utils::to_i32_array2(old_chrom)
    } else {
        None
    };
    let float_chrom = match int_chrom {
        Some(_) => None,
        None => Some(crate::utils::to_f64_array2(old_chrom)?),
    };
    let l = match (&int_chrom, &float_chrom) {
        (Some(c), _) => c.ncols(),
        (_, Some(c)) => c.ncols(),
        _ => unreachable!(),
    };
    let pm = pm_vec(pm, 1, 1.0, "mutmove")?[0];
    let pr = crate::utils::per_gene(pr, 1, 0.0, "Pr")?[0];
    crate::utils::check_prob(
        pr,
        "mutmove",
        "Pr must be in [0, 1]. (逆转概率Pr必须在[0, 1]之间。)",
    )?;
    let (len, fixed) = parse_len(move_len, l, "mutmove", "MoveLen", 1)?;
    if let Some(out) = short_permutation(encoding, old_chrom, l)? {
        return Ok(out);
    }
    let len = len.min(l.saturating_sub(1)).max(1);
    let mut rng = crate::utils::rng();
    if let Some(mut chrom) = int_chrom {
        move_rows(&mut chrom, pm, len, fixed, pr, &mut rng);
        return Ok(chrom.into_pyarray(py).into_any().unbind());
    }
    let mut chrom = float_chrom.expect("set when not integer");
    move_rows(&mut chrom, pm, len, fixed, pr, &mut rng);
    let discrete = if encoding == "RI" {
        random_repair_rows(&mut chrom, field_dr, &mut rng)?
    } else {
        None
    };
    order_out(py, chrom, discrete, as_int)
}

/// Moves one segment of each row, with probability pm, to another of the l - k + 1 positions it can
/// take once removed (reversed first with probability pr), as rotations of the row in place.
fn move_rows<T, R: Rng>(
    chrom: &mut Array2<T>,
    pm: f64,
    len: usize,
    fixed: bool,
    pr: f64,
    rng: &mut R,
) {
    let l = chrom.ncols();
    if l < 2 {
        return;
    }
    for mut row in chrom.rows_mut() {
        if rng.gen::<f64>() >= pm {
            continue;
        }
        let k = if fixed {
            len
        } else {
            segment_len(l, len, 1, rng)
        };
        let start = rng.gen_range(0..=(l - k));
        let slots = l - k + 1;
        let dest = (start + 1 + rng.gen_range(0..(l - k).max(1))) % slots;
        let row = row.as_slice_mut().expect("converted arrays are row-major");
        if rng.gen::<f64>() < pr {
            row[start..start + k].reverse();
        }
        // the segment ends up at dest..dest + k of the row without it
        if dest <= start {
            row[dest..start + k].rotate_right(k);
        } else {
            row[start..dest + k].rotate_left(k);
        }
    }
}

/// Swap mutation: exchange two distinct genes of each chromosome with probability Pm.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr=None, pm=None, parallel=None, params5=None, params6=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutswap<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params5: Opt<'_, 'py>,
    params6: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params5, params6, params7);
    check_order_encoding(encoding, "mutswap")?;
    let as_int = is_integer_like(old_chrom);
    // integer 'P' chromosomes are swapped in place as i32 (no round trip through f64)
    let int_chrom = if as_int && encoding == "P" {
        crate::utils::to_i32_array2(old_chrom)
    } else {
        None
    };
    let float_chrom = match int_chrom {
        Some(_) => None,
        None => Some(crate::utils::to_f64_array2(old_chrom)?),
    };
    let l = match (&int_chrom, &float_chrom) {
        (Some(c), _) => c.ncols(),
        (_, Some(c)) => c.ncols(),
        _ => unreachable!(),
    };
    // geatpy 2.7.0 does not range-check Pm here
    let pm = crate::utils::per_gene(pm, 1, 1.0, "Pm")?[0];
    if let Some(out) = short_permutation(encoding, old_chrom, l)? {
        return Ok(out);
    }
    let mut rng = crate::utils::rng();
    if let Some(mut chrom) = int_chrom {
        swap_rows(&mut chrom, pm, &mut rng);
        return Ok(chrom.into_pyarray(py).into_any().unbind());
    }
    let mut chrom = float_chrom.expect("set when not integer");
    swap_rows(&mut chrom, pm, &mut rng);
    let discrete = if encoding == "RI" {
        random_repair_rows(&mut chrom, field_dr, &mut rng)?
    } else {
        None
    };
    order_out(py, chrom, discrete, as_int)
}

/// Exchanges two distinct genes of each row with probability pm.
fn swap_rows<T, R: Rng>(chrom: &mut Array2<T>, pm: f64, rng: &mut R) {
    let l = chrom.ncols();
    if l < 2 {
        return;
    }
    for mut row in chrom.rows_mut() {
        if rng.gen::<f64>() < pm {
            let a = rng.gen_range(0..l);
            let b = (a + 1 + rng.gen_range(0..l - 1)) % l;
            row.swap(a, b);
        }
    }
}

/// Permutation-point mutation: replace MutN genes by other values of [lb, ub],
/// swapping with the gene that already holds the new value.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr=None, pm=None, mut_n=None, parallel=None, params6=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutpp<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    mut_n: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params6: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params6, params7);
    if encoding != "P" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in mutpp: The encoding must be 'P'.",
        ));
    }
    let as_int = is_integer_like(old_chrom);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    for row in chrom.rows() {
        let mut v = row.to_vec();
        v.sort_by(|a, b| crate::utils::cmp_nan_last(*a, *b));
        if v.iter().any(|x| x.is_nan()) || v.windows(2).any(|w| w[0] == w[1]) {
            return Err(err(
                "error in mutpp: Chromosome error. (染色体不满足排列编码规则。)".to_string(),
            ));
        }
    }
    let (lo, hi) = match field_dr {
        Some(f) if !f.is_none() => {
            let a = crate::utils::to_f64_array2(f)?;
            // numpy's messages for FieldDR[0][0] and FieldDR[1][0]
            let (rows, cols) = a.dim();
            if rows < 2 || cols == 0 {
                let (index, size) = if cols == 0 { (0, 0) } else { (rows, rows) };
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "index {index} is out of bounds for axis 0 with size {size}"
                )));
            }
            let (lo, hi) = (a[[0, 0]].round() as i64, a[[1, 0]].round() as i64);
            if (l as i128) > i128::from(hi) - i128::from(lo) + 1 {
                return Err(err("error in mutpp: Lind must be or smaller than Ub - Lb + 1 for permutation, see the help. (编码长度Lind必须小于Ub - Lb + 1，详见help。)".to_string()));
            }
            if chrom.iter().any(|&x| x < lo as f64 || x > hi as f64) {
                return Err(err(
                    "error in mutpp: Chromosome error. (染色体不满足排列编码规则。)".to_string(),
                ));
            }
            (lo, hi)
        }
        _ => {
            let mn = chrom.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = chrom.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (mn as i64, mx as i64)
        }
    };
    // the value -> position table of a row has one slot per value of [lo, hi]
    let range = usize::try_from(i128::from(hi) - i128::from(lo) + 1)
        .unwrap_or(usize::MAX)
        .max(1);
    crate::utils::check_alloc(&[range], 16, "int64")?;
    let pm = pm_vec(pm, 1, 1.0 / l.max(1) as f64, "mutpp")?[0];
    let times = match mut_n.filter(|m| !m.is_none()) {
        None => 1,
        Some(m) => match int_value(m) {
            Some(k) if k > 0 => (k as usize).min(l),
            _ => {
                return Err(err(
                    "error in mutpp: MutN must be an integer bigger than 0. (传入参数MutN必须是正整数。)"
                        .to_string(),
                ))
            }
        },
    };
    let mut rng = crate::utils::rng();
    if l >= 1 && range >= 2 {
        for i in 0..n {
            if rng.gen::<f64>() >= pm {
                continue;
            }
            // position of each value in the chromosome, or None when absent
            let mut pos: Vec<Option<usize>> = vec![None; range];
            for j in 0..l {
                let v = (chrom[[i, j]] as i64 - lo) as usize;
                if v < range {
                    pos[v] = Some(j);
                }
            }
            for _ in 0..times {
                let a = rng.gen_range(0..l);
                let old = (chrom[[i, a]] as i64 - lo) as usize;
                let new = (old + 1 + rng.gen_range(0..range - 1)) % range;
                match pos[new] {
                    None => {
                        chrom[[i, a]] = (lo + new as i64) as f64;
                        pos[new] = Some(a);
                        pos[old] = None;
                    }
                    Some(b) => {
                        chrom.swap([i, a], [i, b]);
                        pos.swap(old, new);
                    }
                }
            }
        }
    }
    chrom_out(py, chrom, as_int)
}

/// Dispatcher kept for API compatibility: mutate(MUT_F, Encoding, OldChrom, ...) forwards the
/// remaining arguments to the named operator.
#[pyfunction]
#[pyo3(signature = (mut_f, *args, **kwargs))]
pub fn mutate<'py>(
    py: Python<'py>,
    mut_f: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<PyObject> {
    let name = mut_f.to_lowercase();
    let allowed = [
        "mutpolyn", "mutgau", "mutbga", "mutuni", "mutbin", "mutde", "mutinv", "mutmove",
        "mutswap", "mutpp",
    ];
    if !allowed.contains(&name.as_str()) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in mutate: unknown operator {}.",
            mut_f
        )));
    }
    let module = py
        .import("_geatpy_core")
        .or_else(|_| py.import("geatpy._geatpy_core"))?;
    Ok(module.getattr(name.as_str())?.call(args, kwargs)?.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_pick_matches_a_linear_scan() {
        for l in 0..70 {
            for lo in 1..4 {
                for max_len in lo..l + 3 {
                    let max_len = max_len.clamp(lo, l.max(lo));
                    let total = segment_total(l, max_len, lo);
                    let scan: usize = (lo..=max_len).map(|k| segment_weight(l, k)).sum();
                    assert_eq!(total, scan);
                    let mut k = lo;
                    let mut acc = segment_weight(l, k);
                    for r in 0..total {
                        while r >= acc {
                            k += 1;
                            acc += segment_weight(l, k);
                        }
                        assert_eq!(
                            segment_pick(l, max_len, lo, r),
                            k,
                            "l={l} lo={lo} max={max_len} r={r}"
                        );
                    }
                }
            }
        }
    }
}
