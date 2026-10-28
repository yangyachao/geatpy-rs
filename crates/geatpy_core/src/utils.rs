use numpy::ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::IntoPyDict;

/// Random generator for one operator call, seeded from numpy's global RandomState: like geatpy
/// 2.7.0, `np.random.seed` (and `ea.optimize(seed=...)`) then makes whole runs reproducible.
/// Every caller holds the GIL, so `with_gil` only re-enters it.
pub fn rng() -> rand::rngs::SmallRng {
    use rand::SeedableRng;
    static RANDINT: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    let seed = Python::with_gil(|py| -> PyResult<u64> {
        let randint = RANDINT.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
            Ok(py.import("numpy.random")?.getattr("randint")?.unbind())
        })?;
        let kwargs = [("dtype", "uint64")].into_py_dict(py)?;
        randint
            .bind(py)
            .call((0u64, u64::MAX), Some(&kwargs))?
            .extract()
    });
    rand::rngs::SmallRng::seed_from_u64(seed.unwrap_or_else(|_| rand::random()))
}

/// numpy.asarray(obj, dtype=float64) for lists, tuples, scalars and other array-likes.
fn as_float_array<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np = obj.py().import("numpy")?;
    np.call_method1(
        "ascontiguousarray",
        (np.call_method1("asarray", (obj, "float64"))?,),
    )
}

/// Row-major (C order) copy of `a`, or `a` itself when it already is: numpy arrays in Fortran order
/// (e.g. a transpose) keep that order when copied, and code reading rows as slices needs C order.
fn standard<T: Clone>(a: Array2<T>) -> Array2<T> {
    if a.is_standard_layout() {
        a
    } else {
        a.as_standard_layout().into_owned()
    }
}

/// Raise numpy's errors instead of aborting the interpreter when an array of `shape` with
/// `item` bytes per element cannot exist: ValueError when its size overflows (numpy's "array is
/// too big"), MemoryError when the allocator refuses it. Rust aborts the process on a failed
/// allocation, so every size that comes from user input is checked here first; the memory is
/// only reserved, not touched.
pub fn check_alloc(shape: &[usize], item: usize, dtype: &str) -> PyResult<()> {
    let bytes = shape
        .iter()
        .try_fold(item, |acc, &d| acc.checked_mul(d))
        .filter(|&b| b <= isize::MAX as usize);
    let Some(bytes) = bytes else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.",
        ));
    };
    if bytes < 1 << 26 {
        return Ok(());
    }
    let mut probe: Vec<u8> = Vec::new();
    probe.try_reserve_exact(bytes).map_err(|_| {
        let shape: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let shape = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            format!("({})", shape.join(", "))
        };
        pyo3::exceptions::PyMemoryError::new_err(format!(
            "Unable to allocate {} for an array with shape {} and data type {}",
            human_bytes(bytes),
            shape,
            dtype
        ))
    })?;
    Ok(())
}

/// Byte count in numpy's MemoryError style (e.g. "218. TiB", "1.50 GiB").
fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 7] = ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1023.995 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else if v >= 100.0 {
        format!("{:.0}. {}", v, UNITS[u])
    } else if v >= 10.0 {
        format!("{:.1} {}", v, UNITS[u])
    } else {
        format!("{:.2} {}", v, UNITS[u])
    }
}

/// A numeric matrix (1-D input becomes one column) as a row-major f64 array.
pub fn to_f64_array2<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Array2<f64>> {
    to_f64_matrix(obj).map(standard)
}

/// The ValueError geatpy 2.7.0's typed buffers raise for a numpy array with too many dimensions
/// (e.g. a 3-D ObjV), instead of the TypeError for input that is not numeric at all.
fn check_ndim(
    obj: &Bound<'_, PyAny>,
    expected: usize,
    accepted: impl Fn(&[usize]) -> bool,
) -> PyResult<()> {
    if let Ok(arr) = obj.downcast::<numpy::PyUntypedArray>() {
        use numpy::PyUntypedArrayMethods;
        if !accepted(arr.shape()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Buffer has wrong number of dimensions (expected {}, got {})",
                expected,
                arr.ndim()
            )));
        }
    }
    Ok(())
}

/// A 1-D array as one column. `mapv` keeps the memory order of the input, so a reversed view
/// (`x[::-1]`) gives a negative stride that `standard` turns into row-major order later.
fn column<T: Copy>(a: numpy::ndarray::ArrayView1<'_, T>, f: impl Fn(T) -> f64) -> Array2<f64> {
    a.mapv(f).insert_axis(numpy::ndarray::Axis(1))
}

fn to_f64_matrix<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Array2<f64>> {
    if obj.is_none() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a numeric array, got None",
        ));
    }
    check_ndim(obj, 2, |s| s.len() <= 2)?;
    if let Ok(arr) = obj.downcast::<numpy::PyUntypedArray>() {
        use numpy::PyUntypedArrayMethods;
        check_alloc(arr.shape(), 8, "float64")?;
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(ro.as_array().to_owned());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<i64>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<i32>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<f32>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(column(ro.as_array(), |x| x));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i64>>() {
        return Ok(column(ro.as_array(), |x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i32>>() {
        return Ok(column(ro.as_array(), |x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f32>>() {
        return Ok(column(ro.as_array(), |x| x as f64));
    }
    if let Ok(arr) = as_float_array(obj) {
        if let Ok(ro) = arr.extract::<PyReadonlyArray2<f64>>() {
            return Ok(ro.as_array().to_owned());
        }
        if let Ok(ro) = arr.extract::<PyReadonlyArray1<f64>>() {
            return Ok(column(ro.as_array(), |x| x));
        }
        if let Ok(v) = arr.extract::<f64>() {
            return Ok(Array2::from_elem((1, 1), v));
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected 1D or 2D numeric numpy array",
    ))
}

/// A numeric vector (an n x 1 or 1 x n matrix is read as one) as a contiguous f64 array.
pub fn to_f64_array1<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Array1<f64>> {
    to_f64_vector(obj).map(|a| {
        if a.as_slice().is_some() {
            a
        } else {
            a.iter().copied().collect()
        }
    })
}

fn to_f64_vector<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Array1<f64>> {
    if obj.is_none() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a numeric array, got None",
        ));
    }
    check_ndim(obj, 1, |s| {
        s.len() <= 1 || (s.len() == 2 && (s[0] == 1 || s[1] == 1))
    })?;
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(ro.as_array().to_owned());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i64>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i32>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f32>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<f64>>() {
        let arr = ro.as_array();
        if arr.shape()[1] == 1 {
            return Ok(arr.column(0).to_owned());
        } else if arr.shape()[0] == 1 {
            return Ok(arr.row(0).to_owned());
        }
    }
    if let Ok(arr) = as_float_array(obj) {
        if let Ok(ro) = arr.extract::<PyReadonlyArray1<f64>>() {
            return Ok(ro.as_array().to_owned());
        }
        if let Ok(ro) = arr.extract::<PyReadonlyArray2<f64>>() {
            let a = ro.as_array();
            if a.shape()[0] == 1 || a.shape()[1] == 1 {
                return Ok(a.iter().cloned().collect());
            }
        }
        if let Ok(v) = arr.extract::<f64>() {
            return Ok(Array1::from_elem(1, v));
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected 1D numeric numpy array",
    ))
}

/// An int32/int64 numpy matrix as a row-major i32 array (chromosome values are small), or None for
/// any other input.
pub fn to_i32_array2(obj: &Bound<'_, PyAny>) -> Option<Array2<i32>> {
    if let Ok(arr) = obj.downcast::<numpy::PyUntypedArray>() {
        use numpy::PyUntypedArrayMethods;
        // too big to copy: let the f64 conversion raise the error
        if check_alloc(arr.shape(), 8, "float64").is_err() {
            return None;
        }
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<i32>>() {
        return Some(standard(ro.as_array().to_owned()));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<i64>>() {
        return Some(standard(ro.as_array().mapv(|x| x as i32)));
    }
    None
}

/// Column-major copy of a matrix (its transpose, row-major), for loops that run down a column.
pub fn transpose(a: &Array2<f64>) -> Vec<f64> {
    a.t().iter().copied().collect()
}

/// numpy's sort order for floats (`np.sort`, `np.argsort`): NaN after every number, NaNs equal to
/// each other. Unlike `partial_cmp(..).unwrap_or(Equal)` this is a total order, so `sort_by` never
/// panics on NaN; for NaN-free input it agrees with `partial_cmp`.
pub fn cmp_nan_last(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

/// `out[k]` = the dot product of `x` with row k of `rows` (row-major, `x.len()` columns), each a
/// chain of fused multiply-adds starting from zero, `x[0] * row[0]` first: what numpy's float64 `@`
/// computes through OpenBLAS. x86_64 wheels target CPUs without FMA, where every `mul_add` is a
/// call into the software `fma` routine (correctly rounded, as IEEE 754 requires); on CPUs with FMA
/// the loop runs as hardware instructions instead. Both round once per step, so the result does
/// not depend on the CPU. The check runs once per call, so callers pass whole rows of products.
pub fn dot_fma_rows(x: &[f64], rows: &[f64], out: &mut [f64]) {
    assert_eq!(
        rows.len(),
        x.len() * out.len(),
        "dot_fma_rows: shape mismatch"
    );
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU supports FMA
        return unsafe { dot_fma_rows_hw(x, rows, out) };
    }
    dot_fma_rows_sw(x, rows, out)
}

#[inline(always)]
fn dot_fma_rows_sw(x: &[f64], rows: &[f64], out: &mut [f64]) {
    let d = x.len();
    for (k, o) in out.iter_mut().enumerate() {
        let row = &rows[k * d..(k + 1) * d];
        *o = x
            .iter()
            .zip(row)
            .fold(0.0, |s: f64, (p, q)| p.mul_add(*q, s));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn dot_fma_rows_hw(x: &[f64], rows: &[f64], out: &mut [f64]) {
    dot_fma_rows_sw(x, rows, out)
}

/// Calls `f(k)` for every k in 0..total selected by independent Bernoulli(p) trials. Draws the
/// geometric gaps between hits instead of one uniform per index: same distribution, far fewer
/// random numbers when p is small (e.g. the default Pm = 1/L).
pub fn for_each_hit<R: rand::Rng>(total: usize, p: f64, rng: &mut R, mut f: impl FnMut(usize)) {
    if total == 0 || p <= 0.0 {
        return;
    }
    if p >= 1.0 {
        (0..total).for_each(f);
        return;
    }
    let ln_q = (-p).ln_1p();
    let mut k = 0usize;
    while k < total {
        let u = 1.0 - rng.gen::<f64>(); // (0, 1]
        let gap = (u.ln() / ln_q).floor();
        if gap >= (total - k) as f64 {
            break;
        }
        k += gap as usize;
        f(k);
        k += 1;
    }
}

/// True when `obj` is a numpy array (or Python sequence) holding integers or booleans.
pub fn is_integer_like(obj: &Bound<'_, PyAny>) -> bool {
    obj.getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .map(|k| k == "i" || k == "u" || k == "b")
        .unwrap_or(false)
}

/// Return a chromosome matrix, as int32 when it holds integers: geatpy 2.7.0 returns int32 for every
/// integer chromosome (crossovers and mutations of integer input, 'P' and 'BG' operators, decoders).
pub fn chrom_out(py: Python<'_>, chrom: Array2<f64>, as_int: bool) -> PyResult<PyObject> {
    use numpy::IntoPyArray;
    if as_int {
        Ok(chrom
            .mapv(|x| x.round() as i32)
            .into_pyarray(py)
            .into_any()
            .unbind())
    } else {
        Ok(chrom.into_pyarray(py).into_any().unbind())
    }
}

/// Probability argument (Pm, XOVR, Pr, ...) that geatpy 2.7.0 requires to lie in [0, 1].
pub fn check_prob(p: f64, op: &str, msg: &str) -> PyResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: {}",
            op, msg
        )));
    }
    Ok(p)
}

/// Output of an 'RI' operator: like geatpy 2.7.0, an int32 matrix when every variable is an integer
/// (varTypes all 1), a float64 matrix otherwise.
pub fn ri_out(py: Python<'_>, chrom: Array2<f64>, discrete: &[bool]) -> PyResult<PyObject> {
    use numpy::IntoPyArray;
    if !discrete.is_empty() && discrete.iter().all(|&d| d) {
        Ok(chrom
            .mapv(|x| x.round() as i32)
            .into_pyarray(py)
            .into_any()
            .unbind())
    } else {
        Ok(chrom.into_pyarray(py).into_any().unbind())
    }
}

/// None / missing -> None; Python bool -> Some(Bool); number -> Scalar; 1-D array -> Array.
pub enum Knob {
    Bool(bool),
    Scalar(f64),
    Array(Vec<f64>),
}

pub fn knob(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Knob>> {
    let Some(o) = obj else { return Ok(None) };
    if o.is_none() {
        return Ok(None);
    }
    let type_name = o.get_type().name()?.to_string();
    if o.is_instance_of::<pyo3::types::PyBool>() || type_name == "bool_" || type_name == "bool" {
        return Ok(Some(Knob::Bool(o.is_truthy()?)));
    }
    if let Ok(v) = o.extract::<f64>() {
        return Ok(Some(Knob::Scalar(v)));
    }
    let a = to_f64_array2(o)?;
    Ok(Some(Knob::Array(a.iter().cloned().collect())))
}

/// Per-gene parameter: scalar broadcast, 1-D array of length `d`, or `default` when absent.
pub fn per_gene(
    obj: Option<&Bound<'_, PyAny>>,
    d: usize,
    default: f64,
    name: &str,
) -> PyResult<Vec<f64>> {
    match knob(obj)? {
        None => Ok(vec![default; d]),
        Some(Knob::Bool(b)) => Ok(vec![if b { 1.0 } else { 0.0 }; d]),
        Some(Knob::Scalar(v)) => Ok(vec![v; d]),
        Some(Knob::Array(v)) => {
            if v.len() != d {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "error: the length of {} must equal the chromosome length ({}).",
                    name, d
                )));
            }
            Ok(v)
        }
    }
}

/// Optional integer argument that may arrive as None.
pub fn opt_int(obj: Option<&Bound<'_, PyAny>>, default: i64) -> PyResult<i64> {
    match obj {
        Some(o) if !o.is_none() => Ok(o.extract::<f64>()? as i64),
        _ => Ok(default),
    }
}

/// A Python object converted like Cython's C `long` arguments in geatpy 2.7.0: ints (numpy ints and
/// bools too) as they are, floats truncated toward zero through `__int__` (NaN raises ValueError,
/// inf OverflowError), strings and other objects TypeError.
pub fn c_long(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    use pyo3::exceptions::{PyOverflowError, PyTypeError};
    use pyo3::types::{PyBytes, PyString};
    let int = if obj.is_instance_of::<pyo3::types::PyInt>() {
        obj.clone()
    } else if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || !obj.hasattr("__int__")?
    {
        return Err(PyTypeError::new_err("an integer is required"));
    } else {
        obj.call_method0("__int__")?
    };
    int.extract::<i64>()
        .map_err(|_| PyOverflowError::new_err("Python int too large to convert to C long"))
}

/// Like `c_long` for a C `int` argument.
pub fn c_int(obj: &Bound<'_, PyAny>) -> PyResult<i32> {
    c_long(obj).and_then(|v| {
        i32::try_from(v).map_err(|_| {
            pyo3::exceptions::PyOverflowError::new_err("value too large to convert to int")
        })
    })
}

/// FixType argument: 1 (default), 2, 3 or 4; anything else is rejected like in geatpy 2.7.0.
pub fn fix_type(obj: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<i64> {
    let t = opt_int(obj, 1)?;
    if !(1..=4).contains(&t) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: FixType must be an integer that is 1, 2, 3 or 4. (传入参数FixType必须是1,2,3或4。)",
            name
        )));
    }
    Ok(t)
}

/// Lower/upper bounds and spans as used by the real-valued operators: integer variables are
/// widened by 0.499999 on both sides so that rounding maps back uniformly onto the integers.
pub struct Bounds {
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    pub span: Vec<f64>,
    pub discrete: Vec<bool>,
}

pub fn bounds(field: &Array2<f64>, d: usize, widen_discrete: bool) -> PyResult<Bounds> {
    if field.shape()[0] < 2 || field.shape()[1] != d {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error: FieldDR must have 3 rows and as many columns as the chromosome length.",
        ));
    }
    let mut b = Bounds {
        lb: vec![0.0; d],
        ub: vec![0.0; d],
        span: vec![0.0; d],
        discrete: vec![false; d],
    };
    for j in 0..d {
        let discrete = field.shape()[0] > 2 && field[[2, j]] == 1.0;
        let (mut l, mut u) = (field[[0, j]], field[[1, j]]);
        if discrete && widen_discrete {
            l -= 0.499999;
            u += 0.499999;
        }
        b.lb[j] = l;
        b.ub[j] = u;
        b.span[j] = (u - l).abs();
        b.discrete[j] = discrete;
    }
    Ok(b)
}

/// Out-of-range repair. 1: truncate, 2: wrap around, 3: reflect, 4: uniform random, other: none.
/// A degenerate range (span <= 1e-15) collapses to lb, as in the reference implementation.
pub fn fix_value<R: rand::Rng>(
    x: f64,
    lb: f64,
    ub: f64,
    span: f64,
    fix_type: i64,
    rng: &mut R,
) -> f64 {
    if (1..=4).contains(&fix_type) && span <= 1e-15 {
        return lb;
    }
    match fix_type {
        // lower bound first, as geatpy 2.7.0; equals clamp when lb <= ub, and unlike clamp neither
        // panics nor returns NaN for reversed or NaN bounds (a NaN bound is never crossed)
        1 => {
            if x < lb {
                lb
            } else if x > ub {
                ub
            } else {
                x
            }
        }
        2 => {
            if x > ub {
                lb + (x - ub) % span
            } else if x < lb {
                ub - (lb - x) % span
            } else {
                x
            }
        }
        3 => {
            if x > ub {
                ub - (x - ub) % span
            } else if x < lb {
                lb + (lb - x) % span
            } else {
                x
            }
        }
        4 => {
            if x > ub || x < lb {
                lb + rng.gen::<f64>() * span
            } else {
                x
            }
        }
        _ => x,
    }
}

/// Multiply each objective by its maxormins flag (1: minimise, -1: maximise).
pub fn unify_objectives(
    obj: &mut Array2<f64>,
    maxormins: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if let Some(mom) = maxormins {
        if !mom.is_none() {
            let v = to_f64_array1(mom)?;
            if v.len() != obj.shape()[1] {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error: The length of maxormins must equal the number of objectives.",
                ));
            }
            for mut row in obj.rows_mut() {
                for (x, m) in row.iter_mut().zip(v.iter()) {
                    *x *= m;
                }
            }
        }
    }
    Ok(())
}

/// geatpy 2.7.0's error for a CV whose row count differs from ObjV's.
pub fn cv_rows_error(name: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "error in {name}: CV and ObjV disagree. (违反约束程度矩阵的行数必须和目标函数值矩阵的行数相同。)"
    ))
}

/// Total positive constraint violation of each row, if CV is given.
pub fn violation(
    cv: Option<&Bound<'_, PyAny>>,
    n: usize,
    name: &str,
) -> PyResult<Option<Vec<f64>>> {
    match cv {
        Some(c) if !c.is_none() => {
            let a = to_f64_array2(c)?;
            if a.shape()[0] != n {
                return Err(cv_rows_error(name));
            }
            Ok(Some(
                a.rows()
                    .into_iter()
                    .map(|r| r.iter().filter(|&&x| x > 0.0).fold(0.0, |s, x| s + x))
                    .collect(),
            ))
        }
        _ => Ok(None),
    }
}

/// Feasibility rule used across the core: an infeasible row gets, on every objective,
/// the column maximum of the population plus its total violation.
pub fn penalise_infeasible(obj: &mut Array2<f64>, viol: &[f64]) {
    let m = obj.shape()[1];
    let col_max: Vec<f64> = (0..m)
        .map(|j| {
            obj.column(j)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    for (i, &v) in viol.iter().enumerate() {
        if v > 0.0 {
            for j in 0..m {
                obj[[i, j]] = col_max[j] + v;
            }
        }
    }
}

/// ObjV converted to "minimise everything" with infeasible rows penalised.
pub fn prepared_objectives(
    obj_v: &Bound<'_, PyAny>,
    cv: Option<&Bound<'_, PyAny>>,
    maxormins: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Array2<f64>> {
    let mut obj = to_f64_array2(obj_v)?;
    unify_objectives(&mut obj, maxormins)?;
    if let Some(v) = violation(cv, obj.shape()[0], name)? {
        penalise_infeasible(&mut obj, &v);
    }
    Ok(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_fma_rows_fuses_every_step_after_the_first() {
        // a*a rounds away 2^-60, which the fused a*(-a) + round(a*a) recovers
        let a = 1.0 + 2f64.powi(-30);
        let mut out = [1.0];
        dot_fma_rows(&[a, a], &[a, -a], &mut out);
        assert_eq!(out[0], -(2f64.powi(-60)));
        let mut sw = [1.0];
        dot_fma_rows_sw(&[a, a], &[a, -a], &mut sw);
        assert_eq!(sw[0], -(2f64.powi(-60)));
    }

    /// Dot products whose every step was rounded once from the exact rational value (generated
    /// with Python's `fractions`, independently of any FMA instruction), with subnormals and
    /// cancellation; 40 of them come out differently without fusing. Whichever path this CPU takes (hardware FMA, or the software `fma` of CPUs
    /// without it) must reproduce them bit for bit.
    #[test]
    fn dot_fma_rows_matches_exactly_rounded_references() {
        #[rustfmt::skip]
        const CASES: &[(&[u64], &[u64], u64)] = &[
            (&[0xc0b060b907585c68, 0xbff0000003f5ff91], &[0xbcab24daf641b434, 0x3d6bc8f1f7c026e3], 0x39d9ba0ca5db8da0),
            (&[0xbff0000001c84792, 0x424cfc8659f7e668], &[0x42969178618fa8b8, 0x4038ea200f0bd206], 0xbf313bcb156e9320),
            (&[0x419212824fd3dca0, 0x3ff00000023b8505, 0xc34444176498f362, 0x0000000000000703, 0xc2faa3f6632273e2], &[0x3ff00000021d987c, 0xc19212824fb20fe8, 0xbd1db6c5960d80b0, 0x0000000000001ad2, 0xbd8e7a370f31a50c], 0x409e13d3555b8d5c),
            (&[0x3e79b3443deb54ca, 0xbff00000008512d9], &[0xbff00000009c77f8, 0xbe79b3443e10e8ea], 0x3b1ffffb391f8968),
            (&[0x3ff0000001f9f9c0, 0x3e8a32175a64f760, 0x8013939dbc0375ec, 0x80216b0fa2a941c2, 0xbc617c56440e73aa], &[0x3efbf24f96ba99d0, 0xc06111c468e063d7, 0x3ff00000008a1a08, 0xbff00000019a728e, 0xbfd39698ee274db6], 0x3c45673d4651f3a3),
            (&[0x3ff0000000fd5ac5, 0xbff000000278374b, 0x0000000000d22ec2], &[0x3ff0000003921ea4, 0x3ff0000003cbd8aa, 0xc120208de1816226], 0xbe3b4968c96010cc),
            (&[0x3ff00000008492c0, 0x3ff00000029b2733], &[0x3cb88ad928d77e52, 0xbcb88ad925a3808e], 0x390880d2eee0db00),
            (&[0xbff00000004cb4f4, 0x3ff00000005a73fc, 0xc1fd8d8a9b346186, 0xbc32facf497b67a6, 0xbef74c1c4a4b3468], &[0x3ff00000027f607b, 0x3ff000000271a173, 0x8003305b9c344394, 0xc0661bd13ae2f494, 0x80000002b0c3a79a], 0x3cabf3f8e082f342),
            (&[0xbff0000001fe7b6d, 0xbff0000002f089a2, 0xbc248f21c7d020a0, 0xbdb5802f88e3dc40], &[0x3ff0000003aa9372, 0xbff0000002b8853d, 0x800000000000011c, 0x4101c8fb0533da80], 0xbec7e6462584ab0e),
            (&[0x41167cf7524d1aca, 0x3c527fe4234da97e, 0xbff0000000c95142, 0x3ff0000000628b38, 0x40b4fb29d0ba244a], &[0xbc2518a32482f44c, 0xc21b17d983ce1b48, 0xc00fb889d537b2b6, 0x000000000031f926, 0x42a7b1ac8b1c2520], 0x436f1208dd0d06ab),
            (&[0x3eca575550b5b038, 0x3ff00000029fe5d3], &[0x3ff0000000d07ff7, 0xbeca57554dbac952], 0x3b5fcc99583dbb50),
            (&[0x43049f07bee216a4, 0x3ff00000029fda8e, 0x3ff0000003eb3152, 0x0000000000016334], &[0x3dbce287878652da, 0xc0d29d2140ea79f6, 0xbff0000001a3cd90, 0x42b1d3505cae5a84], 0xbff00000058ee964),
            (&[0x3ff0000003a7c038, 0xc0343c93dcdc2ca0, 0xc1082aafba2aca3a], &[0xbff000000073f44d, 0x3ff0000003328d9f, 0x800535f1c0ad3012], 0xc0353c93e129340f),
            (&[0x3fe6e38742f96780, 0x000000038c730d99, 0xbff00000026a20c0], &[0x3ff00000036d8de2, 0xc3719bd76d8e52b4, 0x3ff000000314b4eb], 0xbfd238f18639c2f2),
            (&[0x3f9ef6f86d6db010, 0xbd188363b38f3ab8], &[0x3ff0000001df9789, 0x427436003ebbc15d], 0x3c39c241552bcca0),
            (&[0x3ff00000009a6965, 0x3c416c94ba5826f8, 0x800003a20ec25142], &[0xbff9fc4d3bea3d24, 0x43a7dc952cdcc95c, 0x3ff00000035321aa], 0xbc71b71b49946e00),
            (&[0x3ff000000264cd36, 0x425141cf9166b650], &[0xc069e31be8dfd578, 0x3e08005d3f2872f1], 0x3d0274872d850540),
            (&[0xc2012b0d139d99fc, 0x41f6c06e79288caa], &[0xc08a7a47b5f128dc, 0xc3642ee515263c70], 0xc56cb339154f1990),
            (&[0xbd652ed053223974, 0xbff0000000dfb817, 0x3ff0000003dc0791, 0x3ff0000000746f4f, 0x3d157543da4a349a], &[0xbff00000019cebe8, 0x3ff0000002179943, 0xbe79c17f5ce0fb84, 0x3ff00000014c2c35, 0x3d4ea046da592354], 0xbe7af82aa1f68eb9),
            (&[0xbe17667487d5a1a2, 0xbd91aeea730c2044, 0xbff00000027ecc32, 0x3c4d2485483d9734, 0x000000000065df0a], &[0x3ff0000003b6263b, 0xc0752c39be5cd333, 0x80000002917775f4, 0x806a3f5ed404a480, 0xbd18f51ac7061144], 0xbabecc18163619d0),
            (&[0xbff0000001e8886e, 0xbe119905bae60180, 0xbc922abed5a2ceaa], &[0xbca5da0b239749ee, 0x3e83de256c17a267, 0x802cb36cc54d2b20], 0xb931e2ff0fecd400),
            (&[0x3c5d2db1f015e5d4, 0x00909f8ffcab2530, 0xbff00000027fd555, 0x4151ae14dd121da0], &[0xbff0000003b6c7e5, 0x00468e96559e2220, 0x404696ce2117a138, 0xbff000000346b1f4], 0xc151ae202c17f218),
            (&[0x3ff000000086090a, 0x3ff0000002cf5ae1, 0xbff0000003bc71c3], &[0x3ff00000037c0f81, 0xbff000000132bdaa, 0xbd20865a825a603e], 0x3d207f9ea70cb86f),
            (&[0x409265c5b905b914, 0x3ff0000001ebe5cf, 0xbff0000001bb57db], &[0xbff0000001aa1ad0, 0x3d93c85468ad1400, 0x4396704fb21859a0], 0xc396704fb4861a87),
            (&[0xbda12e0b356e8412, 0xc09906c71f74920e], &[0x423388132400bfa6, 0xbf3ad0c6b0cac9a4], 0xbc780e11ce65c420),
            (&[0xbff000000278a94b, 0xbfa5acf09d677bea, 0xbff0000002e75fe3], &[0x3ff000000093157f, 0xc0379effaae65d49, 0x013bcb49554ebef4], 0x3c956e486584af74),
            (&[0xc3073dd06d9b0262, 0x3ff000000039050f, 0x3ff000000394299c, 0x01143e078be78140], &[0xc0356852f47adec0, 0xc34f18abda22681f, 0x3ff0000001d1659d, 0xc21304fdb39d767c], 0x3fea84097efc4b2f),
            (&[0xc3104e98ad3251de, 0xbc7812d8554c5e58, 0x4325fc01752bd3bc, 0xbed2fa85bf85ab20, 0x3ee3ca3951e0e682], &[0x3ff0000001a494e5, 0x3f45f04e574001f8, 0x3ff00000026b5609, 0xc1e6cbbc95968cb0, 0x3ff0000002b946de], 0x431ba96a421f7e62),
            (&[0x3e65bd82a5b84418, 0x3eff3294d3be33e6, 0x41d85b1b46da3158, 0xc00baa08394a92da, 0xbff000000166e093], &[0x4207ccddc17a02f0, 0x3ff000000224f7f2, 0xbf520723af9e526c, 0x3ff000000242d656, 0x3ff00000007453c5], 0xc13b6f5be6eeaa6a),
            (&[0xbff00000001ebbfa, 0xbeb26ae23d6d7854], &[0xc00e443eb2cefc28, 0x414a4b326c55db09], 0xbcaff414cab45bd0),
            (&[0x3ff00000016e8db2, 0xc206883812ef3426, 0x405f93b000e61ea2, 0x3ff0000003f85a95, 0x3ff0000001b899f1], &[0xbd7d0f39b4be201c, 0x800000000000000f, 0xbff00000006202bf, 0x3e4cec623ae6a000, 0x0058151d90a99b2c], 0xc05f93b0019916e7),
            (&[0x3e4720b8c2bbafa0, 0xbff0000000b4397d, 0x3ff000000391d20a], &[0x3ff0000003d3a3ea, 0xc196685b87f3e078, 0x80000000016fc798], 0x4196685b88f04701),
            (&[0xbf22e61cdc2e969c, 0xc117b6b34899f982, 0x3ff0000002fa95c3, 0xbca5d490423cb538, 0x80006f94259edf70], &[0xc2ebdae041cbb200, 0x40f633081940a849, 0xbfa32dc9c427ff50, 0xbfc05a6b32b932f0, 0xbff00000008550aa], 0xbfa32ddedd40a8a6),
            (&[0x4174c765d8137a72, 0xbff00000019a19f4], &[0x3e7b3ff4857050c4, 0x4001b1c5441a8acb], 0x3cace748d81f8e10),
            (&[0xc2fc3893b2f3f470, 0xbff000000170ea86, 0xbd3808ca9f9bb942, 0xc0eb6587e17c5250], &[0x3ff00000001d102e, 0xc2fc3893b09c8498, 0xc255d3adeed573a8, 0xbce2c206a3dec1ca], 0x3f8580e73fd5898a),
            (&[0x3f7cdb1168670956, 0xbff00000020f8b43], &[0xbc961727010dfa86, 0xbd67767997aa7b88], 0x3d6776785bf84c81),
            (&[0x4287fb675ca80d08, 0xbff00000013b5e7a, 0xc2c67a4fa7386628], &[0xbff0000001d8b671, 0xbd4610b0e9eb9fbc, 0xc071b46382c2e912], 0x4348dde0b7cfa48e),
            (&[0x3ff0000003f8ef6f, 0xbff00000023d7008, 0x3ca4d8c7f0665280, 0xbff0000003c37e46], &[0x4350f700aec36500, 0x4350f700b099a2f9, 0xbc298ead9842db9a, 0xbff000000137bc66], 0x3fc65aafb9f8dc62),
            (&[0x3ff0000003d92bfa, 0x3ff0000002e9938e, 0x00000000000000aa], &[0xc1e70c8079e090a0, 0x41e70c807b39b6f1, 0x3ff0000000498540], 0xbe757820f4b91a90),
            (&[0x401369fca7b9dafa, 0x3d2f6e6ce7f089fc, 0x3ff00000026c79b7], &[0x428546f81e4ed6c8, 0xbf3575279c84767a, 0x4312341b57803c5a], 0x431267bdcafc5286),
            (&[0x3f23bd04d2bcc158, 0x8131d9cb9fa09944, 0xbd3aee15394b34dc, 0x4072276649df1540, 0xc0c46efa9f2cf122], &[0xbce3f9ebcd055470, 0xfad616ae95323871, 0xbffd5f5947453b00, 0xbff0000002835f98, 0xbe57f837a8dbd5b0], 0xc072276557d4d4dd),
            (&[0xbff00000009ef77d, 0xbff0000002cd2a54, 0xc2a66c413b996350, 0x8000000000000076], &[0x0149ca7d94f426dc, 0x3e649a6a6b90ac1a, 0x3ee8c544a5596698, 0x8000000000027396], 0xc1a15b6b783166c3),
            (&[0x3ff00000024164e6, 0x0111b5e817a1d9b0, 0x3ff00000010101b2, 0xbff000000323978e, 0xbff0000000d40b42], &[0xbcac0421b679a1b6, 0xbcde6627b59cd320, 0x3ff00000006bacc3, 0x8000000000000b29, 0x3c9d1de2a0d0e738], 0x3ff00000016cae74),
            (&[0xbff0000002049fb7, 0xbcca7809f046b4c0, 0x3fdf8efeb7d294dc, 0x3ff00000027eac6e, 0x00000000ffaa852c], &[0x3ff00000014aa054, 0xc31357eca9aa2e41, 0x423a815d5037620c, 0x430e9fc3ad24b1b4, 0x00e61bd03a16c390], 0x430ea02c40f8d9d9),
            (&[0x3ff00000003b59f2, 0xbd6c6d19ded35fe0, 0x3fbd378b02c1165a, 0x3edc8f74ffcf00fc], &[0xbdd1539c7eb61da8, 0xc0538145080401ff, 0xbff000000003e882, 0xc133f794f334a8c4], 0xc0220c94cd8c7e2a),
            (&[0x422211401db1233a], &[0xbff00000016d9ccc], 0xc22211401f4dfdcf),
            (&[0x000391bf22f8be88, 0x3ff0000000adea25, 0x3ff0000001042cb2], &[0x3ff0000003b90b75, 0x3fc88af3ff5eaa90, 0xc07325d8bd8161fc], 0xc07322c76038b14d),
            (&[0x0000000000001e23], &[0x3f9bc082411d2f7c], 0x00000000000000d1),
            (&[0xbe227e4847c6ab3a, 0xbff0000001ec9feb], &[0x3ff00000035a2341, 0x005eb5c405a9196c], 0xbe227e484ba687dd),
            (&[0x3ff00000010bcd9c, 0xbfaecd8a19047e80, 0x423bca3ca0203700, 0xbfe4f8457ede1656, 0xbc984c67ab8c5682], &[0x3ff0000003dc2227, 0xbdac5d04de03156a, 0xbff0000000c82ead, 0x4360a3ba4726efcc, 0x80b6524ad63315f8], 0xc355cee1d8d2545e),
            (&[0xc0386621df7a5ea0, 0x3dbfc069e20c6f80], &[0x43a819284702000a, 0x434b0ee58479e25e], 0xc3f25fc7e7e0d711),
            (&[0x804837588cdc7012, 0xbff0000001ecd404, 0x0000000000002b63, 0xc1d61d8be87be4fc], &[0xbe1120e7edb65ba8, 0x3dedec40f3bdec48, 0xc339f9d944d05eb4, 0x3ff0000001ca299c], 0xc1d61d8beaf52c44),
            (&[0x3d8fd4f761373d20, 0xbff00000016c56c7, 0xbff00000002532b3], &[0x80ced1faa222a6c2, 0xbfe3dff1dbe2d8d0, 0x3ff00000039391dd], 0xbfd8401c53943b8e),
            (&[0x43078106356a5080, 0x3ff00000027656f6, 0xc2b422413cd26f24, 0x3d2d4ac40a2cf628], &[0x3ff0000002c7c970, 0xc12f1671a76f28e2, 0xbd5f48b3291bafa4, 0x00005498b8f5d75a], 0x4307810639039475),
            (&[0x3c68a9eb44b05e18, 0xbff000000142a0a2, 0xbff00000021b1f2f], &[0x408e3f7487e1192c, 0x3d075034e66daae7, 0x3ff000000176b2a1], 0xbff000000391d1d0),
            (&[0x3ff00000019b9b6f], &[0xbe3faee3f745ef8c], 0xbe3faee3fa74ffd6),
        ];
        let floats = |v: &[u64]| v.iter().map(|&b| f64::from_bits(b)).collect::<Vec<f64>>();
        for (x, w, want) in CASES {
            let (x, w) = (floats(x), floats(w));
            let mut out = [0.0];
            dot_fma_rows(&x, &w, &mut out);
            assert_eq!(out[0].to_bits(), *want, "{x:?} . {w:?}");
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("fma") {
                let mut hw = [0.0];
                // SAFETY: the CPU supports FMA
                unsafe { dot_fma_rows_hw(&x, &w, &mut hw) };
                assert_eq!(hw[0].to_bits(), *want, "{x:?} . {w:?}");
            }
        }
    }
}
