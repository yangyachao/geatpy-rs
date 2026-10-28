use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::seq::SliceRandom;
use rand::Rng;

/// needNum / needLevel as geatpy 2.7.0 converts them (C ints): None keeps the default (sort every
/// level), floats are truncated, negative values act as 0, NaN raises ValueError and strings
/// TypeError. Values beyond the C int range (inf included) sort every level instead of raising
/// the original's OverflowError.
fn ndsort_count(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<usize>> {
    let Some(o) = obj.filter(|o| !o.is_none()) else {
        return Ok(None);
    };
    match crate::utils::c_long(o) {
        Ok(v) => Ok(usize::try_from(v.clamp(0, i32::MAX as i64)).ok()),
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyOverflowError>(o.py()) => Ok(None),
        Err(e) => Err(e),
    }
}

#[allow(clippy::too_many_arguments)]
fn ndsort_entry<'py>(
    py: Python<'py>,
    name: &str,
    obj_v: &Bound<'py, PyAny>,
    need_num: Option<&Bound<'py, PyAny>>,
    need_level: Option<&Bound<'py, PyAny>>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    tree: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    let need_num = ndsort_count(need_num)?;
    let need_level = ndsort_count(need_level)?;
    let obj = crate::utils::to_f64_array2(obj_v)?;
    let cv = match cv.filter(|c| !c.is_none()) {
        Some(c) => Some(crate::utils::to_f64_array2(c)?),
        None => None,
    };
    ndsort_core(
        py,
        name,
        obj,
        need_level,
        need_num,
        cv,
        maxormins.filter(|m| !m.is_none()),
        tree,
    )
}

/// Efficient Non-dominated Sort (ENS-SS)
#[pyfunction]
#[pyo3(signature = (ObjV, needNum=None, needLevel=None, CV=None, maxormins=None))]
#[allow(non_snake_case)]
pub fn ndsortESS<'py>(
    py: Python<'py>,
    ObjV: &Bound<'py, PyAny>,
    needNum: Option<&Bound<'py, PyAny>>,
    needLevel: Option<&Bound<'py, PyAny>>,
    CV: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    ndsort_entry(
        py,
        "ndsortESS",
        ObjV,
        needNum,
        needLevel,
        CV,
        maxormins,
        false,
    )
}

/// Tree-based Non-dominated Sort (T-ENS)
#[pyfunction]
#[pyo3(signature = (ObjV, needNum=None, needLevel=None, CV=None, maxormins=None))]
#[allow(non_snake_case)]
pub fn ndsortTNS<'py>(
    py: Python<'py>,
    ObjV: &Bound<'py, PyAny>,
    needNum: Option<&Bound<'py, PyAny>>,
    needLevel: Option<&Bound<'py, PyAny>>,
    CV: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    ndsort_entry(
        py,
        "ndsortTNS",
        ObjV,
        needNum,
        needLevel,
        CV,
        maxormins,
        true,
    )
}

/// Non-dominated Sort with Deduplication
#[pyfunction]
#[pyo3(signature = (ObjV, needNum=None, needLevel=None, CV=None, maxormins=None))]
#[allow(non_snake_case)]
pub fn ndsortDED<'py>(
    py: Python<'py>,
    ObjV: &Bound<'py, PyAny>,
    needNum: Option<&Bound<'py, PyAny>>,
    needLevel: Option<&Bound<'py, PyAny>>,
    CV: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    ndsort_entry(
        py,
        "ndsortDED",
        ObjV,
        needNum,
        needLevel,
        CV,
        maxormins,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn ndsort_core<'py>(
    py: Python<'py>,
    name: &str,
    raw_obj: Array2<f64>,
    need_level: Option<usize>,
    need_num: Option<usize>,
    cv: Option<Array2<f64>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    tree: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    let n = raw_obj.shape()[0];
    let m = raw_obj.shape()[1];
    if n == 0 {
        return Ok((Array1::<f64>::zeros(0).into_pyarray(py), 0));
    }
    if let Some(ref c) = cv {
        if c.shape()[0] != n {
            return Err(crate::utils::cv_rows_error(name));
        }
    }

    // Unify to minimisation.
    let mut mult = vec![1.0; m];
    if let Some(mom) = maxormins {
        if !mom.is_none() {
            let v = crate::utils::to_f64_array1(mom)?;
            if v.len() != m {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in ndsort: maxormins and ObjV disagree in number of objectives.",
                ));
            }
            for j in 0..m {
                mult[j] = v[j];
            }
        }
    }
    let mut obj: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| raw_obj[[i, j]] * mult[j]).collect())
        .collect();

    // Constraint domination: an infeasible individual gets (column maximum + total violation)
    // on every objective, so feasible ones dominate it and infeasible ones compare by violation.
    // The column maximum is taken like geatpy 2.7.0: start from the first row and keep any larger
    // value, so a NaN in the first row stays the maximum and later NaNs are skipped.
    if let Some(c) = cv {
        let col_max: Vec<f64> = (0..m)
            .map(|j| {
                obj[1..]
                    .iter()
                    .fold(obj[0][j], |mx, r| if r[j] > mx { r[j] } else { mx })
            })
            .collect();
        for (row, viol) in obj.iter_mut().zip(c.rows()) {
            let s: f64 = viol.iter().filter(|&&x| x > 0.0).fold(0.0, |s, x| s + x);
            if s > 0.0 {
                for (x, mx) in row.iter_mut().zip(&col_max) {
                    *x = mx + s;
                }
            }
        }
    }

    // Unique rows in lexicographic order, with multiplicities and the row -> unique map.
    let has_nan = obj.iter().flatten().any(|x| x.is_nan());
    let (uobj, rep, inverse) = if has_nan {
        numpy_unique_rows(py, &obj)?
    } else {
        unique_rows(&obj)
    };

    let nums = need_num.unwrap_or(n);
    let max_level = need_level.unwrap_or(n);
    let (lv, cri_level) = if !has_nan {
        // objectives 2..M of the unique rows, contiguous (the first one only orders the rows)
        let w = m.saturating_sub(1);
        let pts: Vec<f64> = uobj
            .chunks(m.max(1))
            .flat_map(|r| r.iter().skip(1).copied())
            .collect();
        front_levels(&pts, w, &rep, n, nums, max_level)
    } else if tree {
        tens_levels(
            &uobj,
            m,
            &objective_ranks(py, &uobj, m)?,
            &rep,
            n,
            nums,
            max_level,
        )
    } else {
        ens_levels(&uobj, m, &rep, n, nums, max_level)
    };

    let levels = Array1::from_iter((0..n).map(|i| {
        let l = lv[inverse[i]];
        if l == 0 {
            f64::INFINITY
        } else {
            l as f64
        }
    }));
    Ok((levels.into_pyarray(py), cri_level))
}

/// Unique rows (row-major, M values each), the multiplicity of each and the unique row of every
/// input row.
type UniqueRows = (Vec<f64>, Vec<usize>, Vec<usize>);

/// Unique rows of NaN-free objectives (sorted lexicographically), their multiplicities and the
/// index of every row's unique vector.
fn unique_rows(obj: &[Vec<f64>]) -> UniqueRows {
    let n = obj.len();
    let lex = |a: &[f64], b: &[f64]| {
        for (&x, &y) in a.iter().zip(b) {
            let o = crate::utils::cmp_nan_last(x, y);
            if o.is_ne() {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| lex(&obj[a], &obj[b]));
    let mut uobj: Vec<f64> = Vec::with_capacity(n * obj.first().map_or(0, |r| r.len()));
    let mut last: Option<usize> = None; // representative row of the current unique vector
    let mut rep: Vec<usize> = Vec::new();
    let mut inverse = vec![0usize; n];
    for &i in &order {
        match last {
            Some(u) if lex(&obj[u], &obj[i]).is_eq() => {
                *rep.last_mut().unwrap() += 1;
            }
            _ => {
                last = Some(i);
                uobj.extend_from_slice(&obj[i]);
                rep.push(1);
            }
        }
        inverse[i] = rep.len() - 1;
    }
    (uobj, rep, inverse)
}

/// `np.unique(ObjV, axis=0, return_inverse=True, return_counts=True)`, which geatpy 2.7.0 runs
/// before sorting; used when the objectives contain NaN. numpy sorts NaN after every number, never
/// merges two rows holding NaN, and orders such rows that compare equal by its unstable quicksort,
/// which decides which of them lands on which front.
fn numpy_unique_rows(py: Python<'_>, obj: &[Vec<f64>]) -> PyResult<UniqueRows> {
    let (n, m) = (obj.len(), obj[0].len());
    let arr = Array2::from_shape_fn((n, m), |(i, j)| obj[i][j]).into_pyarray(py);
    let kwargs = [("return_inverse", true), ("return_counts", true)].into_py_dict(py)?;
    kwargs.set_item("axis", 0)?;
    let res = py
        .import("numpy")?
        .call_method("unique", (arr,), Some(&kwargs))?;
    let uni = crate::utils::to_f64_array2(&res.get_item(0)?)?;
    let inverse: Vec<usize> = res
        .get_item(1)?
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract()?;
    let counts: Vec<usize> = res.get_item(2)?.call_method0("tolist")?.extract()?;
    Ok((uni.iter().cloned().collect(), counts, inverse))
}

/// Front members `front_dominates` compares with a candidate at once.
const FRONT_BLOCK: usize = 64;

/// Whether a member of a front weakly dominates `x`; `cols` holds the members' objectives by column,
/// oldest first. The latest member is tested alone first (with many fronts it usually settles it),
/// then blocks of the latest members, every member of a block on every objective: no early exit,
/// which the branch predictor cannot follow, so the loops vectorise.
fn front_dominates(cols: &[Vec<f64>], x: &[f64]) -> bool {
    let len = cols.first().map_or(0, Vec::len);
    if len == 0 {
        return false;
    }
    if cols.iter().zip(x).all(|(c, &b)| c[len - 1] <= b) {
        return true;
    }
    let mut hit = [false; FRONT_BLOCK];
    let mut end = len - 1;
    while end > 0 {
        let start = end.saturating_sub(FRONT_BLOCK);
        let hit = &mut hit[..end - start];
        for (h, &a) in hit.iter_mut().zip(&cols[0][start..end]) {
            *h = a <= x[0];
        }
        for (col, &b) in cols.iter().zip(x).skip(1) {
            for (h, &a) in hit.iter_mut().zip(&col[start..end]) {
                *h &= a <= b;
            }
        }
        if hit.iter().any(|&h| h) {
            return true;
        }
        end = start;
    }
    false
}

/// Efficient non-dominated sort, one complete front per pass. `pts` holds objectives 2..M (`w` of
/// them) of the unique rows sorted lexicographically, `rep` their multiplicities. Returns the level
/// of every unique row (0 = unranked) and the last level visited. NaN-free input only.
///
/// Only earlier rows can dominate a row (the first objective is sorted), and uniqueness turns
/// weak dominance into dominance. A row joins the front iff no earlier front member dominates it.
/// The latest members are checked first: they are the likeliest to dominate it, which keeps long
/// chains of fronts (such as infeasible rows penalised by their violation) linear per pass. With
/// two objectives (`w` = 1) the second objective strictly decreases along a front, so only its
/// last member needs checking.
fn front_levels(
    pts: &[f64],
    w: usize,
    rep: &[usize],
    n: usize,
    nums: usize,
    max_level: usize,
) -> (Vec<usize>, usize) {
    let u = rep.len();
    let row = |i: usize| &pts[i * w..(i + 1) * w];
    let dominates = |j: usize, i: usize| row(j).iter().zip(row(i)).all(|(a, b)| a <= b);
    let mut lv = vec![0usize; u];
    let mut cri_level = 0usize;
    let mut sums = 0usize;
    if nums == 0 {
        return (lv, cri_level);
    }
    let mut front: Vec<usize> = Vec::new();
    // the front's objectives by column, for `front_dominates`
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); w];
    loop {
        cri_level += 1;
        if cri_level > max_level {
            break;
        }
        front.clear();
        cols.iter_mut().for_each(Vec::clear);
        for (i, &l) in lv.iter().enumerate() {
            if l != 0 {
                continue;
            }
            let open = match w {
                // one objective: every earlier member (weakly) dominates
                0 => front.is_empty(),
                1 => front.last().is_none_or(|&j| !dominates(j, i)),
                _ => !front_dominates(&cols, row(i)),
            };
            if open {
                front.push(i);
                for (col, &v) in cols.iter_mut().zip(row(i)) {
                    col.push(v);
                }
            }
        }
        for &i in &front {
            lv[i] = cri_level;
            sums += rep[i];
        }
        if sums >= nums || sums >= n {
            break;
        }
    }
    (lv, cri_level)
}

/// Efficient non-dominated sort (ENS-SS) over unique rows sorted lexicographically, one complete
/// front per pass, compared exactly as geatpy 2.7.0 does. Returns the front of every unique row
/// (0 = unranked) and the last front built. Used for NaN input only: without NaN it gives the same
/// fronts as the faster `front_levels`.
fn ens_levels(
    uobj: &[f64],
    m: usize,
    rep: &[usize],
    n: usize,
    nums: usize,
    max_level: usize,
) -> (Vec<usize>, usize) {
    let u = rep.len();
    let mut lv = vec![0usize; u];
    let mut cri_level = 0usize;
    let mut sums = 0usize;
    if nums > 0 {
        loop {
            cri_level += 1;
            if cri_level > max_level {
                break;
            }
            for i in 0..u {
                if lv[i] != 0 {
                    continue;
                }
                // Rows are sorted on the first objective, so only earlier rows of this front can
                // dominate row i; uniqueness turns weak dominance into dominance. With two
                // objectives the nearest earlier member of the front has the smallest second
                // objective, so geatpy 2.7.0 compares with it alone (this matters only for NaN).
                let dominated = if m == 2 {
                    lv[..i]
                        .iter()
                        .rposition(|&l| l == cri_level)
                        .is_some_and(|j| uobj[j * 2 + 1] <= uobj[i * 2 + 1])
                } else {
                    (0..i).rev().any(|j| {
                        lv[j] == cri_level && (1..m).all(|k| uobj[j * m + k] <= uobj[i * m + k])
                    })
                };
                if !dominated {
                    lv[i] = cri_level;
                    sums += rep[i];
                }
            }
            if sums >= nums || sums >= n {
                break;
            }
        }
    }
    (lv, cri_level)
}

/// For T-ENS: the columns 1..M of every unique row ordered from its worst objective to its best,
/// as geatpy 2.7.0 computes them (`np.argsort(-ObjV[:, 1:], 1) + 1`, NaN last).
fn objective_ranks(py: Python<'_>, uobj: &[f64], m: usize) -> PyResult<Vec<Vec<usize>>> {
    let u = uobj.len() / m;
    if m < 2 {
        return Ok(vec![Vec::new(); u]);
    }
    let neg = Array2::from_shape_fn((u, m - 1), |(i, k)| -uobj[i * m + k + 1]).into_pyarray(py);
    let ranks: Vec<Vec<usize>> = py
        .import("numpy")?
        .call_method1("argsort", (neg, 1))?
        .call_method0("tolist")?
        .extract()?;
    Ok(ranks
        .into_iter()
        .map(|r| r.into_iter().map(|c| c + 1).collect())
        .collect())
}

/// Tree-based ENS (T-ENS, Zhang et al. 2018) as geatpy 2.7.0's ndsortTNS runs it. Without NaN it
/// gives the same fronts as `ens_levels`; with NaN its comparisons decide differently, so it is
/// used for NaN input only. Each front is a tree rooted at its first unranked row; a candidate p
/// is compared with node q on q's objectives from worst to best and skips the subtrees that
/// cannot dominate it.
#[allow(clippy::too_many_arguments)]
fn tens_levels(
    uobj: &[f64],
    m: usize,
    orank: &[Vec<usize>],
    rep: &[usize],
    n: usize,
    nums: usize,
    max_level: usize,
) -> (Vec<usize>, usize) {
    const NONE: usize = usize::MAX;
    let u = rep.len();
    let m1 = orank.first().map_or(0, |r| r.len());
    let mut lv = vec![0usize; u];
    let mut left_child = vec![m1; u];
    let mut father = vec![NONE; u];
    let mut brother = vec![m1; u];
    let mut children = vec![vec![NONE; m1]; u];
    let mut pruning = vec![0usize; u];
    let mut cri_level = 0usize;
    let mut sums = 0usize;
    if nums == 0 {
        return (lv, cri_level);
    }
    loop {
        cri_level += 1;
        if cri_level > max_level {
            break;
        }
        let Some(root) = lv.iter().position(|&l| l == 0) else {
            break;
        };
        lv[root] = cri_level;
        sums += rep[root];
        for p in 0..u {
            if lv[p] != 0 {
                continue;
            }
            pruning.fill(0);
            let mut q = root;
            let dominated = loop {
                // first of q's objectives (worst first) on which q is not <= p
                let k = orank[q]
                    .iter()
                    .take_while(|&&c| uobj[q * m + c] <= uobj[p * m + c])
                    .count();
                if k == m1 {
                    break true;
                }
                pruning[q] = k;
                if left_child[q] <= k {
                    q = children[q][left_child[q]];
                    continue;
                }
                while father[q] != NONE && brother[q] > pruning[father[q]] {
                    q = father[q];
                }
                if father[q] == NONE {
                    break false;
                }
                q = children[father[q]][brother[q]];
            };
            if dominated {
                continue;
            }
            lv[p] = cri_level;
            sums += rep[p];
            let mut q = root;
            while children[q][pruning[q]] != NONE {
                q = children[q][pruning[q]];
            }
            children[q][pruning[q]] = p;
            father[p] = q;
            if left_child[q] > pruning[q] {
                brother[p] = left_child[q];
                left_child[q] = pruning[q];
            } else {
                let mut bro = children[q][left_child[q]];
                while brother[bro] < pruning[q] {
                    bro = children[q][brother[bro]];
                }
                brother[p] = brother[bro];
                brother[bro] = pruning[q];
            }
        }
        if sums >= nums || sums >= n {
            break;
        }
    }
    (lv, cri_level)
}

/// Crowding distance (NSGA-II). Unranked individuals (level Inf) get 0; boundary points get Inf.
#[pyfunction]
#[pyo3(signature = (obj_v, levels, enhance_flag=None, parallel=None))]
pub fn crowdis<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    levels: &Bound<'py, PyAny>,
    enhance_flag: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let _ = (enhance_flag, parallel);
    let obj = crate::utils::to_f64_array2(obj_v)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    let lev = if levels.is_none() {
        Array1::<f64>::ones(n)
    } else {
        crate::utils::to_f64_array1(levels)?
    };
    if lev.len() != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crowdis: The length of levels must equal the number of rows of ObjV.",
        ));
    }
    let mut dist = Array1::<f64>::zeros(n);
    let mut unique_levels: Vec<f64> = lev.iter().cloned().filter(|v| v.is_finite()).collect();
    unique_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_levels.dedup();
    for l in unique_levels {
        let members: Vec<usize> = (0..n).filter(|&i| lev[i] == l).collect();
        let count = members.len();
        if count <= 2 {
            for &i in &members {
                dist[i] = f64::INFINITY;
            }
            continue;
        }
        for j in 0..m {
            let mut sorted = members.clone();
            // stable, NaN last: geatpy 2.7.0 uses np.argsort(kind='mergesort')
            sorted.sort_by(|&a, &b| crate::utils::cmp_nan_last(obj[[a, j]], obj[[b, j]]));
            let (first, last) = (sorted[0], sorted[count - 1]);
            let span = obj[[last, j]] - obj[[first, j]] + 1e-6;
            for k in 1..count - 1 {
                let cur = sorted[k];
                dist[cur] += (obj[[sorted[k + 1], j]] - obj[[sorted[k - 1], j]]) / span;
            }
            // geatpy 2.7.0 sums the per-objective terms with Inf for the boundary points, so a NaN
            // term from another objective keeps a boundary point at NaN
            dist[first] += f64::INFINITY;
            dist[last] += f64::INFINITY;
        }
    }
    Ok(dist.into_pyarray(py))
}

/// Pairwise distances between the rows of A and B: 'euclidean', 'cityblock',
/// 'cosine' (1 - cosine similarity) or 'cosine_similarity'.
#[pyfunction]
#[pyo3(signature = (a, b=None, metric=None, parallel=None))]
pub fn cdist<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: Option<&Bound<'py, PyAny>>,
    metric: Option<&str>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = parallel;
    let arr_a = crate::utils::to_f64_array2(a)?;
    let arr_b = match b {
        Some(x) if !x.is_none() => crate::utils::to_f64_array2(x)?,
        _ => arr_a.clone(),
    };
    let (na, nb, dim) = (arr_a.shape()[0], arr_b.shape()[0], arr_a.shape()[1]);
    if arr_b.shape()[1] != dim {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in cdist: A and B must have the same number of columns.",
        ));
    }
    let metric = metric.unwrap_or("euclidean");
    let unit = |x: &Array2<f64>| -> Array2<f64> {
        let mut y = x.clone();
        for mut r in y.rows_mut() {
            let norm = r.iter().map(|v| v * v).fold(0.0, |s, x| s + x).sqrt();
            r.mapv_inplace(|v| v / norm);
        }
        y
    };
    crate::utils::check_alloc(&[na, nb], 8, "float64")?;
    let mut out = Array2::<f64>::zeros((na, nb));
    // Row i of the result accumulates term(a[i, k], b[j, k]) over k = 0, 1, ... for all j at once
    // from a column-major copy of B, so the loop over j vectorises; every entry is still the sum
    // from 0.0 in order of k.
    let rows =
        |a: &Array2<f64>, b: &Array2<f64>, out: &mut Array2<f64>, term: fn(f64, f64) -> f64| {
            let bt = crate::utils::transpose(b);
            let fa = a.as_slice().expect("converted arrays are row-major");
            let fo = out.as_slice_mut().expect("fresh arrays are row-major");
            for (i, row) in fo.chunks_exact_mut(nb.max(1)).enumerate().take(na) {
                for (k, col) in bt.chunks_exact(nb.max(1)).enumerate().take(dim) {
                    let x = fa[i * dim + k];
                    for (o, &y) in row.iter_mut().zip(col) {
                        *o += term(x, y);
                    }
                }
            }
        };
    match metric {
        "euclidean" => {
            rows(&arr_a, &arr_b, &mut out, |x, y| (x - y) * (x - y));
            out.mapv_inplace(f64::sqrt);
        }
        "cityblock" => rows(&arr_a, &arr_b, &mut out, |x, y| (x - y).abs()),
        "cosine" | "cosine_similarity" => {
            rows(&unit(&arr_a), &unit(&arr_b), &mut out, |x, y| x * y);
            if metric == "cosine" {
                out.mapv_inplace(|dot| 1.0 - dot);
            }
        }
        other => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "error in cdist: unsupported metric '{}' (支持 'euclidean', 'cityblock', 'cosine', 'cosine_similarity')。",
                other
            )))
        }
    }
    Ok(out.into_pyarray(py))
}

/// Merge the columns of CV into one violation column: the sum of the entries above the threshold
/// (0 for a scalar threshold, element-wise for an array threshold). The original 2.7.0 binary
/// accumulates this sum across rows by mistake; here every row is summed on its own, as documented.
#[pyfunction]
#[pyo3(signature = (cv, threshold=None, return_count=None, parallel=None))]
pub fn mergecv<'py>(
    py: Python<'py>,
    cv: &Bound<'py, PyAny>,
    threshold: Option<&Bound<'py, PyAny>>,
    return_count: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let arr = crate::utils::to_f64_array2(cv)?;
    let (n, c) = (arr.shape()[0], arr.shape()[1]);
    let (per_column, scalar) = match crate::utils::knob(threshold)? {
        Some(crate::utils::Knob::Array(v)) => {
            if v.len() != c {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in mergecv: The length of Threshold must equal the number of columns of CV.",
                ));
            }
            (Some(v), 0.0)
        }
        Some(crate::utils::Knob::Scalar(t)) => (None, t),
        _ => (None, 0.0),
    };
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        out[[i, 0]] = (0..c)
            .filter(|&j| match &per_column {
                Some(t) => arr[[i, j]] > t[j],
                None => arr[[i, j]] > 0.0,
            })
            .map(|j| arr[[i, j]])
            .fold(0.0, |s, x| s + x);
    }
    let want_count = return_count
        .map(|o| o.is_truthy())
        .transpose()?
        .unwrap_or(false);
    if want_count {
        let limit = if per_column.is_some() { 0.0 } else { scalar };
        let count = out.iter().filter(|&&v| v <= limit).count();
        let tuple = (out.into_pyarray(py), count);
        return Ok(tuple.into_pyobject(py)?.into_any().unbind());
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// Pairs every row of ObjV with its weight vector; a single ObjV row is broadcast over all weights.
/// The weights must have one column per objective (geatpy 2.7.0's message for either mismatch).
fn decomposition_rows(obj: &Array2<f64>, w: &Array2<f64>, name: &str) -> PyResult<usize> {
    let (n_obj, n_w) = (obj.shape()[0], w.shape()[0]);
    if (n_obj == n_w || n_obj == 1) && w.shape()[1] == obj.shape()[1] {
        Ok(n_w)
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {name}: weights and ObjV disagree. (weights和ObjV不匹配，详见help()帮助文档。)"
        )))
    }
}

fn ideal_or_min(obj: &Array2<f64>, ideal: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
    let m = obj.shape()[1];
    match ideal {
        Some(z) if !z.is_none() => {
            let v = crate::utils::to_f64_array1(z)?;
            if v.len() != m {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error: idealPoint must be a 1-D array with one entry per objective.",
                ));
            }
            Ok(v.to_vec())
        }
        _ => Ok((0..m)
            .map(|j| obj.column(j).iter().cloned().fold(f64::INFINITY, f64::min))
            .collect()),
    }
}

/// Tchebycheff aggregation (MOEA/D): max_j w_j |f_j - z_j|.
#[pyfunction]
#[pyo3(signature = (obj_v, weights, ideal_point=None, cv=None, maxormins=None, parallel=None))]
pub fn tcheby<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    ideal_point: Option<&Bound<'py, PyAny>>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = parallel;
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins, "tcheby")?;
    let w = crate::utils::to_f64_array2(weights)?;
    let rows = decomposition_rows(&obj, &w, "tcheby")?;
    let z = ideal_or_min(&obj, ideal_point)?;
    let m = obj.shape()[1];
    let mut out = Array2::<f64>::zeros((rows, 1));
    for i in 0..rows {
        let r = if obj.shape()[0] == 1 { 0 } else { i };
        out[[i, 0]] = (0..m)
            .map(|j| w[[i, j]] * (obj[[r, j]] - z[j]).abs())
            .fold(f64::NEG_INFINITY, f64::max);
    }
    Ok(out.into_pyarray(py))
}

/// Penalty-based boundary intersection (MOEA/D): d1 + Theta * d2.
#[pyfunction]
#[pyo3(signature = (obj_v, weights, ideal_point=None, cv=None, maxormins=None, theta=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn pbi<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    ideal_point: Option<&Bound<'py, PyAny>>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    theta: Option<f64>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = parallel;
    let theta = theta.unwrap_or(5.0);
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins, "pbi")?;
    let w = crate::utils::to_f64_array2(weights)?;
    let rows = decomposition_rows(&obj, &w, "pbi")?;
    let z = ideal_or_min(&obj, ideal_point)?;
    let m = obj.shape()[1];
    let mut out = Array2::<f64>::zeros((rows, 1));
    for i in 0..rows {
        let r = if obj.shape()[0] == 1 { 0 } else { i };
        let norm = (0..m)
            .map(|j| w[[i, j]] * w[[i, j]])
            .fold(0.0, |s, x| s + x)
            .sqrt();
        let d1 = (0..m)
            .map(|j| (obj[[r, j]] - z[j]) * w[[i, j]])
            .fold(0.0, |s, x| s + x)
            / norm;
        let d2 = (0..m)
            .map(|j| (obj[[r, j]] - z[j] - d1 * w[[i, j]] / norm).powi(2))
            .fold(0.0, |s, x| s + x)
            .sqrt();
        out[[i, 0]] = d1 + theta * d2;
    }
    Ok(out.into_pyarray(py))
}

/// Ideal point of ObjV, optionally merged with an older one. With CV the infeasible rows are first
/// penalised (column maximum + violation, as in every other core operator) and the extreme is then
/// taken over all rows, exactly like geatpy 2.7.0. `reverse` flips the convention to
/// "larger is better" (column maxima).
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, old_ideal_point=None, reverse=None))]
pub fn crtidp<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    old_ideal_point: Option<&Bound<'py, PyAny>>,
    reverse: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins, "crtidp")?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    let reverse = reverse.map(|r| r.is_truthy()).transpose()?.unwrap_or(false);
    let rows: Vec<usize> = (0..n).collect();
    let pick = |a: f64, b: f64| if reverse { a.max(b) } else { a.min(b) };
    let start = if reverse {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut z: Vec<f64> = (0..m)
        .map(|j| rows.iter().map(|&i| obj[[i, j]]).fold(start, pick))
        .collect();
    if let Some(old) = old_ideal_point {
        if !old.is_none() {
            let o = crate::utils::to_f64_array1(old)?;
            for j in 0..m.min(o.len()) {
                z[j] = pick(z[j], o[j]);
            }
        }
    }
    Ok(Array1::from(z).into_pyarray(py))
}

/// Adaptive-weight aggregation (Gen & Cheng awGA), smaller is better.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None))]
pub fn awGA<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins, "awGA")?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    let zp: Vec<f64> = (0..m)
        .map(|j| obj.column(j).iter().cloned().fold(f64::INFINITY, f64::min))
        .collect();
    let zm: Vec<f64> = (0..m)
        .map(|j| {
            obj.column(j)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let weight: Vec<f64> = (0..m)
        .map(|j| {
            (0..n)
                .map(|i| (obj[[i, j]] - zp[j]) / (zm[j] - zp[j] + 0.1))
                .fold(0.0, |s, x| s + x)
        })
        .collect();
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        out[[i, 0]] = (0..m)
            .map(|j| weight[j] * obj[[i, j]])
            .fold(0.0, |s, x| s + x);
    }
    Ok(out.into_pyarray(py))
}

/// Random-weight aggregation: one random weight vector (summing to 1) shared by the population.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None))]
pub fn rwGA<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins, "rwGA")?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    let mut rng = crate::utils::rng();
    let r: Vec<f64> = (0..m).map(|_| rng.gen::<f64>()).collect();
    let total: f64 = r.iter().fold(0.0, |s, x| s + x);
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        out[[i, 0]] = (0..m)
            .map(|j| r[j] / total * obj[[i, j]])
            .fold(0.0, |s, x| s + x);
    }
    Ok(out.into_pyarray(py))
}

/// Sum of squares of a row, accumulated left to right like the original C loops.
fn sq_norm(r: ArrayView1<f64>) -> f64 {
    r.iter().fold(0.0, |s, v| s + v * v)
}

/// Rows divided by their Euclidean norm (geatpy 2.7.0 `_matrix_normalize`).
fn unit_rows(x: &Array2<f64>) -> Array2<f64> {
    let mut u = x.clone();
    for mut r in u.rows_mut() {
        let norm = sq_norm(r.view()).sqrt();
        r.mapv_inplace(|v| v / norm);
    }
    u
}

/// Row `i` of numpy `a @ b.T` for float64 matrices (see `utils::dot_fma_rows`).
fn matmul_row(a: &Array2<f64>, i: usize, b: &Array2<f64>, out: &mut [f64]) {
    let x = a.row(i);
    let x = x.as_slice().expect("owned arrays are contiguous");
    let rows = b.as_slice().expect("owned arrays are contiguous");
    crate::utils::dot_fma_rows(x, rows, out);
}

/// numpy's error for `a @ b.T` when `a` has `m` columns and `b` has `mb`, as geatpy 2.7.0 raises it.
fn matmul_mismatch(mb: usize, m: usize) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature \
         (n?,k),(k,m?)->(n?,m?) (size {mb} is different from {m})"
    ))
}

/// Raised instead of geatpy 2.7.0's arbitrary selection when there is no reference point.
fn no_reference_points() -> PyErr {
    pyo3::exceptions::PyValueError::new_err("refPoint must contain at least one reference point")
}

/// numpy's float64 `np.sum` along a contiguous axis (pairwise summation with 8 accumulators).
fn numpy_pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        a.iter().fold(0.0, |s, &x| s + x)
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let full = n - n % 8;
        for i in (8..full).step_by(8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
        }
        let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        a[full..].iter().fold(res, |s, &x| s + x)
    } else {
        let half = n / 2 - (n / 2) % 8;
        numpy_pairwise_sum(&a[..half]) + numpy_pairwise_sum(&a[half..])
    }
}

/// Index of the pivot `idamax` picks in `v`, as the OpenBLAS 0.3.x arm64 kernel (`iamax.S`) behind
/// numpy's LAPACK does. For NaN-free input this is the first largest |v|. A NaN never survives as
/// the running maximum of the element-wise scan: the next element replaces it, so an exact zero can
/// become the pivot. Past 8 elements the kernel compares blocks of 8 by their `fmax` (NaN if any
/// element is NaN, and such a block always wins), then takes the first element of the winning block
/// equal to the maximum, or the block's first element when the maximum is NaN.
fn openblas_iamax(v: &[f64]) -> usize {
    // the kernel keeps its maximum on `fcmp max, x` + `ge`, which is false when either is NaN
    let keeps = |mx: f64, x: f64| {
        matches!(
            mx.partial_cmp(&x),
            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
        )
    };
    let (mut mx, mut idx) = (v[0].abs(), 0);
    let blocks = (v.len() - 1) / 8;
    for b in 0..blocks {
        let blk = &v[1 + 8 * b..9 + 8 * b];
        let t = if blk.iter().any(|x| x.is_nan()) {
            f64::NAN
        } else {
            blk.iter().fold(0.0, |m: f64, x| m.max(x.abs()))
        };
        if !keeps(mx, t) {
            (mx, idx) = (t, 1 + 8 * b);
        }
    }
    if blocks > 0 {
        let start = idx;
        if let Some(q) = (start..start + 8).find(|&q| v[q].abs() == mx) {
            idx = q;
        }
    }
    for (r, x) in v.iter().enumerate().skip(1 + 8 * blocks) {
        if !keeps(mx, x.abs()) {
            (mx, idx) = (x.abs(), r);
        }
    }
    idx
}

/// Solve A x = rhs by Gaussian elimination with partial pivoting; None when a pivot is exactly zero.
/// Pivots are chosen like numpy's LAPACK (see `openblas_iamax`), so a NaN in A leads to the same
/// zero pivot (geatpy 2.7.0: `det == 0`) or NaN solution as in the original. Dividing by the pivot
/// (LAPACK multiplies by its reciprocal) keeps exactly duplicate rows exactly cancelling, so a
/// singular A with repeated extreme points is reported as singular instead of solved from roundoff.
fn solve_linear(mut a: Array2<f64>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let m = rhs.len();
    for col in 0..m {
        let column: Vec<f64> = (col..m).map(|r| a[[r, col]]).collect();
        let piv = col + openblas_iamax(&column);
        if a[[piv, col]] == 0.0 {
            return None;
        }
        if piv != col {
            for k in 0..m {
                a.swap([piv, k], [col, k]);
            }
            rhs.swap(piv, col);
        }
        for r in col + 1..m {
            let f = a[[r, col]] / a[[col, col]];
            for k in col..m {
                a[[r, k]] -= f * a[[col, k]];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    let mut x = vec![0.0; m];
    for r in (0..m).rev() {
        let s: f64 = (r + 1..m).map(|k| a[[r, k]] * x[k]).fold(0.0, |s, x| s + x);
        x[r] = (rhs[r] - s) / a[[r, r]];
    }
    Some(x)
}

/// NSGA-III environmental selection on the critical front (Deb & Jain). Individuals of the fronts
/// before criLevel are all kept; the remaining quota is filled niche by niche: the least crowded
/// reference point first, the closest individual for an empty niche, otherwise the first candidate.
#[pyfunction]
#[pyo3(signature = (obj_v, levels, cri_level, need_num, ref_point, maxormins=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn refselect<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    levels: &Bound<'py, PyAny>,
    cri_level: &Bound<'py, PyAny>,
    need_num: &Bound<'py, PyAny>,
    ref_point: &Bound<'py, PyAny>,
    maxormins: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let _ = parallel;
    // criLevel is a C int in geatpy 2.7.0 (1.5 -> 1, 2**31 -> OverflowError)
    let cri_level = crate::utils::c_int(cri_level)? as f64;
    let mut obj = crate::utils::to_f64_array2(obj_v)?;
    crate::utils::unify_objectives(&mut obj, maxormins)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    // NUM is compared as a Python object first (any int or float, 2**64 included), then truncated
    if n.into_pyobject(py)?.le(need_num)? {
        return Ok(Array1::from_elem(n, true).into_pyarray(py));
    }
    let need_num = crate::utils::c_long(need_num)?;
    let need_num = usize::try_from(need_num).map_err(|_| {
        pyo3::exceptions::PyOverflowError::new_err("can't convert negative int to unsigned")
    })?;
    let lev = crate::utils::to_f64_array1(levels)?;
    if !lev.is_empty() && lev.len() != n {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "boolean index did not match indexed array along dimension 0; dimension is {} but \
             corresponding boolean dimension is {}",
            n,
            lev.len()
        )));
    }
    let at = |i: usize| lev.get(i).copied().unwrap_or(f64::NAN);
    let z = crate::utils::to_f64_array2(ref_point)?;
    let nz = z.shape()[0];
    let idx1: Vec<usize> = (0..n).filter(|&i| at(i) < cri_level).collect();
    let idx2: Vec<usize> = (0..n).filter(|&i| at(i) == cri_level).collect();
    let (n1, n2) = (idx1.len(), idx2.len());
    if n1 + n2 < need_num {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in refselect: The number of individuals which are in and below the critical \
             level is less than the required number. (处于临界层及其前面各层的个体总数不能小于所需要选择的个体数目。)",
        ));
    }
    if z.shape()[1] != m {
        return Err(matmul_mismatch(z.shape()[1], m));
    }
    let mut flag = Array1::from_elem(n, false);
    for &i in &idx1 {
        flag[i] = true;
    }
    let quota = need_num.saturating_sub(n1);
    if quota == 0 || n2 == 0 {
        return Ok(flag.into_pyarray(py));
    }
    if nz == 0 {
        return Err(no_reference_points());
    }
    // candidates translated by their ideal point
    let rows: Vec<usize> = idx1.iter().chain(idx2.iter()).cloned().collect();
    let mut c = Array2::<f64>::zeros((rows.len(), m));
    for (r, &i) in rows.iter().enumerate() {
        c.row_mut(r).assign(&obj.row(i));
    }
    // Minima, maxima and argmins below follow geatpy 2.7.0: start from the first value and replace
    // it only by a strictly better one, so a leading NaN is kept and later NaNs are skipped.
    for j in 0..m {
        let col = c.column(j);
        let mn = col
            .iter()
            .skip(1)
            .fold(col[0], |mn, &v| if v < mn { v } else { mn });
        c.column_mut(j).mapv_inplace(|v| v - mn);
    }
    // extreme points (achievement scalarising function with weights 1e-6 off the diagonal)
    let extreme: Vec<usize> = (0..m)
        .map(|i| {
            let asf = |r: usize| {
                (1..m).map(|s| (i + s) % m).fold(c[[r, i]], |a, j| {
                    let v = c[[r, j]] / 1e-6;
                    if a < v {
                        v
                    } else {
                        a
                    }
                })
            };
            (1..rows.len())
                .fold((0, asf(0)), |best, r| {
                    let v = asf(r);
                    if v < best.1 {
                        (r, v)
                    } else {
                        best
                    }
                })
                .0
        })
        .collect();
    let mut ext = Array2::<f64>::zeros((m, m));
    for (k, &r) in extreme.iter().enumerate() {
        ext.row_mut(k).assign(&c.row(r));
    }
    // Like the reference implementation, fall back to the column maxima only for a singular system.
    let intercept_scale: Vec<f64> = match solve_linear(ext, vec![1.0; m]) {
        Some(h) => h,
        None => (0..m)
            .map(|j| {
                let col = c.column(j);
                let mx = col
                    .iter()
                    .skip(1)
                    .fold(col[0], |mx, &v| if v > mx { v } else { mx });
                1.0 / if mx <= 1e-6 { mx + 1e-6 } else { mx }
            })
            .collect(),
    };
    for (j, &scale) in intercept_scale.iter().enumerate() {
        c.column_mut(j).mapv_inplace(|v| v * scale);
    }
    // Association with the reference points, rounded like geatpy 2.7.0 so that ties (common with
    // integer objectives) break the same way: rows are normalised first and multiplied with numpy
    // `@` (an FMA chain in OpenBLAS), then each individual goes to the first reference point with
    // the smallest 1 - cos^2, and d = sqrt(|c|^2 (1 - cos^2)) is NaN when roundoff makes it negative.
    let (cu, zu) = (unit_rows(&c), unit_rows(&z));
    let mut pi = vec![0usize; rows.len()];
    let mut d = vec![0.0; rows.len()];
    let mut cos = vec![0.0; nz];
    for r in 0..rows.len() {
        matmul_row(&cu, r, &zu, &mut cos);
        let one_minus_cos2 = |k: usize| 1.0 - cos[k] * cos[k];
        let (best, cm) = (1..nz).fold((0, one_minus_cos2(0)), |b, k| {
            let v = one_minus_cos2(k);
            if v < b.1 {
                (k, v)
            } else {
                b
            }
        });
        pi[r] = best;
        d[r] = (sq_norm(c.row(r)) * cm).sqrt();
    }
    let mut rho = vec![0usize; nz];
    for &p in &pi[..n1] {
        rho[p] += 1;
    }
    let mut excluded = vec![false; nz];
    let mut chosen = vec![false; n2];
    let mut picked = 0;
    while picked < quota {
        let Some(&min_rho) = (0..nz).filter(|&k| !excluded[k]).map(|k| &rho[k]).min() else {
            break;
        };
        let j = (0..nz)
            .find(|&k| !excluded[k] && rho[k] == min_rho)
            .unwrap();
        let cands: Vec<usize> = (0..n2).filter(|&t| pi[n1 + t] == j && !chosen[t]).collect();
        if cands.is_empty() {
            excluded[j] = true;
            continue;
        }
        let s = if rho[j] == 0 {
            // the first closest candidate; like geatpy 2.7.0 only a strictly smaller distance
            // replaces the current one, so NaN distances are never picked after the first
            cands[1..]
                .iter()
                .fold(cands[0], |s, &t| if d[n1 + t] < d[n1 + s] { t } else { s })
        } else {
            cands[0]
        };
        chosen[s] = true;
        rho[j] += 1;
        picked += 1;
    }
    for (t, &i) in idx2.iter().enumerate() {
        if chosen[t] {
            flag[i] = true;
        }
    }
    Ok(flag.into_pyarray(py))
}

type FlagsAndGamma<'py> = (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<f64>>);

/// RVEA reference-vector guided selection (angle-penalised distance). Returns the selection flags
/// and Gamma (smallest angle between each reference vector and the others), reusing Gamma if given.
#[pyfunction]
#[pyo3(signature = (obj_v, ref_point, p_theta, cv=None, gamma=None, maxormins=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn refgselect<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    ref_point: &Bound<'py, PyAny>,
    p_theta: f64,
    cv: Option<&Bound<'py, PyAny>>,
    gamma: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<FlagsAndGamma<'py>> {
    let _ = parallel;
    let mut obj = crate::utils::to_f64_array2(obj_v)?;
    crate::utils::unify_objectives(&mut obj, maxormins)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    // Rounded like geatpy 2.7.0 throughout: minima start from the first value and take only a
    // strictly smaller one, cosines are normalised rows multiplied as numpy `@` (see `matmul_row`).
    for j in 0..m {
        let col = obj.column(j);
        let Some(&first) = col.first() else { continue };
        let mn = col
            .iter()
            .skip(1)
            .fold(first, |mn, &v| if v < mn { v } else { mn });
        obj.column_mut(j).mapv_inplace(|v| v - mn);
    }
    let v = crate::utils::to_f64_array2(ref_point)?;
    let nv = v.shape()[0];
    let vu = unit_rows(&v);
    // Same order as geatpy 2.7.0: Gamma (and its warning), then the violations, then the matrix
    // product that raises numpy's error for reference points of another width.
    let gamma: Vec<f64> = match gamma {
        Some(g) if !g.is_none() => {
            if !g.is_instance_of::<numpy::PyUntypedArray>()
                && !g.is_instance_of::<pyo3::types::PyList>()
                && !g.is_instance_of::<pyo3::types::PyTuple>()
            {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Cannot convert {} to numpy.ndarray",
                    g.get_type().name()?
                )));
            }
            if let Ok(a) = g.downcast::<numpy::PyUntypedArray>() {
                use numpy::PyUntypedArrayMethods;
                if a.ndim() == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Buffer has wrong number of dimensions (expected 1, got 0)",
                    ));
                }
            }
            let g = crate::utils::to_f64_array1(g)?.to_vec();
            // geatpy 2.7.0 reads past the end of a short Gamma
            if g.len() < nv {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "error in refgselect: Gamma must have one element per reference point (got {}, expected {}).",
                    g.len(),
                    nv
                )));
            }
            g
        }
        _ => {
            // cosine between reference vectors with the diagonal zeroed; an exact 1 (a repeated
            // vector) is zeroed too, with the original's warning
            let mut cvv = Array2::<f64>::zeros((nv, nv));
            for (i, mut row) in cvv.rows_mut().into_iter().enumerate() {
                matmul_row(&vu, i, &vu, row.as_slice_mut().unwrap());
                row[i] = 0.0;
            }
            if cvv.iter().any(|&x| x == 1.0) {
                // stack level 2: the caller of the geatpy.core wrapper, like the original's warning
                PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
                    c"Warning in refgselect: Some reference points are so close to the original point.(部分参考点过于接近坐标原点，此时将受到浮点精度限制的影响。)",
                    2,
                )?;
                cvv.mapv_inplace(|x| if x == 1.0 { 0.0 } else { x });
            }
            // the largest cosine (first value, then strictly larger ones) and its angle; acos is not
            // clamped, so a cosine rounded above 1 gives NaN as in the original
            (0..nv)
                .map(|i| {
                    let row = cvv.row(i);
                    row.iter()
                        .skip(1)
                        .fold(row[0], |mx, &x| if x > mx { x } else { mx })
                        .acos()
                })
                .collect()
        }
    };
    // Total violation as the original computes it: negative entries set to 0, then `np.sum(_, 1)`,
    // so a NaN stays NaN (and counts as infeasible below). Unlike the original the caller's CV is
    // left unchanged.
    let viol: Option<Array2<f64>> = match cv {
        Some(c) if !c.is_none() => Some(crate::utils::to_f64_array2(c)?),
        _ => None,
    };
    if v.shape()[1] != m {
        return Err(matmul_mismatch(v.shape()[1], m));
    }
    // geatpy 2.7.0 does not check the rows of CV (it reads past the end of a short one)
    let viol: Vec<f64> = match viol {
        Some(a) => {
            if a.shape()[0] != n {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in refgselect: CV and ObjV disagree. (违反约束程度矩阵的行数必须和目标函数值矩阵的行数相同。)",
                ));
            }
            a.rows()
                .into_iter()
                .map(|r| {
                    let r: Vec<f64> = r.iter().map(|&x| if x < 0.0 { 0.0 } else { x }).collect();
                    numpy_pairwise_sum(&r)
                })
                .collect()
        }
        None => vec![0.0; n],
    };
    if nv == 0 && n > 0 {
        return Err(no_reference_points());
    }
    let ou = unit_rows(&obj);
    let norm: Vec<f64> = obj.rows().into_iter().map(|r| sq_norm(r).sqrt()).collect();
    // cosine distance 1 - cos; each individual goes to the first reference vector with the smallest
    let mut cos_dist = Array2::<f64>::zeros((n, nv));
    for (i, mut row) in cos_dist.rows_mut().into_iter().enumerate() {
        let out = row.as_slice_mut().unwrap();
        matmul_row(&ou, i, &vu, out);
        out.iter_mut().for_each(|x| *x = 1.0 - *x);
    }
    let assoc: Vec<usize> = (0..n)
        .map(|i| {
            (1..nv)
                .fold((0, cos_dist[[i, 0]]), |b, k| {
                    if cos_dist[[i, k]] < b.1 {
                        (k, cos_dist[[i, k]])
                    } else {
                        b
                    }
                })
                .0
        })
        .collect();
    let mut flag = Array1::from_elem(n, false);
    let mut refs: Vec<usize> = assoc.clone();
    refs.sort();
    refs.dedup();
    for k in refs {
        let members: Vec<usize> = (0..n).filter(|&i| assoc[i] == k).collect();
        let feasible: Vec<usize> = members
            .iter()
            .cloned()
            .filter(|&i| viol[i] <= 1e-12)
            .collect();
        // As in geatpy 2.7.0 `cal_choose_flag`, the first member is the initial best and a later one
        // replaces it only when strictly better; an individual at the ideal point (zero vector, APD
        // NaN) therefore survives when it comes first in its group.
        let first_strictly_best = |cands: &[usize], key: &dyn Fn(usize) -> f64| -> Option<usize> {
            let (&first, rest) = cands.split_first()?;
            Some(
                rest.iter()
                    .fold((first, key(first)), |b, &i| {
                        let v = key(i);
                        if v < b.1 {
                            (i, v)
                        } else {
                            b
                        }
                    })
                    .0,
            )
        };
        let best = if !feasible.is_empty() {
            let w = p_theta / gamma[k];
            let apd = |i: usize| (w * (1.0 - cos_dist[[i, k]]).acos() + 1.0) * norm[i];
            first_strictly_best(&feasible, &apd)
        } else {
            first_strictly_best(&members, &|i: usize| viol[i])
        };
        if let Some(best) = best {
            flag[best] = true;
        }
    }
    Ok((flag.into_pyarray(py), Array1::from(gamma).into_pyarray(py)))
}

/// Migration between sub-populations. Each population sends max(1, floor(MIGR * size)) individuals
/// (random, or the fittest with Select=1) to the population determined by Structure (0: all
/// populations shifted by one random offset, 1: a random neighbour, 2: ring). Replacement 0 moves
/// the emigrants out; 1 and 2 copy them and drop as many residents, randomly (1) or the least fit (2).
#[pyfunction]
#[pyo3(signature = (pop_sizes, migr=None, structure=None, select=None, replacement=None, fitn_vs=None, separate=None))]
#[allow(clippy::too_many_arguments)]
pub fn migrate<'py>(
    py: Python<'py>,
    pop_sizes: Vec<usize>,
    migr: Option<f64>,
    structure: Option<i64>,
    select: Option<i64>,
    replacement: Option<i64>,
    fitn_vs: Option<&Bound<'py, PyAny>>,
    separate: Option<bool>,
) -> PyResult<PyObject> {
    let p = pop_sizes.len();
    let migr = migr.unwrap_or(0.2);
    let (structure, select, replacement) = (
        structure.unwrap_or(0),
        select.unwrap_or(0),
        replacement.unwrap_or(0),
    );
    let checks = [
        (
            p > 1,
            "The number of populations must be bigger than 1. (种群数目必须大于1。)",
        ),
        (
            pop_sizes.iter().all(|&s| s > 0),
            "The size of every population must be bigger than 0. (种群规模必须大于0。)",
        ),
        (
            migr > 0.0 && migr <= 1.0,
            "Migration rate must be in (0, 1]. (子种群间的迁移概率MIGR必须在(0, 1]之间。)",
        ),
        (
            (0..=2).contains(&structure),
            "Structure must be 0, 1 or 2. (迁移种群的结构Structure必须是0, 1或2。)",
        ),
        (
            (0..=1).contains(&select),
            "Select must be 0 or 1. (输入参数Select必须是0或1。)",
        ),
        (
            (0..=2).contains(&replacement),
            "Replacement must be 0, 1 or 2. (输入参数Replacement必须是0，1或2。)",
        ),
    ];
    if let Some((_, msg)) = checks.iter().find(|(ok, _)| !ok) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in migrate: {}",
            msg
        )));
    }
    // the index lists below hold every individual
    let total = pop_sizes
        .iter()
        .try_fold(0usize, |t, &s| t.checked_add(s))
        .unwrap_or(usize::MAX);
    crate::utils::check_alloc(&[total], 8, "int64")?;
    // FitnVs is read only when fitness is used, like geatpy 2.7.0 (which ignores it otherwise)
    let needs_fitness = select == 1 || replacement == 2;
    let fits: Option<Vec<Vec<f64>>> = match fitn_vs {
        Some(f) if needs_fitness && !f.is_none() => {
            let fits: Vec<Vec<f64>> = f
                .try_iter()?
                .map(|x| {
                    let m = crate::utils::to_f64_array2(&x?)?;
                    if m.ncols() == 0 {
                        // numpy's message for FitnV[:, 0] on an empty column set
                        return Err(pyo3::exceptions::PyIndexError::new_err(
                            "index 0 is out of bounds for axis 0 with size 0",
                        ));
                    }
                    Ok(m.column(0).to_vec())
                })
                .collect::<PyResult<_>>()?;
            // geatpy 2.7.0 reads past the end of a missing or short FitnV
            if fits.len() < p || fits.iter().zip(&pop_sizes).any(|(f, &s)| f.len() < s) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in migrate: FitnVs and PopSizes disagree. (FitnVs与PopSizes不匹配：每个种群都需要一个至少有PopSizes[i]行的FitnV。)",
                ));
            }
            Some(fits)
        }
        _ => None,
    };
    if needs_fitness && fits.is_none() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in migrate: FitnVs must not be None when Select=1 or Replacement=2 . (当Select=1或Replacement=2时，必须传入FitnVs参数。)",
        ));
    }
    let mut rng = crate::utils::rng();
    let by_fitness_desc = |i: usize| -> Vec<usize> {
        let f = &fits.as_ref().unwrap()[i];
        let mut o: Vec<usize> = (0..pop_sizes[i]).collect();
        // np.argsort(-FitnV): best first, NaN last
        o.sort_by(|&a, &b| crate::utils::cmp_nan_last(-f[a], -f[b]));
        o
    };
    let n_emig: Vec<usize> = pop_sizes
        .iter()
        .map(|&s| ((migr * s as f64).floor() as usize).max(1).min(s))
        .collect();
    // emigrants of every population
    let emigrants: Vec<Vec<usize>> = (0..p)
        .map(|i| {
            if select == 1 {
                by_fitness_desc(i).into_iter().take(n_emig[i]).collect()
            } else {
                let mut v: Vec<usize> = (0..pop_sizes[i]).collect();
                v.shuffle(&mut rng);
                v.truncate(n_emig[i]);
                v
            }
        })
        .collect();
    let shift = if p > 1 { rng.gen_range(1..p) } else { 0 };
    let from: Vec<usize> = (0..p)
        .map(|i| match structure {
            1 if p > 1 => {
                if rng.gen::<bool>() {
                    (i + p - 1) % p
                } else {
                    (i + 1) % p
                }
            }
            2 => (i + p - 1) % p,
            _ => (i + p - shift) % p,
        })
        .collect();
    let mut aborigines: Vec<Vec<usize>> = Vec::with_capacity(p);
    for i in 0..p {
        let keep = match replacement {
            0 => (0..pop_sizes[i])
                .filter(|x| !emigrants[i].contains(x))
                .collect(),
            2 => by_fitness_desc(i)
                .into_iter()
                .take(pop_sizes[i].saturating_sub(n_emig[from[i]]))
                .collect(),
            _ => {
                let mut v: Vec<usize> = (0..pop_sizes[i]).collect();
                v.shuffle(&mut rng);
                v.truncate(pop_sizes[i].saturating_sub(n_emig[from[i]]));
                v
            }
        };
        aborigines.push(keep);
    }
    let foreigners: Vec<Vec<usize>> = (0..p).map(|i| emigrants[from[i]].clone()).collect();
    // Python lists of indices, like geatpy 2.7.0.
    let to_arr = |v: Vec<usize>| pyo3::types::PyList::new(py, v.into_iter().map(|x| x as i64));
    if separate.unwrap_or(true) {
        let ab = pyo3::types::PyList::empty(py);
        let fo = pyo3::types::PyList::empty(py);
        for i in 0..p {
            ab.append(to_arr(aborigines[i].clone())?)?;
            fo.append(to_arr(foreigners[i].clone())?)?;
        }
        let fp = Array1::from_iter(from.iter().map(|&x| x as i32)).into_pyarray(py);
        return Ok((ab, fo, fp).into_pyobject(py)?.into_any().unbind());
    }
    let offsets: Vec<usize> = pop_sizes
        .iter()
        .scan(0, |acc, &s| {
            let o = *acc;
            *acc += s;
            Some(o)
        })
        .collect();
    let out = pyo3::types::PyList::empty(py);
    for i in 0..p {
        let mut v: Vec<usize> = aborigines[i].iter().map(|&x| x + offsets[i]).collect();
        v.extend(foreigners[i].iter().map(|&x| x + offsets[from[i]]));
        out.append(to_arr(v)?)?;
    }
    Ok(out.into_any().unbind())
}
