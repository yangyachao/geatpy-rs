//! Recombination and crossover operators.
//!
//! Pairing follows geatpy: with Half_N False/None the first half of the population is paired with
//! the second half (individual i with i + N/2) and both offspring are returned in place (an odd last
//! individual is copied); Half_N True returns only the first offspring of every pair; a positive
//! integer Half_N switches to global mode, producing that many offspring from random parents.

use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rand::seq::SliceRandom;
use rand::Rng;

use crate::utils::{chrom_out, is_integer_like, to_f64_array1, to_f64_array2};

type Opt<'a, 'py> = Option<&'a Bound<'py, PyAny>>;

#[derive(Clone, Copy)]
enum Pairing {
    Full,
    Half,
    Global(usize),
}

fn pairing(half_n: Opt, n: usize, name: &str) -> PyResult<Pairing> {
    match half_n {
        None => Ok(Pairing::Full),
        Some(o) if o.is_none() => Ok(Pairing::Full),
        Some(o) => {
            let type_name = o.get_type().name()?.to_string();
            if o.is_instance_of::<pyo3::types::PyBool>()
                || type_name == "bool_"
                || type_name == "bool"
            {
                return Ok(if o.is_truthy()? {
                    Pairing::Half
                } else {
                    Pairing::Full
                });
            }
            // a positive int (geatpy 2.7.0 rejects floats such as 3.0, and anything that is not a
            // number, with the same message)
            let k: f64 = if o.is_instance_of::<pyo3::types::PyFloat>() {
                0.0
            } else {
                o.extract().unwrap_or(0.0)
            };
            if k < 1.0 || k.fract() != 0.0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "error in {}: Half_N error. (传入参数Half_N有误，详见API文档。)",
                    name
                )));
            }
            Ok(Pairing::Global((k as usize).min(n)))
        }
    }
}

/// Crossover probability `XOVR` (or `RecOpt` for recint/reclin/recdis), which must lie in [0, 1].
fn prob(obj: Opt, default: f64, name: &str) -> PyResult<f64> {
    let p: f64 = match obj {
        Some(o) if !o.is_none() => o.extract()?,
        _ => default,
    };
    if !(0.0..=1.0).contains(&p) {
        let msg = if matches!(name, "recint" | "reclin" | "recdis") {
            "RecOpt must be in [0,1]. (重组概率RecOpt必须在[0, 1]之间。)"
        } else {
            "XOVR must be in [0,1]. (交叉概率XOVR必须在[0, 1]之间。)"
        };
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: {}",
            name, msg
        )));
    }
    Ok(p)
}

/// Non-negative scalar or per-gene parameter (recsbx `n`, recndx `A`).
fn non_negative(obj: Opt, l: usize, default: f64, op: &str, name: &str) -> PyResult<Vec<f64>> {
    let v = crate::utils::per_gene(obj, l, default, name)?;
    if v.iter().any(|&x| x < 0.0 || x.is_nan()) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: {} must be or bigger than 0. (当输入参数{}为一个数时，它必须不小于0。)",
            op, name, name
        )));
    }
    Ok(v)
}

/// Element type of a permutation chromosome: integer arrays are processed as i32 without a round
/// trip through f64.
trait Gene: Copy + PartialEq + PartialOrd + Default {
    /// The gene as an integer, or None when it is not a whole number.
    fn whole(self) -> Option<i64>;

    /// The smallest and largest gene as integers (None when one is not a whole number); i64::MAX
    /// and i64::MIN for no genes.
    fn whole_range(genes: &[Self]) -> Option<(i64, i64)> {
        let (mut min, mut max) = (i64::MAX, i64::MIN);
        for &v in genes {
            let v = v.whole()?;
            min = min.min(v);
            max = max.max(v);
        }
        Some((min, max))
    }
}

impl Gene for f64 {
    fn whole(self) -> Option<i64> {
        (self.is_finite() && self.fract() == 0.0 && self.abs() < 9e15).then_some(self as i64)
    }
}

impl Gene for i32 {
    fn whole(self) -> Option<i64> {
        Some(self as i64)
    }

    fn whole_range(genes: &[Self]) -> Option<(i64, i64)> {
        // every i32 is whole, so this is a plain (vectorised) min / max
        let min = genes.iter().copied().min().map_or(i64::MAX, i64::from);
        let max = genes.iter().copied().max().map_or(i64::MIN, i64::from);
        Some((min, max))
    }
}

/// geatpy 2.7.0 checks that every row of a permutation chromosome holds distinct genes.
fn check_permutation<T: Gene>(chrom: &Array2<T>, name: &str) -> PyResult<()> {
    let err = || {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: Chromosome error. (染色体不满足排列编码规则。)",
            name
        ))
    };
    let l = chrom.ncols();
    if let (Some(flat), true) = (chrom.as_slice(), l > 0) {
        // the usual case: one index for the whole matrix, and a table of the last row that held
        // each value, so it never has to be cleared
        if let Some(index) = DenseIndex::new(flat, &[]).filter(|ix| ix.size <= 8 * l + 16) {
            let mut last = vec![0usize; index.size];
            for (r, row) in flat.chunks_exact(l).enumerate() {
                for &v in row {
                    let k = index.key(v);
                    if last[k] == r + 1 {
                        return Err(err());
                    }
                    last[k] = r + 1;
                }
            }
            return Ok(());
        }
    }
    let mut seen = Vec::new();
    for row in chrom.rows() {
        let owned;
        let row: &[T] = match row.as_slice() {
            Some(r) => r,
            None => {
                owned = row.to_vec();
                &owned
            }
        };
        if let Some(index) = DenseIndex::new(row, &[]) {
            // small integer values (the usual 0..L-1 / 1..L): a bitmap finds duplicates in O(L)
            seen.clear();
            seen.resize(index.size, false);
            for &v in row {
                let k = index.key(v);
                if seen[k] {
                    return Err(err());
                }
                seen[k] = true;
            }
        } else {
            let mut v = row.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if v.windows(2).any(|w| w[0] == w[1]) {
                return Err(err());
            }
        }
    }
    Ok(())
}

/// Maps the values of permutation rows to dense slots when they are all integers within a small
/// range, so positions can be looked up in O(1) instead of searching the row.
#[derive(Clone, Copy)]
struct DenseIndex {
    min: i64,
    size: usize,
}

impl DenseIndex {
    fn new<T: Gene>(a: &[T], b: &[T]) -> Option<Self> {
        let (min_a, max_a) = T::whole_range(a)?;
        let (min_b, max_b) = T::whole_range(b)?;
        let (min, max) = (min_a.min(min_b), max_a.max(max_b));
        let size = if a.is_empty() && b.is_empty() {
            0
        } else {
            (max - min) as usize + 1
        };
        (size <= 4 * (a.len() + b.len()) + 16).then_some(DenseIndex { min, size })
    }

    /// Slot of a gene seen by `new` (always a whole number in range).
    fn key<T: Gene>(&self, v: T) -> usize {
        (v.whole().unwrap_or(self.min) - self.min) as usize
    }
}

fn random_pair<R: Rng>(n: usize, rng: &mut R) -> (usize, usize) {
    let a = rng.gen_range(0..n);
    let b = if n > 1 {
        (a + 1 + rng.gen_range(0..n - 1)) % n
    } else {
        a
    };
    (a, b)
}

/// Length of an exponential-crossover section: geometric with ratio pc, at least 1, at most l.
fn exp_len<R: Rng>(pc: f64, l: usize, rng: &mut R) -> usize {
    if pc >= 1.0 {
        return l;
    }
    if pc <= 0.0 {
        return 1;
    }
    let u: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    ((u.ln() / pc.ln()).ceil().max(1.0) as usize).min(l)
}

#[derive(Clone, Copy, PartialEq)]
enum CopyOp {
    SinglePoint,
    DoublePoint,
    Uniform,
    Shuffle,
    ShuffleExp,
    Exponential,
    Binomial,
    Discrete,
}

impl CopyOp {
    /// Whether the operator first decides, with probability pc, if a pair crosses at all.
    fn pair_level(self) -> bool {
        matches!(
            self,
            CopyOp::SinglePoint | CopyOp::DoublePoint | CopyOp::Shuffle
        )
    }

    /// The gene range a single- or two-point crossover exchanges, drawn after the pair-level
    /// probability check (None for other operators, or when the chromosome is too short).
    fn point_range<R: Rng>(self, l: usize, rng: &mut R) -> Option<std::ops::Range<usize>> {
        match self {
            CopyOp::SinglePoint if l >= 2 => Some(rng.gen_range(0..l - 1) + 1..l),
            CopyOp::DoublePoint if l >= 1 => {
                let (mut a, mut b) = (rng.gen_range(0..l), rng.gen_range(0..l));
                if b < a {
                    std::mem::swap(&mut a, &mut b);
                }
                Some(a..b + 1)
            }
            _ => None,
        }
    }

    /// Genes the first offspring takes from the partner (None: the pair does not cross).
    fn mask<R: Rng>(self, l: usize, pc: f64, rng: &mut R, forced: bool) -> Option<Vec<bool>> {
        if l == 0 {
            return None;
        }
        if self.pair_level() && !forced && rng.gen::<f64>() >= pc {
            return None;
        }
        let mut m = vec![false; l];
        match self {
            CopyOp::SinglePoint | CopyOp::DoublePoint => {
                let r = self.point_range(l, rng)?;
                m[r].iter_mut().for_each(|x| *x = true);
            }
            CopyOp::Uniform | CopyOp::Discrete => {
                m.iter_mut().for_each(|x| *x = rng.gen::<f64>() < pc)
            }
            CopyOp::Shuffle => {
                if l < 2 {
                    return None;
                }
                let mut perm: Vec<usize> = (0..l).collect();
                perm.shuffle(rng);
                let point = rng.gen_range(0..l - 1);
                perm[point + 1..].iter().for_each(|&j| m[j] = true);
            }
            CopyOp::ShuffleExp => {
                let mut perm: Vec<usize> = (0..l).collect();
                perm.shuffle(rng);
                let start = rng.gen_range(0..l);
                let len = exp_len(pc, l, rng);
                (0..len).for_each(|t| m[perm[(start + t) % l]] = true);
            }
            CopyOp::Exponential => {
                let start = rng.gen_range(0..l);
                let len = exp_len(pc, l, rng);
                (0..len).for_each(|t| m[(start + t) % l] = true);
            }
            CopyOp::Binomial => {
                let jrand = rng.gen_range(0..l);
                for (j, x) in m.iter_mut().enumerate() {
                    *x = j == jrand || rng.gen::<f64>() < pc;
                }
            }
        }
        Some(m)
    }
}

/// For every offspring row and every (virtual) gene, the parent row it is copied from.
fn copy_sources(op: CopyOp, n: usize, l: usize, pc: f64, mode: Pairing) -> Vec<Vec<usize>> {
    let mut rng = crate::utils::rng();
    let half = n / 2;
    match mode {
        // Global uniform/discrete recombination: each (virtual) gene comes from a random individual
        // of the whole population, and the probability is not used (geatpy 2.7.0 `*_gb`).
        Pairing::Global(k) if matches!(op, CopyOp::Uniform | CopyOp::Discrete) => (0..k)
            .map(|_| (0..l).map(|_| rng.gen_range(0..n)).collect())
            .collect(),
        Pairing::Global(k) => (0..k)
            .map(|_| {
                let (a, b) = random_pair(n, &mut rng);
                match op.mask(l, pc, &mut rng, true) {
                    Some(m) => m.iter().map(|&t| if t { b } else { a }).collect(),
                    None => vec![a; l],
                }
            })
            .collect(),
        Pairing::Half => (0..half)
            .map(|i| match op.mask(l, pc, &mut rng, false) {
                Some(m) => m.iter().map(|&t| if t { i + half } else { i }).collect(),
                None => vec![i; l],
            })
            .collect(),
        Pairing::Full => {
            let mut out: Vec<Vec<usize>> = (0..n).map(|i| vec![i; l]).collect();
            for i in 0..half {
                let j = i + half;
                if let Some(m) = op.mask(l, pc, &mut rng, false) {
                    // discrete recombination draws an independent mask for the second offspring
                    let m2 = if op == CopyOp::Discrete {
                        op.mask(l, pc, &mut rng, false).unwrap()
                    } else {
                        m.clone()
                    };
                    out[i] = m.iter().map(|&t| if t { j } else { i }).collect();
                    out[j] = m2.iter().map(|&t| if t { i } else { j }).collect();
                }
            }
            out
        }
    }
}

/// Pairwise crossover (individual i with i + N/2) applied directly to the chromosome matrix: the
/// same masks as `copy_sources` for `Pairing::Full` / `Pairing::Half`, without building the
/// per-gene source table. With `half_only` only the first offspring of every pair is returned.
fn paired_crossover<T: Copy>(
    chrom: Array2<T>,
    op: CopyOp,
    pc: f64,
    half_only: bool,
    seg: &[usize],
    virtual_len: usize,
) -> Array2<T> {
    let mut rng = crate::utils::rng();
    let (n, l) = chrom.dim();
    let half = n / 2;
    // rows are crossed in place: row i only ever reads row i + N/2 and vice versa
    let mut buf = if chrom.is_standard_layout() {
        chrom.into_raw_vec_and_offset().0
    } else {
        chrom.iter().copied().collect()
    };
    let identity = seg.len() == virtual_len && seg.iter().enumerate().all(|(g, &s)| g == s);
    // per-gene mask from the (virtual) segment mask
    let per_gene = |m: Vec<bool>| -> Vec<bool> {
        if identity {
            m
        } else {
            seg.iter().map(|&s| m[s]).collect()
        }
    };
    let point_op = identity && matches!(op, CopyOp::SinglePoint | CopyOp::DoublePoint);
    for i in 0..half {
        let j = i + half;
        let (lo, hi) = buf.split_at_mut(j * l);
        let (ri, rj) = (&mut lo[i * l..(i + 1) * l], &mut hi[..l]);
        if point_op {
            // same draws as `mask`, but the exchanged genes are one contiguous slice
            if l == 0 || rng.gen::<f64>() >= pc {
                continue;
            }
            let Some(r) = op.point_range(l, &mut rng) else {
                continue;
            };
            if half_only {
                ri[r.clone()].copy_from_slice(&rj[r]);
            } else {
                ri[r.clone()].swap_with_slice(&mut rj[r]);
            }
            continue;
        }
        let Some(m) = op.mask(virtual_len, pc, &mut rng, false) else {
            continue;
        };
        // discrete recombination draws an independent mask for the second offspring
        let m2 = if !half_only && op == CopyOp::Discrete {
            op.mask(virtual_len, pc, &mut rng, false).map(per_gene)
        } else {
            None
        };
        let m = per_gene(m);
        match (half_only, m2) {
            (true, _) => {
                for ((x, &y), &t) in ri.iter_mut().zip(rj.iter()).zip(&m) {
                    *x = if t { y } else { *x };
                }
            }
            (false, None) => {
                for ((x, y), &t) in ri.iter_mut().zip(rj.iter_mut()).zip(&m) {
                    let (a, b) = if t { (*y, *x) } else { (*x, *y) };
                    (*x, *y) = (a, b);
                }
            }
            (false, Some(m2)) => {
                for (((x, y), &t), &t2) in ri.iter_mut().zip(rj.iter_mut()).zip(&m).zip(&m2) {
                    let (a, b) = (*x, *y);
                    *x = if t { b } else { a };
                    *y = if t2 { a } else { b };
                }
            }
        }
    }
    let rows = if half_only { half } else { n };
    buf.truncate(rows * l);
    Array2::from_shape_vec((rows, l), buf).expect("shape matches")
}

/// Segment index of every gene for GeneID (same ID => crossed as a whole).
fn gene_segments(gene_id: Opt, l: usize) -> PyResult<Option<(Vec<usize>, usize)>> {
    match gene_id {
        Some(g) if !g.is_none() => {
            let ids = to_f64_array1(g)?;
            if ids.len() != l {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error: The length of GeneID must equal the chromosome length.",
                ));
            }
            // Segments are ordered by ID value (geatpy 2.7.0 sorts the genes by GeneID first), so
            // position-based crossovers cut between consecutive IDs, not in order of appearance.
            // geatpy 2.7.0 casts GeneID to int32 (NaN becomes 0, as numpy does on arm64)
            let ids: Vec<f64> = ids.iter().map(|&v| f64::from(v as i32)).collect();
            let mut uniq: Vec<f64> = ids.clone();
            uniq.sort_by(f64::total_cmp);
            uniq.dedup();
            let seg = ids
                .iter()
                .map(|&v| uniq.iter().position(|&s| s == v).unwrap())
                .collect();
            Ok(Some((seg, uniq.len())))
        }
        _ => Ok(None),
    }
}

/// Runs a value-copying crossover on a chromosome matrix or on a list of chromosome matrices
/// (in which case whole chromosomes are exchanged).
fn copy_crossover<'py>(
    py: Python<'py>,
    name: &str,
    op: CopyOp,
    old: &Bound<'py, PyAny>,
    pc: f64,
    half_n: Opt,
    gene_id: Opt,
) -> PyResult<PyObject> {
    if let Ok(list) = old.downcast::<PyList>() {
        let mats: Vec<(Array2<f64>, bool)> = list
            .iter()
            .map(|m| Ok((to_f64_array2(&m)?, is_integer_like(&m))))
            .collect::<PyResult<_>>()?;
        let n = mats.first().map(|m| m.0.shape()[0]).unwrap_or(0);
        let src = copy_sources(op, n, mats.len(), pc, pairing(half_n, n, name)?);
        let out = PyList::empty(py);
        for (c, (m, as_int)) in mats.iter().enumerate() {
            let mut new = Array2::<f64>::zeros((src.len(), m.shape()[1]));
            for (r, s) in src.iter().enumerate() {
                new.row_mut(r).assign(&m.row(s[c]));
            }
            out.append(chrom_out(py, new, *as_int)?)?;
        }
        return Ok(out.into_any().unbind());
    }
    let as_int = is_integer_like(old);
    let int_chrom = if as_int {
        crate::utils::to_i32_array2(old)
    } else {
        None
    };
    let chrom = match int_chrom {
        Some(_) => None,
        None => Some(to_f64_array2(old)?),
    };
    let shape = match (&int_chrom, &chrom) {
        (Some(c), _) => c.dim(),
        (_, Some(c)) => c.dim(),
        _ => unreachable!(),
    };
    let (n, l) = shape;
    let mode = pairing(half_n, n, name)?;
    let (seg, virtual_len) = match gene_segments(gene_id, l)? {
        Some((s, k)) => (s, k),
        None => ((0..l).collect(), l),
    };
    if !matches!(mode, Pairing::Global(_)) {
        let half_only = matches!(mode, Pairing::Half);
        return match (int_chrom, chrom) {
            (Some(c), _) => {
                let out = paired_crossover(c, op, pc, half_only, &seg, virtual_len);
                Ok(numpy::IntoPyArray::into_pyarray(out, py)
                    .into_any()
                    .unbind())
            }
            (_, Some(c)) => {
                let out = paired_crossover(c, op, pc, half_only, &seg, virtual_len);
                chrom_out(py, out, as_int)
            }
            _ => unreachable!(),
        };
    }
    let chrom = match (int_chrom, chrom) {
        (Some(c), _) => c.mapv(f64::from),
        (_, Some(c)) => c,
        _ => unreachable!(),
    };
    let src = copy_sources(op, n, virtual_len, pc, mode);
    let mut out = Array2::<f64>::zeros((src.len(), l));
    for (r, s) in src.iter().enumerate() {
        for j in 0..l {
            out[[r, j]] = chrom[[s[seg[j]], j]];
        }
    }
    chrom_out(py, out, as_int)
}

macro_rules! copy_operator {
    ($name:ident, $op:expr, $default_pc:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (old_chrom, xovr=None, half_n=None, gene_id=None, parallel=None))]
        pub fn $name<'py>(
            py: Python<'py>,
            old_chrom: &Bound<'py, PyAny>,
            xovr: Opt<'_, 'py>,
            half_n: Opt<'_, 'py>,
            gene_id: Opt<'_, 'py>,
            parallel: Opt<'_, 'py>,
        ) -> PyResult<PyObject> {
            let _ = parallel;
            let name = stringify!($name);
            copy_crossover(
                py,
                name,
                $op,
                old_chrom,
                prob(xovr, $default_pc, name)?,
                half_n,
                gene_id,
            )
        }
    };
}

copy_operator!(xovsp, CopyOp::SinglePoint, 0.7, "Single-point crossover.");
copy_operator!(
    xovdp,
    CopyOp::DoublePoint,
    0.7,
    "Two-point crossover (inclusive segment)."
);
copy_operator!(
    xovud,
    CopyOp::Uniform,
    0.5,
    "Uniform crossover: every gene is exchanged with probability XOVR."
);
copy_operator!(
    xovsh,
    CopyOp::Shuffle,
    0.7,
    "Shuffle crossover: single-point crossover on a random gene order."
);
copy_operator!(
    xovsec,
    CopyOp::ShuffleExp,
    0.7,
    "Shuffle exponential crossover (Tanabe & Fukunaga)."
);
copy_operator!(
    xovexp,
    CopyOp::Exponential,
    0.7,
    "Exponential crossover (DE): a circular section of geometric length."
);
copy_operator!(
    xovbd,
    CopyOp::Binomial,
    0.7,
    "Binomial crossover (DE): one forced gene plus each other gene with probability XOVR."
);

/// Discrete recombination: every gene of each offspring comes from the partner with probability RecOpt.
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=None, half_n=None, gene_id=None, parallel=None))]
pub fn recdis<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    gene_id: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    copy_crossover(
        py,
        "recdis",
        CopyOp::Discrete,
        old_chrom,
        prob(rec_opt, 0.5, "recdis")?,
        half_n,
        gene_id,
    )
}

/// A chromosome matrix as row-major f64: numpy's own buffer when it already is a C-contiguous
/// float64 matrix (no copy), a converted copy otherwise.
enum F64Rows<'py> {
    View(numpy::PyReadonlyArray2<'py, f64>),
    Owned(Array2<f64>),
}

impl<'py> F64Rows<'py> {
    fn new(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(ro) = obj.extract::<numpy::PyReadonlyArray2<'py, f64>>() {
            if ro.as_slice().is_ok() {
                use numpy::PyUntypedArrayMethods;
                // the size check the conversion makes first
                crate::utils::check_alloc(ro.shape(), 8, "float64")?;
                return Ok(F64Rows::View(ro));
            }
        }
        Ok(F64Rows::Owned(to_f64_array2(obj)?))
    }

    fn dim(&self) -> (usize, usize) {
        use numpy::PyUntypedArrayMethods;
        match self {
            F64Rows::View(v) => (v.shape()[0], v.shape()[1]),
            F64Rows::Owned(a) => a.dim(),
        }
    }

    fn flat(&self) -> &[f64] {
        match self {
            F64Rows::View(v) => v.as_slice().expect("checked contiguous"),
            F64Rows::Owned(a) => a.as_slice().expect("converted arrays are row-major"),
        }
    }
}

/// Real-valued recombination framework: `pair(i, p1, p2, c1, c2, rng)` writes both offspring of
/// pair i into `c1` and `c2`, which start as copies of the parents (the second offspring is computed
/// but dropped when only N/2 offspring are returned); `global(r, j, a, pa, pb, rng)` is gene j of
/// global-mode offspring r, taken from random parents a and b (pa, pb their genes).
fn real_recombination<'py, F, G>(
    py: Python<'py>,
    name: &str,
    chrom: F64Rows<'py>,
    pc: f64,
    half_n: Opt,
    mut pair: F,
    mut global: G,
) -> PyResult<PyObject>
where
    F: FnMut(usize, &[f64], &[f64], &mut [f64], &mut [f64], &mut rand::rngs::SmallRng),
    G: FnMut(usize, usize, usize, f64, f64, &mut rand::rngs::SmallRng) -> f64,
{
    let (n, l) = chrom.dim();
    let flat = chrom.flat();
    let half = n / 2;
    let mut rng = crate::utils::rng();
    let out = match pairing(half_n, n, name)? {
        Pairing::Global(k) => {
            crate::utils::check_alloc(&[k, l], 8, "float64")?;
            let mut out = Array2::<f64>::zeros((k, l));
            for r in 0..k {
                for j in 0..l {
                    let (a, b) = random_pair(n, &mut rng);
                    out[[r, j]] = global(r, j, a, flat[a * l + j], flat[b * l + j], &mut rng);
                }
            }
            out
        }
        mode => {
            let full = matches!(mode, Pairing::Full);
            let mut out = if full {
                flat.to_vec()
            } else {
                flat[..half * l].to_vec()
            };
            let mut spare = vec![0.0; l];
            for i in 0..half {
                let (p1, p2) = (&flat[i * l..(i + 1) * l], &flat[(i + half) * l..][..l]);
                if rng.gen::<f64>() >= pc {
                    continue;
                }
                let (c1, c2) = if full {
                    let (lo, hi) = out.split_at_mut((i + half) * l);
                    (&mut lo[i * l..(i + 1) * l], &mut hi[..l])
                } else {
                    spare.copy_from_slice(p2);
                    (&mut out[i * l..(i + 1) * l], &mut spare[..])
                };
                pair(i, p1, p2, c1, c2, &mut rng);
            }
            let rows = if full { n } else { half };
            Array2::from_shape_vec((rows, l), out).expect("shape matches")
        }
    };
    chrom_out(py, out, false)
}

fn sbx_beta<R: Rng>(eta: f64, rng: &mut R) -> f64 {
    let u: f64 = rng.gen();
    let b = if u > 0.5 {
        1.0 / (2.0 * (1.0 - u))
    } else {
        2.0 * u
    };
    let b = b.powf(1.0 / (eta + 1.0));
    if rng.gen::<f64>() < 0.5 {
        -b
    } else {
        b
    }
}

/// Simulated binary crossover: each variable of a crossing pair is recombined with probability 0.5.
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=None, half_n=None, n=None, parallel=None))]
pub fn recsbx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    n: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let chrom = F64Rows::new(old_chrom)?;
    let eta = non_negative(n, chrom.dim().1, 20.0, "recsbx", "n")?;
    let eta2 = eta.clone();
    real_recombination(
        py,
        "recsbx",
        chrom,
        prob(xovr, 0.7, "recsbx")?,
        half_n,
        move |_, p1, p2, c1, c2, rng| {
            for j in 0..p1.len() {
                if rng.gen::<f64>() < 0.5 {
                    let b = sbx_beta(eta[j], rng);
                    c1[j] = 0.5 * ((1.0 + b) * p1[j] + (1.0 - b) * p2[j]);
                    c2[j] = 0.5 * ((1.0 - b) * p1[j] + (1.0 + b) * p2[j]);
                }
            }
        },
        move |_, j, _, a, b, rng| {
            if rng.gen::<f64>() < 0.5 {
                let beta = sbx_beta(eta2[j], rng);
                0.5 * ((1.0 + beta) * a + (1.0 - beta) * b)
            } else {
                a
            }
        },
    )
}

enum Alpha {
    Random,
    Scalar(f64),
    Matrix(Array2<f64>),
}

/// Alpha as geatpy 2.7.0 checks it: a number in [-0.25, 1.25), or a numpy array of such factors
/// shaped like the chromosome matrix (recint, one factor per gene) or with one factor per
/// individual (reclin, a 1-D array of length N). The range tests are `a < -0.25 or a > 1.25` for
/// a number and `a < -0.25 or a >= 1.25` for the elements of an array, so NaN passes; a bool array
/// is refused.
fn parse_alpha(alpha: Opt, name: &str, n: usize, l: usize, per_gene: bool) -> PyResult<Alpha> {
    use numpy::{PyArrayDescrMethods, PyUntypedArrayMethods};
    let bad = |v: f64| v < -0.25 || v >= 1.25;
    let err = |msg: &str| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("error in {0}: {1}", name, msg))
    };
    let shape_err = || {
        err(&format!(
            "Alpha illegal. (输入参数Alpha不合法，详见{}帮助文档。)",
            name
        ))
    };
    let Some(a) = alpha.filter(|a| !a.is_none()) else {
        return Ok(Alpha::Random);
    };
    let Ok(arr) = a.downcast::<numpy::PyUntypedArray>() else {
        let v: f64 = a.extract().map_err(|_| shape_err())?;
        if v < -0.25 || v > 1.25 {
            return Err(err(&format!(
                "The value of Alpha is illegal. (输入参数Alpha的值不合法，详见{}帮助文档。)",
                name
            )));
        }
        return Ok(Alpha::Scalar(v));
    };
    let shape_ok = if per_gene {
        arr.shape() == [n, l]
    } else {
        arr.shape() == [n]
    };
    if !shape_ok {
        return Err(shape_err());
    }
    let elements_err = || {
        err(&format!(
            "Some elements of Alpha are illegal. (输入参数Alpha的数据不合法，详见{}帮助文档。)",
            name
        ))
    };
    if arr.dtype().is_equiv_to(&numpy::dtype::<bool>(a.py())) {
        return Err(elements_err());
    }
    // a 1-D array becomes one column: row r holds the factor of individual r
    let m = to_f64_array2(a)?;
    if m.iter().any(|&v| bad(v)) {
        return Err(elements_err());
    }
    Ok(Alpha::Matrix(m))
}

fn intermediate<'py>(
    py: Python<'py>,
    name: &str,
    old: &Bound<'py, PyAny>,
    rec_opt: Opt,
    half_n: Opt,
    alpha: Opt,
    per_gene: bool,
) -> PyResult<PyObject> {
    // geatpy 2.7.0 checks RecOpt and Half_N before the chromosome and Alpha
    let pc = prob(rec_opt, 0.7, name)?;
    pairing(half_n, usize::MAX, name)?;
    let chrom = F64Rows::new(old)?;
    let (n, l) = chrom.dim();
    let alpha = parse_alpha(alpha, name, n, l, per_gene)?;
    let draw = |rng: &mut rand::rngs::SmallRng| rng.gen::<f64>() * 1.5 - 0.25;
    let half = n / 2;
    // c = p1 + a (p2 - p1) for every gene of offspring `row` (the row of the Alpha matrix it uses)
    let child = |row: usize,
                 p1: &[f64],
                 p2: &[f64],
                 c: &mut [f64],
                 rng: &mut rand::rngs::SmallRng,
                 shared: f64| {
        let genes = c.iter_mut().zip(p1.iter().zip(p2));
        match &alpha {
            Alpha::Scalar(a) => genes.for_each(|(c, (&x, &y))| *c = x + a * (y - x)),
            Alpha::Matrix(m) if per_gene => {
                for (j, (c, (&x, &y))) in genes.enumerate() {
                    *c = x + m[[row, j]] * (y - x);
                }
            }
            Alpha::Matrix(m) => {
                let a = m[[row, 0]];
                genes.for_each(|(c, (&x, &y))| *c = x + a * (y - x))
            }
            Alpha::Random if per_gene => {
                genes.for_each(|(c, (&x, &y))| *c = x + draw(rng) * (y - x))
            }
            Alpha::Random => genes.for_each(|(c, (&x, &y))| *c = x + shared * (y - x)),
        }
    };
    // Global mode, as geatpy 2.7.0: recint takes gene j's factor from the row of the individual
    // the gene starts from (its first random parent), reclin uses one factor per offspring.
    let mut row_factor = 0.0;
    real_recombination(
        py,
        name,
        chrom,
        pc,
        half_n,
        |i, p1, p2, c1, c2, rng| {
            let (s1, s2) = (draw(rng), draw(rng));
            child(i, p1, p2, c1, rng, s1);
            child(i + half, p1, p2, c2, rng, s2);
        },
        |r, j, from, a, b, rng| {
            // drawn for every gene whatever Alpha is, which keeps the random sequence
            let s = draw(rng);
            if j == 0 {
                row_factor = s;
            }
            let f = match &alpha {
                Alpha::Scalar(a) => *a,
                Alpha::Matrix(m) if per_gene => m[[from, j]],
                Alpha::Matrix(m) => m[[r, 0]],
                Alpha::Random if per_gene => draw(rng),
                Alpha::Random => row_factor,
            };
            a + f * (b - a)
        },
    )
}

/// Intermediate recombination: offspring = parent1 + a (parent2 - parent1), a in [-0.25, 1.25) per gene.
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=None, half_n=None, alpha=None, parallel=None))]
pub fn recint<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    alpha: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    intermediate(py, "recint", old_chrom, rec_opt, half_n, alpha, true)
}

/// Line recombination: like intermediate recombination with one factor a per offspring.
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=None, half_n=None, alpha=None, parallel=None))]
pub fn reclin<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    alpha: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    intermediate(py, "reclin", old_chrom, rec_opt, half_n, alpha, false)
}

fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    rng.sample(rand_distr::StandardNormal)
}

/// Normal distribution crossover. As in geatpy 2.7.0, a recombined gene is spread around the
/// parents' mean only when parent1 > parent2; otherwise both offspring take the mean.
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=None, half_n=None, a=None, parallel=None))]
pub fn recndx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    a: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let chrom = F64Rows::new(old_chrom)?;
    let aa = non_negative(a, chrom.dim().1, 1.4826, "recndx", "A")?;
    let aa2 = aa.clone();
    let gene = |aj: f64, x: f64, y: f64, rng: &mut rand::rngs::SmallRng| -> (f64, f64) {
        let mean = (x + y) * 0.5;
        let d = (x - y) * 0.5;
        let s = if rng.gen::<f64>() > 0.5 { -aj } else { aj };
        let g = standard_normal(rng);
        if d > 1e-15 {
            (mean + g * s * d, mean - g * s * d)
        } else {
            (mean, mean)
        }
    };
    real_recombination(
        py,
        "recndx",
        chrom,
        prob(xovr, 0.7, "recndx")?,
        half_n,
        move |_, p1, p2, c1, c2, rng| {
            for j in 0..p1.len() {
                if rng.gen::<f64>() >= 0.5 {
                    let (x, y) = gene(aa[j], p1[j], p2[j], rng);
                    c1[j] = x;
                    c2[j] = y;
                }
            }
        },
        move |_, j, _, x, y, rng| {
            if rng.gen::<f64>() >= 0.5 {
                gene(aa2[j], x, y, rng).0
            } else {
                x
            }
        },
    )
}

fn distinct_cuts<R: Rng>(l: usize, rng: &mut R) -> (usize, usize) {
    let a = rng.gen_range(0..l);
    let b = (a + 1 + rng.gen_range(0..l - 1)) % l;
    (a.min(b), a.max(b))
}

/// PMX on the positions where `take` is true, in geatpy 2.7.0's sequential form: at every crossing
/// position the two offspring exchange their *current* genes, and each offspring swaps the incoming
/// gene with the position that held it. Both offspring stay permutations.
/// Buffers reused across the pairs of one call.
#[derive(Default)]
struct PermScratch<T> {
    pos1: Vec<usize>,
    pos2: Vec<usize>,
    take: Vec<bool>,
    /// kept[slot] == tag marks the genes of the section kept by the current OX offspring
    kept: Vec<usize>,
    tag: usize,
    rest: Vec<T>,
    /// An index covering every row of the matrix (the rows of a permutation matrix share one value
    /// range), so it is not rebuilt for every pair.
    index: Option<DenseIndex>,
}

impl<T: Gene> PermScratch<T> {
    fn index(&self, a: &[T], b: &[T]) -> Option<DenseIndex> {
        self.index.or_else(|| DenseIndex::new(a, b))
    }
}

/// PMX on two offspring that start as copies of their parents: for every position k taken, each
/// offspring swaps its gene at k with the position holding the partner's gene at k.
fn pmx_in_place<T: Gene>(c1: &mut [T], c2: &mut [T], take: &[bool], s: &mut PermScratch<T>) {
    let Some(index) = s.index(c1, c2) else {
        for (k, &t) in take.iter().enumerate() {
            if !t {
                continue;
            }
            let (v1, v2) = (c1[k], c2[k]);
            if let Some(pos) = c1.iter().position(|&x| x == v2) {
                c1.swap(k, pos);
            }
            if let Some(pos) = c2.iter().position(|&x| x == v1) {
                c2.swap(k, pos);
            }
        }
        return;
    };
    // same swaps as above, with value -> position tables kept up to date
    const NONE: usize = usize::MAX;
    let (pos1, pos2) = (&mut s.pos1, &mut s.pos2);
    for pos in [&mut *pos1, &mut *pos2] {
        pos.clear();
        pos.resize(index.size, NONE);
    }
    for (k, (&a, &b)) in c1.iter().zip(c2.iter()).enumerate() {
        pos1[index.key(a)] = k;
        pos2[index.key(b)] = k;
    }
    let swap = |c: &mut [T], pos: &mut [usize], k: usize, v: T| {
        let p = pos[index.key(v)];
        if p != NONE {
            c.swap(k, p);
            pos[index.key(c[k])] = k;
            pos[index.key(c[p])] = p;
        }
    };
    for (k, &t) in take.iter().enumerate() {
        if t {
            let (v1, v2) = (c1[k], c2[k]);
            swap(c1, pos1, k, v2);
            swap(c2, pos2, k, v1);
        }
    }
}

/// Order crossover: each offspring keeps its own [lo, hi] section and fills the remaining positions
/// from position 0 onwards with the partner's genes in the partner's order.
fn ox_into<T: Gene>(
    p1: &[T],
    p2: &[T],
    lo: usize,
    hi: usize,
    c1: &mut [T],
    c2: &mut [T],
    s: &mut PermScratch<T>,
) {
    let index = s.index(p1, p2);
    for (own, other, c) in [(p1, p2, c1), (p2, p1, c2)] {
        let kept = &own[lo..=hi];
        // membership in the kept section: a table for small integer genes, a scan otherwise
        if let Some(ix) = &index {
            s.tag += 1;
            s.kept.resize(ix.size, 0);
            for &v in kept {
                s.kept[ix.key(v)] = s.tag;
            }
        }
        let (head, tail) = c.split_at_mut(lo);
        let (mid, tail) = tail.split_at_mut(hi + 1 - lo);
        match &index {
            Some(ix) => {
                // the partner's genes outside the kept section, compacted without branching
                s.rest.clear();
                s.rest.resize(other.len() + 1, T::default());
                let mut w = 0;
                for &v in other {
                    s.rest[w] = v;
                    w += usize::from(s.kept[ix.key(v)] != s.tag);
                }
                // check_permutation makes the genes of a row distinct, so at most kept.len() of the
                // partner's genes are dropped and enough remain
                assert!(
                    w >= lo + tail.len(),
                    "the partner lacks genes to fill the offspring"
                );
                head.copy_from_slice(&s.rest[..lo]);
                tail.copy_from_slice(&s.rest[lo..lo + tail.len()]);
            }
            None => {
                let mut rest = other.iter().copied().filter(|v| !kept.contains(v));
                for x in head.iter_mut().chain(tail.iter_mut()) {
                    *x = rest.next().unwrap();
                }
            }
        }
        mid.copy_from_slice(kept);
    }
}

/// The permutation crossovers that share the pairing logic below.
#[derive(Clone, Copy)]
enum PermOp {
    /// PMX; `uniform` is Method 2 (every position crosses with probability `pc`).
    Pmx {
        uniform: bool,
        pc: f64,
    },
    Ox,
}

impl PermOp {
    /// Both offspring of parents a and b into c1 and c2.
    #[allow(clippy::too_many_arguments)]
    fn cross<T: Gene, R: Rng>(
        self,
        a: &[T],
        b: &[T],
        global: bool,
        rng: &mut R,
        c1: &mut [T],
        c2: &mut [T],
        s: &mut PermScratch<T>,
    ) {
        let l = a.len();
        if l < 2 {
            c1.copy_from_slice(b);
            c2.copy_from_slice(a);
            return;
        }
        match self {
            PermOp::Pmx { uniform, pc } => {
                let mut take = std::mem::take(&mut s.take);
                take.clear();
                if uniform {
                    take.extend((0..l).map(|_| global || rng.gen::<f64>() < pc));
                } else {
                    let (lo, hi) = distinct_cuts(l, rng);
                    take.extend((0..l).map(|j| j >= lo && j <= hi));
                }
                c1.copy_from_slice(a);
                c2.copy_from_slice(b);
                pmx_in_place(c1, c2, &take, s);
                s.take = take;
            }
            PermOp::Ox => {
                let (lo, hi) = distinct_cuts(l, rng);
                ox_into(a, b, lo, hi, c1, c2, s);
            }
        }
    }
}

/// Pairs parents like the other crossovers. With `pair_gate` a pair crosses with probability `pc`;
/// without it the operator always runs and handles `pc` itself (global mode: Half_N an integer).
fn permutation_rows<T: Gene + Default>(
    chrom: &Array2<T>,
    op: PermOp,
    pc: f64,
    mode: Pairing,
    pair_gate: bool,
) -> Array2<T> {
    let (n, l) = chrom.dim();
    let half = n / 2;
    let mut rng = crate::utils::rng();
    let flat = chrom.as_slice().expect("converted arrays are row-major");
    let row = |i: usize| &flat[i * l..(i + 1) * l];
    // one index for all rows when the whole matrix spans a range small enough for one pair
    let index = DenseIndex::new(flat, &[]).filter(|ix| ix.size <= 8 * l + 16);
    let mut s = PermScratch {
        index,
        ..PermScratch::default()
    };
    let mut spare = vec![T::default(); l];
    match mode {
        Pairing::Global(k) => {
            let mut out = vec![T::default(); k * l];
            for c in out.chunks_exact_mut(l.max(1)).take(k) {
                let (a, b) = random_pair(n, &mut rng);
                op.cross(row(a), row(b), true, &mut rng, c, &mut spare, &mut s);
            }
            Array2::from_shape_vec((k, l), out).expect("shape matches")
        }
        mode => {
            let full = matches!(mode, Pairing::Full);
            let mut out = if full {
                flat.to_vec()
            } else {
                flat[..half * l].to_vec()
            };
            for i in 0..half {
                if pair_gate && rng.gen::<f64>() >= pc {
                    continue;
                }
                let (c1, c2) = if full {
                    let (lo, hi) = out.split_at_mut((i + half) * l);
                    (&mut lo[i * l..(i + 1) * l], &mut hi[..l])
                } else {
                    (&mut out[i * l..(i + 1) * l], &mut spare[..])
                };
                op.cross(row(i), row(i + half), false, &mut rng, c1, c2, &mut s);
            }
            let rows = if full { n } else { half };
            Array2::from_shape_vec((rows, l), out).expect("shape matches")
        }
    }
}

fn permutation_crossover<'py>(
    py: Python<'py>,
    name: &str,
    old: &Bound<'py, PyAny>,
    pc: f64,
    half_n: Opt,
    pair_gate: bool,
    op: PermOp,
) -> PyResult<PyObject> {
    let as_int = is_integer_like(old);
    if as_int {
        if let Some(chrom) = crate::utils::to_i32_array2(old) {
            check_permutation(&chrom, name)?;
            let mode = pairing(half_n, chrom.nrows(), name)?;
            let out = permutation_rows(&chrom, op, pc, mode, pair_gate);
            return Ok(numpy::IntoPyArray::into_pyarray(out, py)
                .into_any()
                .unbind());
        }
    }
    let chrom = to_f64_array2(old)?;
    // geatpy 2.7.0 reads the gene range with int(np.min(Chrom)) and int(np.max(Chrom)) first.
    if chrom.iter().any(|v| v.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cannot convert float NaN to integer",
        ));
    }
    if chrom.iter().any(|v| v.is_infinite()) {
        return Err(pyo3::exceptions::PyOverflowError::new_err(
            "cannot convert float infinity to integer",
        ));
    }
    check_permutation(&chrom, name)?;
    let mode = pairing(half_n, chrom.nrows(), name)?;
    chrom_out(
        py,
        permutation_rows(&chrom, op, pc, mode, pair_gate),
        as_int,
    )
}

/// Partially mapped crossover. Method 1: a pair crosses with probability XOVR on a two-point section;
/// Method 2 (uniform PMX): every position crosses with probability XOVR (all positions in global mode).
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=None, half_n=None, method=None, parallel=None))]
pub fn xovpmx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    method: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let method = crate::utils::opt_int(method, 1)?;
    if method != 1 && method != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in xovpmx: Method must be 1 or 2. (传入参数Method必须是1或者2。)",
        ));
    }
    let pc = prob(xovr, 0.7, "xovpmx")?;
    permutation_crossover(
        py,
        "xovpmx",
        old_chrom,
        pc,
        half_n,
        method != 2,
        PermOp::Pmx {
            uniform: method == 2,
            pc,
        },
    )
}

/// Order crossover (OX) for permutations.
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=None, half_n=None, parallel=None, params4=None))]
pub fn xovox<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params4: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params4);
    permutation_crossover(
        py,
        "xovox",
        old_chrom,
        prob(xovr, 0.7, "xovox")?,
        half_n,
        true,
        PermOp::Ox,
    )
}

/// Dispatcher kept for API compatibility: recombin(REC_F, OldChrom, ...) forwards the remaining
/// arguments to the named operator.
#[pyfunction]
#[pyo3(signature = (rec_f, *args, **kwargs))]
pub fn recombin<'py>(
    py: Python<'py>,
    rec_f: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<PyObject> {
    let name = rec_f.to_lowercase();
    let allowed = [
        "recsbx", "recdis", "recint", "reclin", "recndx", "xovsp", "xovdp", "xovud", "xovsh",
        "xovpmx", "xovox", "xovsec", "xovbd", "xovexp",
    ];
    if !allowed.contains(&name.as_str()) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in recombin: unknown operator {}.",
            rec_f
        )));
    }
    let module = py
        .import("_geatpy_core")
        .or_else(|_| py.import("geatpy._geatpy_core"))?;
    Ok(module.getattr(name.as_str())?.call(args, kwargs)?.unbind())
}
