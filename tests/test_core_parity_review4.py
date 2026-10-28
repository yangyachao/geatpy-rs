# -*- coding: utf-8 -*-
"""Regression tests for the 2026-10-20 review: argument checks, error types and messages, empty and
reversed inputs, donor drawing in mutde and the indicator formulas.

Expected values and messages come from the original geatpy 2.7.0 binaries (macOS arm64 wheel, see
scripts/parity/fuzz/README.md), so most of the file runs against both cores. Tests marked
`rust_only` check where the Rust core deliberately differs: it raises a clean exception where the
original crashes or reads past the end of an array (see AGENTS.md).
"""
import importlib.util
import warnings

import numpy as np
import pytest

import geatpy as ea

RUST_CORE = importlib.util.find_spec('_geatpy_core') is not None
rust_only = pytest.mark.skipif(not RUST_CORE, reason='the original crashes or reads out of bounds here')
I = ea.indicator
nan = np.nan


# --- array conversion -------------------------------------------------------------------------

@pytest.mark.parametrize('dtype', ['f8', 'f4', 'i8', 'i4'])
@pytest.mark.parametrize('op', [
    lambda x: ea.ndsortESS(x)[0],
    lambda x: ea.mergecv(x),
    lambda x: ea.cdist(x, np.arange(5.0)),
    lambda x: ea.crowdis(np.c_[x, -x], np.ones(5)),
])
@rust_only  # the original rejects 1-D input altogether
def test_reversed_1d_input_equals_its_copy(op, dtype):
    x = np.array([3, 1, 2, 0, 5], dtype)
    np.testing.assert_array_equal(op(x[::-1]), op(x[::-1].copy()))


@rust_only  # the original segfaults on a 1-D chromosome
def test_mutpolyn_accepts_a_reversed_1d_chromosome():
    x = np.array([3., 1., 2., 0., 5.])
    field = np.array([[0.] * 5, [6.] * 5, [0] * 5])
    np.random.seed(0)
    a = ea.mutpolyn('RI', x[::-1], field)
    np.random.seed(0)
    b = ea.mutpolyn('RI', x[::-1].copy(), field)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('op', [
    lambda o: ea.refselect(o, np.ones(4), 1, 2, np.ones((3, 2))),
    lambda o: ea.refgselect(o, np.ones((3, 2)), 2.0),
    pytest.param(lambda o: ea.ndsortESS(o), marks=rust_only),  # the original reads it as 2-D
    pytest.param(lambda o: ea.cdist(o, o), marks=rust_only),
])
def test_3d_objectives_raise_the_buffer_value_error(op):
    with pytest.raises(ValueError, match='Buffer has wrong number of dimensions'):
        op(np.ones((4, 2, 1)))


# --- mutde ------------------------------------------------------------------------------------

@pytest.mark.parametrize('n', [2, 3, 4])
def test_mutde_refills_the_donor_pool_without_replacement(n):
    # x = 32**k, F = [0, 1/16]: the trial x[r0] + (x[r3] - x[r4]) / 16 is a row value iff r3 == r4.
    # The pool of n - 1 donors runs out before r3, and r3 and r4 come from the same refill.
    x = (32.0 ** np.arange(n)).reshape(-1, 1)
    field = np.array([[-1e9], [1e9], [0]])
    np.random.seed(1)
    for _ in range(300):
        t = ea.mutde('RI', x, field, [None] * 5, [0.0, 1 / 16])[:, 0]
        assert not np.isin(t, x[:, 0]).any()


@rust_only  # the original crashes (division by zero)
def test_mutde_empty_population_needing_donors_raises():
    for mask in (2, np.ones((2, 2), bool)):
        with pytest.raises(RuntimeError, match='OldChrom must not be empty'):
            ea.mutde('RI', np.zeros((0, 2)), np.array([[0., 0], [1, 1], [0, 0]]), None, 0.5, 1, mask)


def test_mutde_empty_population_without_donors():
    out = ea.mutde('RI', np.zeros((0, 2)), np.array([[0., 0], [1, 1], [0, 0]]))
    assert out.shape == (0, 2)


_C = np.random.RandomState(0).rand(6, 3)
_FD = np.array([[0.] * 3, [1.] * 3, [0] * 3])


@pytest.mark.parametrize('kwargs, exc, message', [
    (dict(Mask_N=0), RuntimeError, 'Mask_N must be bigger than 0 when it is a integer'),
    (dict(Mask_N=False), RuntimeError, 'Mask_N must be bigger than 0 when it is a integer'),
    (dict(Mask_N=-1), RuntimeError, 'Mask_N must be bigger than 0 when it is a integer'),
    (dict(Mask_N=True), TypeError, 'an integer is required'),
    (dict(Mask_N=2.0), RuntimeError, 'Mask_N type error'),
    (dict(Mask_N=[[True] * 3] * 6), RuntimeError, 'Mask_N type error'),
    (dict(Mask_N=np.ones((6, 3))), RuntimeError, 'The element of Mask_N must be True or False'),
    (dict(Mask_N=np.ones((6, 3, 1), bool)), RuntimeError, 'The dimension of Mask_N must be 2'),
    (dict(XrList=(None, None, None)), RuntimeError, 'XrList must be a list'),
    (dict(XrList=[None] * 6), RuntimeError, 'The length of XrList must not be over than 5'),
    (dict(XrList=[[0, 1, 2, 3, 4, 5]]), RuntimeError, r'Data error of the parameter r0\(or Xr0\)'),
    (dict(XrList=[None, (0, 1, 2, 3, 4, 5)]), RuntimeError, r'Data error of the parameter r1\(or Xr1\)'),
    (dict(XrList=[np.arange(7) % 6]), RuntimeError, 'The length of r0 must be equal to others'),
    (dict(XrList=[np.arange(5)]), RuntimeError, 'The length of r0 must be equal to others'),
    (dict(XrList=[None, np.arange(7) % 6]), RuntimeError, 'r1 and OldChrom disagree'),
    (dict(XrList=[None, None, np.arange(5)]), RuntimeError, 'The length of r2 must be equal to others'),
    (dict(XrList=[np.arange(6)], Mask_N=3), RuntimeError, 'The length of r0 must be equal to others'),
    (dict(XrList=[np.r_[np.arange(5), 6]]), RuntimeError, 'Some elements of r0 is illegal'),
    (dict(XrList=[None, None, np.r_[np.arange(5), -1]]), RuntimeError, 'Some elements of r2 is illegal'),
])
def test_mutde_argument_checks(kwargs, exc, message):
    with pytest.raises(exc, match=message):
        ea.mutde('RI', _C, _FD, **kwargs)


def test_mutde_accepts_what_the_original_accepts():
    assert ea.mutde('RI', _C, _FD, [np.arange(3)], Mask_N=3).shape == (3, 3)
    assert ea.mutde('RI', _C, _FD, [np.zeros((8, 3))]).shape == (6, 3)
    assert ea.mutde('RI', _C, _FD, [np.arange(6) + 0.5]).shape == (6, 3)
    assert ea.mutde('RI', _C, _FD, Mask_N=np.ones((6, 3), bool)).shape == (18,)


# --- refselect --------------------------------------------------------------------------------

_RS = np.random.RandomState(13)
_O = _RS.rand(40, 3)
_Z = ea.crtup(3, 12)[0]
_LVQ = np.r_[np.ones(10), 2 * np.ones(3), 3 * np.ones(27)]  # 10 before the critical level, 3 on it

_SHORT = 'The number of individuals which are in and below the critical level is less than the required number'


@pytest.mark.parametrize('levels, cri, num, ref', [
    (_LVQ, 2, 20, _Z),
    (_LVQ, 0, 5, _Z),
    (_LVQ, -1, 5, _Z),
    (np.full(40, np.inf), 2, 5, _Z),
    (np.empty(0), 2, 5, _Z),
    (_LVQ, 2, 20, ea.crtup(2, 12)[0]),  # raised before the width check
])
def test_refselect_too_few_individuals_up_to_the_critical_level(levels, cri, num, ref):
    with pytest.raises(RuntimeError, match=_SHORT):
        ea.refselect(_O, levels, cri, num, ref)


def test_refselect_converts_crilevel_and_num_like_c_ints():
    np.random.seed(0)
    base = ea.refselect(_O, _LVQ, 2, 12, _Z)
    for cri in (2.9, np.float64(2.5), np.int64(2)):
        assert ea.refselect(_O, _LVQ, cri, 12, _Z).sum() == base.sum()
    assert ea.refselect(_O, _LVQ, 1.5, 5, _Z).sum() == 5  # criLevel 1, not every level below 1.5
    assert ea.refselect(_O, _LVQ, 2, 12.7, _Z).sum() == 12
    assert ea.refselect(_O, _LVQ, 2, np.float64(12.0), _Z).sum() == 12
    with pytest.raises(OverflowError, match='value too large to convert to int'):
        ea.refselect(_O, _LVQ, 2 ** 31, 5, _Z)
    with pytest.raises(TypeError, match='an integer is required'):
        ea.refselect(_O, _LVQ, '2', 5, _Z)
    with pytest.raises(ValueError, match='cannot convert float NaN to integer'):
        ea.refselect(_O, _LVQ, 2, nan, _Z)
    r = ea.refselect(_O, _LVQ, 2, 2 ** 64, _Z)  # everyone (the original returns np.arange)
    assert r.all() if r.dtype == bool else len(r) == 40


@pytest.mark.parametrize('levels', [np.ones(15), np.ones(41), np.ones(1)])
def test_refselect_levels_of_another_length_raise_index_error(levels):
    with pytest.raises(IndexError, match='boolean index did not match'):
        ea.refselect(_O, levels, 1, 5, _Z)


@pytest.mark.parametrize('cri, num', [(2, 10), (2, 8), (3, 10)])
def test_refselect_width_check_also_on_the_early_return(cri, num):
    lv = np.r_[np.ones(10), 3 * np.ones(30)]  # no individual on level 2
    with pytest.raises(ValueError, match='size 2 is different from 3'):
        ea.refselect(_O, lv, cri, num, ea.crtup(2, 12)[0])


@rust_only  # the original picks individuals without any reference point to go by
def test_reference_selection_without_reference_points_raises():
    with pytest.raises(ValueError, match='refPoint must contain at least one reference point'):
        ea.refselect(_O, _LVQ, 3, 20, np.zeros((0, 3)))
    with pytest.raises(ValueError, match='refPoint must contain at least one reference point'):
        ea.refgselect(_O, np.zeros((0, 3)), 2.0)


# --- refgselect -------------------------------------------------------------------------------

@rust_only  # without CV the original raises numpy's TypeError for ~ on an empty object array
def test_refgselect_without_individuals_returns_gamma():
    flag, gamma = ea.refgselect(np.zeros((0, 2)), ea.crtup(2, 4)[0], 2.0)
    assert flag.shape == (0,) and gamma.shape == (4,)


def test_refgselect_without_individuals_with_cv_returns_gamma():
    flag, gamma = ea.refgselect(np.zeros((0, 2)), ea.crtup(2, 4)[0], 2.0, np.zeros((0, 1)))
    assert flag.shape == (0,) and gamma.shape == (4,)


def test_refgselect_gamma_must_be_an_array():
    with pytest.raises(TypeError, match='Cannot convert float to numpy.ndarray'):
        ea.refgselect(_O, _Z, 2.0, None, 0.5)
    with pytest.raises(ValueError, match=r'expected 1, got 0'):
        ea.refgselect(_O, _Z, 2.0, None, np.array(0.5))


@rust_only  # the original reads past the end of a short Gamma
def test_refgselect_short_gamma_raises():
    with pytest.raises(ValueError, match='Gamma must have one element per reference point'):
        ea.refgselect(_O, _Z, 2.0, None, np.ones(len(_Z) - 1))


@rust_only  # the original ignores extra CV rows and reads past the end of a short CV
def test_refgselect_cv_rows_are_checked_after_the_width():
    with pytest.raises(RuntimeError, match=r'error in refgselect: CV and ObjV disagree\. \('):
        ea.refgselect(_O, _Z, 2.0, np.zeros((39, 1)))
    with pytest.raises(ValueError, match='size 2 is different from 3'):
        ea.refgselect(_O, ea.crtup(2, 12)[0], 2.0, np.zeros((39, 1)))


def test_refgselect_warning_points_at_the_caller():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ea.refgselect(_O, np.array([[1., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), 2.0)
    assert [w.filename for w in caught] == [__file__]


# --- ndsort arguments and CV messages ---------------------------------------------------------

_ND = np.random.RandomState(3).rand(10, 2)


@pytest.mark.parametrize('ndsort', [ea.ndsortESS, ea.ndsortTNS, ea.ndsortDED])
def test_ndsort_need_num_is_converted_like_a_c_int(ndsort):
    two = ndsort(_ND, 2)
    for v in (2.0, 2.7, np.float64(2.7), np.int32(2)):
        np.testing.assert_array_equal(ndsort(_ND, v)[0], two[0])
    np.testing.assert_array_equal(ndsort(_ND, -1)[0], ndsort(_ND, 0)[0])
    with pytest.raises(TypeError):
        ndsort(_ND, '2')
    with pytest.raises(ValueError, match='cannot convert float NaN to integer'):
        ndsort(_ND, nan)


@pytest.mark.parametrize('ndsort', [ea.ndsortESS, ea.ndsortTNS, ea.ndsortDED])
def test_ndsort_rejects_extra_arguments(ndsort):
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        ndsort(_ND, foo=1)
    with pytest.raises(TypeError):
        ndsort(_ND, None, None, None, None, 1)


@pytest.mark.parametrize('name, call', [
    ('ndsortESS', lambda cv: ea.ndsortESS(_ND, None, None, cv)),
    ('ndsortDED', lambda cv: ea.ndsortDED(_ND, None, None, cv)),
    ('ranking', lambda cv: ea.ranking(_ND[:, :1], cv)),
    ('tcheby', lambda cv: ea.tcheby(_ND, np.ones((10, 2)), None, cv)),
])
def test_cv_row_mismatch_uses_the_original_message(name, call):
    with pytest.raises(RuntimeError, match=r'error in %s: CV and ObjV disagree\. \(' % name):
        call(np.zeros((12, 1)))


# --- indicators -------------------------------------------------------------------------------

_P = np.random.RandomState(2).rand(6, 2)
_R = np.random.RandomState(2).rand(11, 2)[6:]


def test_gd_and_igd_round_like_the_original():
    rs = np.random.RandomState(2)
    p, r = rs.rand(6, 2), rs.rand(5, 2)
    assert I.GD(p, r) == 0.09636125991567641
    assert I.IGD(p, r) == 0.20465262235826706


def test_indicators_take_the_original_argument_names():
    assert I.GD(ObjV=_P, PF=_R) == I.GD(_P, _R)
    assert I.GD(_P, _R, None, True) == I.GD(_P, _R)
    assert I.IGD(ObjV=_P, PF=_R, Parallel=False) == I.IGD(_P, _R)
    assert I.Spacing(_P, None, True) == I.Spacing(_P)
    assert I.HV(ObjV=_P, PF=_R) == I.HV(_P, _R)
    with pytest.raises(TypeError):
        I.IGD(_P, _R, None, True)
    for f, args in ((I.GD, (_P, _R, None)), (I.IGD, (_P, _R)), (I.Spacing, (_P, None)), (I.HV, (_P, _R))):
        with pytest.raises(RuntimeError, match='Parallel must be None, True or False'):
            f(*args, 'yes')


def test_spacing_like_the_original():
    assert isinstance(I.Spacing(_P), np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert np.isnan(I.Spacing(_P[:1]))
    with pytest.raises(ValueError, match='zero-size array'):
        I.Spacing(_P[:0])


def test_indicators_on_empty_and_nan_input():
    with pytest.raises(ZeroDivisionError):
        I.GD(_P[:0], _R)
    assert np.isnan(I.IGD(_P, _R[:0]))
    o = _P.copy()
    o[0] = nan
    assert np.isnan(I.GD(o, _R))
    pf = _R.copy()
    pf[0] = nan
    assert np.isnan(I.IGD(_P, pf))
    pf[:, 0] = nan
    assert np.isnan(I.HV(_P, pf))
    with pytest.raises(RuntimeError, match='PF must be a matrix'):
        I.HV(np.zeros((3, 0)))
    with pytest.raises(RuntimeError, match='PF must be a matrix'):
        I.HV(_P, _R[:, :, None])


@rust_only  # the original reads uninitialised memory for an empty PF
def test_hv_of_an_empty_pf_is_zero():
    assert I.HV(_P, _R[:0]) == 0.0


def test_empty_sums_are_positive_zero():
    out = ea.mergecv(np.array([[-1.], [0.], [3.]]))
    assert not np.signbit(out[:2]).any()
    assert not np.signbit(ea.cdist(np.zeros((2, 0)), np.zeros((3, 0)))).any()
    assert not np.signbit(I.GD(np.zeros((3, 0)), np.zeros((2, 0))))


# --- sizes from user input --------------------------------------------------------------------

@pytest.mark.parametrize('call', [
    lambda: ea.tour(np.random.rand(5, 1), 2 ** 61),
    lambda: ea.rws(np.random.rand(5, 1), 2 ** 61),
    lambda: ea.crtbp(2 ** 62, 4),
    lambda: ea.crtip(2 ** 62, np.array([[0., 0], [5, 5], [1, 1]])),
    lambda: ea.cdist(np.zeros((2 ** 40, 0)), np.zeros((2 ** 40, 0))),
    lambda: ea.ndsortESS(np.broadcast_to(np.zeros((1, 2)), (2 ** 59, 2))),
])
def test_impossible_sizes_raise_numpy_value_error(call):
    with pytest.raises(ValueError, match='array is too big'):
        call()


# Sizes of 8 PiB and more: beyond any process address space (48-bit, or 52-bit on some arm64
# kernels), so the allocator refuses them on every platform. Smaller ones (tens of TiB) may be
# reserved under memory overcommit (macOS, Linux with vm.overcommit_memory=1) and the process is
# killed once the memory is written.
@rust_only  # the original's messages differ slightly; both raise MemoryError
@pytest.mark.parametrize('call', [
    lambda: ea.crtbp(2 ** 49, 3),
    lambda: ea.crtri(2 ** 49, np.array([[0., 0], [1, 1], [0, 0]])),
    lambda: ea.urs(5, 2 ** 50),
    lambda: ea.mutde('RI', np.random.rand(4, 2), np.array([[0., 0], [1, 1], [0, 0]]), None, 0.5, 1, 2 ** 49),
])
def test_failed_allocations_raise_memory_error(call):
    with pytest.raises(MemoryError, match='Unable to allocate'):
        call()


@rust_only  # the original only refills populations that got no individual at all
@pytest.mark.parametrize('sel', ['dup', 'tour', 'rws', 'sus', 'urs', 'rcs'])
@pytest.mark.parametrize('msel', [0.5, 0.3, 8])
def test_mselecting_gives_every_population_its_minimum(sel, msel):
    # geatpy-dev/geatpy#330: a small population with poor fitness kept a single individual
    np.random.seed(3)
    sizes = [100, 124, 16]
    fits = [np.random.rand(100, 1) + 1, np.random.rand(124, 1) + 1, np.random.rand(16, 1) * 0.01]
    mins = [int(np.ceil(s * msel)) if msel < 1 else min(int(np.ceil(msel)), s) for s in sizes]
    for nsel in (150, 300):
        chosen = ea.mselecting(sel, fits, nsel, msel)
        assert [len(c) >= m for c, m in zip(chosen, mins)] == [True] * 3
        assert all(0 <= i < s for c, s in zip(chosen, sizes) for i in c)


# --- recint / reclin Alpha, Half_N (2026-10-28) -----------------------------------------------

_X = np.random.RandomState(7).rand(6, 3)


@pytest.mark.parametrize('fn, alpha', [
    (ea.recint, np.full((6, 2), 0.3)), (ea.recint, np.full((3, 3), 0.3)), (ea.recint, np.full(3, 0.3)),
    (ea.recint, np.zeros((0, 3))), (ea.recint, np.full((6, 3, 1), 0.3)), (ea.recint, [0.3] * 3),
    (ea.recint, 'x'), (ea.reclin, np.full(3, 0.3)), (ea.reclin, np.full((6, 1), 0.3)),
    (ea.reclin, np.full((1, 6), 0.3)), (ea.reclin, np.full((6, 3), 0.3)), (ea.reclin, np.zeros(0)),
    (ea.reclin, [0.3] * 6),
])
def test_alpha_must_have_the_documented_shape(fn, alpha):
    # recint: an N x L matrix; reclin: a 1-D array of length N (a bad shape is reported before
    # bad values)
    with pytest.raises(RuntimeError, match=r'error in %s: Alpha illegal\.' % fn.__name__):
        fn(_X, 1, False, alpha)


@pytest.mark.parametrize('fn, alpha', [
    (ea.recint, np.full((6, 3), 1.25)), (ea.recint, np.full((6, 3), np.inf)),
    (ea.reclin, np.full(6, -0.3)), (ea.reclin, np.full(6, 1.25)),
])
def test_alpha_elements_out_of_range(fn, alpha):
    with pytest.raises(RuntimeError, match='Some elements of Alpha are illegal'):
        fn(_X, 1, False, alpha)


@pytest.mark.parametrize('fn', [ea.recint, ea.reclin])
def test_alpha_scalar_range_is_closed(fn):
    # a number may be 1.25 (an array element may not)
    for v in (-0.25, 1.25):
        assert fn(_X, 1, False, v).shape == (6, 3)
    for v in (-0.2500001, 1.2500001):
        with pytest.raises(RuntimeError, match='The value of Alpha is illegal'):
            fn(_X, 1, False, v)


@pytest.mark.parametrize('fn, alpha', [
    (ea.recint, nan), (ea.reclin, nan), (ea.recint, np.full((6, 3), nan)), (ea.reclin, np.full(6, nan)),
    (ea.reclin, np.full(6, -0.25)),
])
def test_alpha_range_check_lets_nan_through(fn, alpha):
    # geatpy 2.7.0 tests a < -0.25 or a >= 1.25
    assert fn(_X, 1, False, alpha).shape == (6, 3)


@pytest.mark.parametrize('fn', [ea.recint, ea.reclin])
def test_recopt_and_half_n_are_checked_before_alpha(fn):
    with pytest.raises(RuntimeError, match='RecOpt must be in'):
        fn(_X, 5, False, np.full((6, 2), 9.0))
    with pytest.raises(RuntimeError, match='Half_N error'):
        fn(_X, 1, 'x', np.full((6, 2), 9.0))


@pytest.mark.parametrize('fn, chrom', [
    (ea.recint, _X), (ea.reclin, _X), (ea.recsbx, _X), (ea.xovdp, _X), (ea.recdis, _X),
])
@pytest.mark.parametrize('half_n', ['x', [1], 2.0, -1])
def test_half_n_that_is_not_a_positive_int(fn, chrom, half_n):
    with pytest.raises(RuntimeError, match='Half_N error'):
        fn(chrom, 0.5, half_n)


def _global_rows(fn, alpha, seeds=10):
    # X[r] = 3^r makes (parent a, parent b, Alpha row) recoverable from every offspring gene
    n, l = 6, 4
    x = np.array([[3.0 ** r + 0.1 * j for j in range(l)] for r in range(n)])
    used = set()
    for seed in range(seeds):
        np.random.seed(seed)
        out = fn(x, 1, n, alpha)
        for i in range(out.shape[0]):
            for j in range(l):
                hits = {(a, r) if alpha.ndim == 2 else (i, r)
                        for a in range(n) for b in range(n) for r in range(n)
                        if a != b and abs(x[a, j] + (alpha[r, j] if alpha.ndim == 2 else alpha[r])
                                          * (x[b, j] - x[a, j]) - out[i, j]) < 1e-9}
                used.add(any(k == r for k, r in hits))
    return used


def test_global_recint_takes_alpha_from_the_first_parent():
    alpha = np.array([[0.05 + 0.1 * r + 0.013 * j for j in range(4)] for r in range(6)])
    assert _global_rows(ea.recint, alpha) == {True}


def test_global_reclin_uses_one_alpha_per_offspring():
    alpha = np.array([0.05, 0.17, 0.29, 0.41, 0.53, 0.65])
    assert _global_rows(ea.reclin, alpha) == {True}
    x = np.vstack([np.zeros((3, 5)), np.ones((3, 5))])
    for seed in range(5):
        np.random.seed(seed)
        for row in ea.reclin(x, 1, 6):
            mixed = {round(min(v, 1 - v), 9) for v in row if v not in (0.0, 1.0)}
            assert len(mixed) <= 1


@rust_only  # the original reads a bool, integer or float32 Alpha buffer as float64 (garbage)
def test_alpha_arrays_of_other_dtypes():
    with pytest.raises(RuntimeError, match='Some elements of Alpha are illegal'):
        ea.recint(_X, 1, False, np.zeros((6, 3), bool))
    for dtype in (np.int64, np.int32, np.float32):
        np.random.seed(0)
        got = ea.recint(_X, 1, False, np.ones((6, 3), dtype))
        np.random.seed(0)
        np.testing.assert_array_equal(got, ea.recint(_X, 1, False, np.ones((6, 3))))
    np.random.seed(0)
    got = ea.reclin(_X, 1, False, np.full(6, 0.5, np.float32))
    np.random.seed(0)
    np.testing.assert_array_equal(got, ea.reclin(_X, 1, False, np.full(6, 0.5)))


@rust_only  # 2^0 - 1 = 0: the x86_64 original traps on the integer division; arm64 returns lb
def test_bs2int_zero_length_code_gives_the_lower_bound():
    fd = np.array([[0, 3], [2, 0], [9, 7], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1]], float)
    b = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int32)
    np.testing.assert_array_equal(ea.bs2int(b, fd), [[2, 5], [2, 3]])


# --- full review follow-ups (2026-10-28) -------------------------------------------------------

def _fdr(lb, ub, vt=0.0):
    return np.array([[lb], [ub], [vt]], float)


@pytest.mark.parametrize('x, want', [(-3.0, 5.0), (0.0, 5.0), (3.0, 5.0), (10.0, 1.0)])
def test_truncation_with_reversed_bounds_checks_the_lower_one_first(x, want):
    assert ea.boundfix('RI', np.array([[x]]), _fdr(5.0, 1.0), 1)[0, 0] == want


@pytest.mark.parametrize('lb, ub, x, want', [
    (nan, 5.0, -3.0, -3.0), (nan, 5.0, 10.0, 5.0), (0.0, nan, -3.0, 0.0), (0.0, nan, 10.0, 10.0),
])
def test_truncation_never_crosses_a_nan_bound(lb, ub, x, want):
    assert ea.boundfix('RI', np.array([[x]]), _fdr(lb, ub), 1)[0, 0] == want


@pytest.mark.parametrize('op', [
    lambda x, f: ea.mutpolyn('RI', x, f, 1), lambda x, f: ea.mutgau('RI', x, f, 1),
    lambda x, f: ea.mutuni('RI', x, f, 1), lambda x, f: ea.mutbga('RI', x, f, 1),
    lambda x, f: ea.mutde('RI', x, f, None, 0.5),
])
def test_mutations_accept_reversed_bounds(op):
    x = np.random.RandomState(0).rand(4, 3)
    np.random.seed(0)
    assert op(x, np.array([[1.0] * 3, [0.0] * 3, [0] * 3])).shape == (4, 3)


def test_random_reals_over_an_unbounded_range():
    np.random.seed(0)
    assert np.all(ea.crtrp(2, np.array([[-1e308], [1e308], [0.0]])) == np.inf)
    assert np.all(ea.crtri(2, _fdr(0.0, np.inf)) == np.inf)
    assert np.all(np.isnan(ea.crtri(2, _fdr(0.0, nan))))


@pytest.mark.parametrize('name', ['crtrp', 'crtri', 'crtip', 'crtpp'])
@pytest.mark.parametrize('rows', [2, 4])
def test_creation_wants_a_three_row_field_dr(name, rows):
    f = np.vstack([np.zeros(2), np.full(2, 5.0), np.ones(2), np.ones(2)])[:rows]
    with pytest.raises(RuntimeError, match='error in %s: FieldDR must be a matrix with 3 rows' % name):
        getattr(ea, name)(3, f)


@pytest.mark.parametrize('op', [ea.tcheby, ea.pbi])
@pytest.mark.parametrize('w', [np.ones((4, 2)), np.ones((4, 4)), np.ones((3, 3))])
def test_decomposition_weights_must_match_objv(op, w):
    with pytest.raises(RuntimeError, match='error in %s: weights and ObjV disagree' % op.__name__):
        op(np.ones((4, 3)), w)


_POPS = [np.random.RandomState(1).rand(3, 1), np.random.RandomState(2).rand(3, 1)]


@pytest.mark.parametrize('fitn_vs', [[_POPS[0]], [np.zeros((3, 0))] * 2, 'not used'])
def test_migrate_ignores_fitn_vs_unless_fitness_is_used(fitn_vs):
    np.random.seed(0)
    assert len(ea.migrate([3, 3], 0.5, 0, 0, 0, fitn_vs)) == 3


@pytest.mark.parametrize('select, replacement', [(1, 0), (0, 2)])
def test_migrate_needs_fitn_vs_with_fitness(select, replacement):
    with pytest.raises(RuntimeError, match='FitnVs must not be None when Select=1 or Replacement=2'):
        ea.migrate([3, 3], 0.5, 0, select, replacement, None)
    with pytest.raises(IndexError, match='index 0 is out of bounds for axis 0 with size 0'):
        ea.migrate([3, 3], 0.5, 0, select, replacement, [np.zeros((3, 0))] * 2)


@rust_only  # the original reads past the end of a missing or short FitnV
@pytest.mark.parametrize('fitn_vs', [[_POPS[0]], [_POPS[0], _POPS[1][:2]]])
def test_migrate_refuses_missing_or_short_fitn_vs(fitn_vs):
    with pytest.raises(RuntimeError, match='FitnVs and PopSizes disagree'):
        ea.migrate([3, 3], 0.5, 0, 1, 0, fitn_vs)


@rust_only  # the original returns uninitialised indices (often zeros)
def test_roulette_selection_of_an_empty_population_gives_zeros():
    for op in (ea.rws, ea.sus):
        np.random.seed(0)
        np.testing.assert_array_equal(op(np.zeros((0, 1)), 3, None), [0, 0, 0])


def test_selection_on_an_empty_population():
    empty = np.zeros((0, 1))
    with pytest.raises(ValueError, match='high <= 0'):
        ea.urs(empty, 3)
    with pytest.raises(ValueError, match='need at least one array to concatenate'):
        ea.mselecting('rws', [], 2)


@rust_only  # the original crashes
def test_tournament_and_truncation_of_an_empty_population():
    assert len(ea.tour(np.zeros((0, 1)), 3)) == 0
    assert len(ea.dup(np.zeros((0, 1)), 3)) == 0


@pytest.mark.parametrize('field, msg, exc', [
    (np.array([[1, 1, 1], [3, 3, 3]]), 'Chromosome error', RuntimeError),
    (np.array([[3, 3, 3], [0, 0, 0]]), r'Lind must be or smaller than Ub - Lb \+ 1', RuntimeError),
    (np.array([[0, 0, 0]]), 'index 1 is out of bounds for axis 0 with size 1', IndexError),
])
def test_mutpp_checks_field_dr(field, msg, exc):
    with pytest.raises(exc, match=msg):
        ea.mutpp('P', np.array([[0, 1, 2], [2, 0, 1]]), field, 1)


def test_crtfld_checks_borders():
    vt, ranges = np.zeros(2), np.array([[0, 0], [1, 1]])
    with pytest.raises(IndexError, match='index 1 is out of bounds for axis 0 with size 1'):
        ea.crtfld('BG', vt, ranges, np.array([[1, 1]]), [2, 2])
    with pytest.raises(RuntimeError, match='VarTypes and borders disagree'):
        ea.crtfld('BG', vt, ranges, np.array([[1], [1]]), [2, 2])


@pytest.mark.parametrize('op', [ea.xovsp, ea.xovdp, ea.xovud, ea.xovexp, ea.xovsec])
def test_crossover_of_a_list_with_a_shorter_matrix(op):
    np.random.seed(0)
    with pytest.raises(IndexError, match='is out of bounds for axis 0 with size 8'):
        op([np.zeros((10, 3)), np.zeros((8, 2))], 1.0)


@pytest.mark.parametrize('rm', [2, -1, 0.5, 'x', [1]])
def test_ranking_rm_must_be_0_or_1(rm):
    with pytest.raises(RuntimeError, match='PM must be in 0 or 1'):
        ea.ranking(np.arange(10.0)[:, None], None, None, rm, 1.5)


@pytest.mark.parametrize('rm', [1.0, True])
def test_ranking_rm_is_compared_as_a_number(rm):
    assert ea.ranking(np.arange(10.0)[:, None], None, None, rm, 1.5).shape == (10, 1)


@rust_only  # x^(N-1) overflows in the original, which returns NaN
@pytest.mark.parametrize('n, sp', [(500, 400), (1100, 1000)])
def test_nonlinear_ranking_of_a_large_population(n, sp):
    f = ea.ranking(np.arange(float(n))[:, None], None, None, 1, sp)
    assert np.isclose(f.max(), sp) and np.isclose(f.sum(), n)


def test_rps_and_selecting_check_nsel():
    fit = np.random.RandomState(0).rand(3, 1)
    for nsel in (0, 5):
        with pytest.raises(RuntimeError, match='error in rps: Nsel must be an integer bigger than 0'):
            ea.rps(fit, nsel)
    with pytest.raises(RuntimeError, match='error in rps: Nsel must be an integer bigger than 0'):
        ea.selecting('rps', fit, 5)
    for sel in ('rws', 'tour', 'otos', 'rps'):
        for nsel in (0, -1):
            with pytest.raises(RuntimeError, match='error in selecting: NSel must be bigger than 0'):
                ea.selecting(sel, fit, nsel)


@rust_only  # the original raises IndexError on a 1-D chromosome
def test_mutbin_takes_a_1d_chromosome_as_one_individual():
    chrom = (np.random.RandomState(3).rand(40) < 0.5).astype(np.int32)
    np.random.seed(0)
    one = ea.mutbin('BG', chrom)
    np.random.seed(0)
    row = ea.mutbin('BG', chrom[None, :])  # the same draws as a 1 x L matrix (default Pm = 1/L)
    assert one.shape == (40,)
    np.testing.assert_array_equal(one, row.ravel())


@pytest.mark.parametrize('lens, exc, msg', [
    ([0, 0], RuntimeError, 'error in crtbp: Lind must be a int number > 0'),
    ([-3, -2], RuntimeError, 'error in crtbp: Lind must be a int number > 0'),
    ([nan, 2], ValueError, 'cannot convert float NaN to integer'),
])
def test_crtpc_bg_checks_the_code_length(lens, exc, msg):
    fd = np.array([lens, [0, 0], [1, 1], [0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], float)
    with pytest.raises(exc, match=msg):
        ea.crtpc('BG', 3, fd)


@pytest.mark.parametrize('dim, num', [(64, 10 ** 6), (2, 3)])
def test_crtgp_needs_two_points_per_axis(dim, num):
    with pytest.raises(RuntimeError, match='The NUM is too small to build a grid'):
        ea.crtgp(dim, num)
