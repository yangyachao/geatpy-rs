# -*- coding: utf-8 -*-
"""
Compare the Rust core against recorded behaviour of the official geatpy 2.7.0 core.

Fixture: tests/fixtures/core_parity_orig_2_7_0.json (regenerate with scripts/parity/run_original_core.sh).
Template runs are slow and only execute when GEATPY_PARITY_TEMPLATES=1.
"""
import json
import os

import numpy as np
import pytest

import geatpy as ea
from tests import parity_cases as pc

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'core_parity_orig_2_7_0.json')
with open(FIXTURE) as _fh:
    GOLDEN = json.load(_fh)

DET = pc.deterministic_cases(ea)
RND = pc.random_stats(ea)


def _assert_same(expected, actual, path='value'):
    if isinstance(expected, list):
        assert isinstance(actual, list), '%s: expected a sequence of %d items' % (path, len(expected))
        assert len(actual) == len(expected), '%s: expected %d items, got %d' % (path, len(expected), len(actual))
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_same(e, a, '%s[%d]' % (path, i))
        return
    if isinstance(expected, str):
        assert expected == actual, path
        return
    assert not isinstance(actual, (list, str)), '%s: expected an array' % path
    assert actual.size == expected.size, '%s: shape %s != %s' % (path, actual.shape, expected.shape)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), rtol=1e-6, atol=1e-8, err_msg=path)


@pytest.mark.parametrize('name', sorted(GOLDEN['deterministic']))
def test_deterministic_parity(name):
    golden = GOLDEN['deterministic'][name]
    if pc.known_defect(name):
        pytest.skip('original core is wrong here: ' + pc.known_defect(name))
    if golden['status'] != 'ok':
        pytest.skip('original core rejects this input: ' + golden['error'])
    got = pc.run_case(DET[name])
    assert got['status'] == 'ok', got.get('error')
    expected, actual = pc.decode(golden['value']), pc.decode(got['value'])
    if name in pc.ORDER_FREE:
        expected, actual = np.sort(expected), np.sort(actual)
    _assert_same(expected, actual)


@pytest.mark.parametrize('name', sorted(GOLDEN['random']))
def test_random_operator_statistics(name):
    golden = GOLDEN['random'][name]
    if golden['status'] != 'ok':
        pytest.skip('original core rejects this input: ' + golden['error'])
    f, tol = RND[name]
    got = pc.run_stat(f, 5)
    assert got['status'] == 'ok', got.get('error')
    expected = float(pc.decode(golden['value']))
    actual = float(pc.decode(got['value']))
    assert abs(actual - expected) <= tol, 'expected %.4f +- %.4f, got %.4f' % (expected, tol, actual)


def test_ndsortDED_matches_ESS():
    for args in [(pc.X2,), (pc.X2, 30), (pc.X3, 30), (pc.DUP2,), (pc.X2, None, None, pc.CV2), (pc.X2, None, 2)]:
        a, b = ea.ndsortDED(*args), ea.ndsortESS(*args)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]


def test_mergecv_sums_each_row():
    expected = np.where(pc.CV2 > 0, pc.CV2, 0).sum(1, keepdims=True)
    np.testing.assert_allclose(ea.mergecv(pc.CV2), expected)
    np.testing.assert_allclose(ea.mergecv(pc.CV2, np.array([0.1, 0.0])),
                               np.where(pc.CV2 > [0.1, 0.0], pc.CV2, 0).sum(1, keepdims=True))
    v, count = ea.mergecv(pc.CV2, 0.05, True)
    assert count == int((expected <= 0.05).sum())


def _refselect_reference(obj, levels, cri, need, z):
    """Deb & Jain NSGA-III niching with geatpy's deterministic tie rules (first least-crowded point,
    closest individual for an empty niche, otherwise the first candidate)."""
    n, m = obj.shape
    idx1, idx2 = np.where(levels < cri)[0], np.where(levels == cri)[0]
    c = obj[np.r_[idx1, idx2]]
    c = c - c.min(0)
    w = np.full((m, m), 1e-6) + np.eye(m) * (1 - 1e-6)
    ext = [int(np.argmin(np.max(c / w[i], 1))) for i in range(m)]
    a = c[ext]
    h = 1 / c.max(0) if np.linalg.det(a) == 0 else np.linalg.solve(a, np.ones(m))
    c = c * h
    norm = np.sqrt((c ** 2).sum(1))
    cos = (c / norm[:, None]) @ (z / np.sqrt((z ** 2).sum(1))[:, None]).T
    dist = norm[:, None] * np.sqrt(np.maximum(1 - cos ** 2, 0))
    pi, d = dist.argmin(1), dist.min(1)
    n1 = len(idx1)
    rho = np.bincount(pi[:n1], minlength=len(z))
    excluded, chosen = np.zeros(len(z), bool), np.zeros(len(idx2), bool)
    while chosen.sum() < need - n1:
        cand = np.where(~excluded)[0]
        j = cand[rho[cand] == rho[cand].min()][0]
        members = np.where((pi[n1:] == j) & ~chosen)[0]
        if len(members) == 0:
            excluded[j] = True
            continue
        s = members[np.argmin(d[n1 + members])] if rho[j] == 0 else members[0]
        chosen[s], rho[j] = True, rho[j] + 1
    flag = np.zeros(n, bool)
    flag[idx1] = True
    flag[idx2[chosen]] = True
    return flag


@pytest.mark.parametrize('objv,need,npoints,mom', [(pc.X2, 20, 20, None), (pc.X3, 30, 28, None), (pc.X4, 20, 35, None),
                                                  (pc.X2, 25, 12, pc.MOM), (pc.X3, 45, 91, None)])
def test_refselect_matches_reference(objv, need, npoints, mom):
    levels, cri = ea.ndsortESS(objv, need, None, None, mom)
    z = ea.crtup(objv.shape[1], npoints)[0]
    got = ea.refselect(objv, levels, cri, need, z, mom)
    expected = _refselect_reference(objv * (1 if mom is None else mom), levels, cri, need, z)
    np.testing.assert_array_equal(got, expected)
    assert got.sum() == need


_RUNS = {r[0]: r[1:] for r in pc.template_runs(ea)}


@pytest.mark.skipif(os.environ.get('GEATPY_PARITY_TEMPLATES') != '1', reason='set GEATPY_PARITY_TEMPLATES=1')
@pytest.mark.parametrize('key', sorted(GOLDEN['templates']))
def test_template_quality(key):
    golden = GOLDEN['templates'][key]
    if golden['error']:
        pytest.skip('original core fails: ' + golden['error'])
    name, mk, enc, nind, gen = _RUNS[key]
    # fixed seeds, so the check does not fail at random (the median of a few runs is noisy)
    vals = []
    for r in range(len(golden['values'])):
        np.random.seed(1000 + r)
        vals.append(pc.run_template(ea, name, mk, enc, nind, gen))
    ref, got = float(np.median(golden['values'])), float(np.median(vals))
    if name.startswith('moea'):
        limit = 2.0 * ref + 0.01          # IGD
    else:
        limit = 1.5 * ref + 2.0           # best objective on Rastrigin (multimodal, noisy)
    assert got <= limit, 'median %.4g exceeds %.4g (original median %.4g)' % (got, limit, ref)
