"""Population slicing and merging hand out arrays the new population owns, so it must stay
independent of its source (as in geatpy 2.7.0, which copied every array twice)."""
import numpy as np
import pytest

import geatpy as ea


def _pop():
    N, D = 6, 4
    rng = np.random.default_rng(0)
    return ea.Population(
        "RI",
        ea.crtfld("RI", np.zeros(D), np.vstack([np.zeros(D), np.ones(D)]), np.ones((2, D))),
        N,
        rng.random((N, D)),
        rng.random((N, 2)),
        rng.random((N, 1)),
        rng.random((N, 1)),
        rng.random((N, D)),
    )


def _psy_pop():
    N = 6
    varTypes = np.array([1, 1, 0, 0])
    ranges = np.array([[0, 0, 0, 0], [5, 5, 1, 1]])
    borders = np.ones((2, 4))
    pop = ea.PsyPopulation(["BG", "RI"], (varTypes, ranges, borders), N, EncoIdxs=[[0, 1], [2, 3]])
    pop.initChrom(N)
    rng = np.random.default_rng(1)
    pop.ObjV, pop.FitnV, pop.CV, pop.Phen = rng.random((N, 2)), rng.random((N, 1)), rng.random((N, 1)), rng.random((N, 4))
    return pop


ATTRS = ("ObjV", "FitnV", "CV", "Phen")


def _arrays(pop):
    chroms = pop.Chroms if isinstance(pop, ea.PsyPopulation) else [pop.Chrom]
    return list(chroms) + [getattr(pop, a) for a in ATTRS]


def _assert_independent(new, *sources):
    for a in _arrays(new):
        for src in sources:
            for b in _arrays(src):
                assert not np.shares_memory(a, b)


@pytest.mark.parametrize("make", [_pop, _psy_pop])
@pytest.mark.parametrize("index", [[0, 2, 4], np.array([1, 3]), np.array([True, False] * 3), 2, np.int64(5), slice(1, 4), []])
def test_getitem_is_independent(make, index):
    pop = make()
    before = [a.copy() for a in _arrays(pop)]
    sub = pop[index]
    _assert_independent(sub, pop)
    for a in _arrays(sub):
        a[...] = -1
    for a, b in zip(_arrays(pop), before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("make", [_pop, _psy_pop])
def test_add_is_independent(make):
    p1, p2 = make(), make()[[0, 1]]
    merged = p1 + p2
    assert merged.sizes == p1.sizes + p2.sizes
    _assert_independent(merged, p1, p2)
    for a, b, c in zip(_arrays(merged), _arrays(p1), _arrays(p2)):
        np.testing.assert_array_equal(a, np.vstack([b, c]))


def test_lind_and_sizes_match_constructor():
    pop = _pop()
    for index in ([0, 1, 2], slice(0, 3)):
        sub = pop[index]
        assert (sub.sizes, sub.Lind, sub.ChromNum) == (3, pop.Lind, 1)
    psy = _psy_pop()
    sub = psy[[0, 1, 2]]
    assert (sub.sizes, sub.Linds, sub.ChromNum) == (3, [c.shape[1] for c in psy.Chroms], 2)


# Original geatpy 2.7.0 defects in the Python layer, fixed (2026-10-28 review).

def _ri_population(n=5):
    field = ea.crtfld('RI', np.zeros(3), np.array([[0] * 3, [1] * 3]))
    pop = ea.Population('RI', field, n)
    pop.initChrom()
    return pop


@pytest.mark.parametrize('index, want', [
    (slice(None), 5), (slice(None, -1), 4), (slice(None, 9), 5), (slice(1, 3), 2),
    (slice(None, None, 2), 3), (slice(-2, None), 2),
])
def test_slice_size_matches_the_selected_rows(index, want):
    sub = _ri_population()[index]
    assert sub.sizes == want == sub.Chrom.shape[0]


def test_setitem_without_phenotypes():
    pop, other = _ri_population(), _ri_population()
    pop.Phen = None
    pop[[0, 1]] = other[[2, 3]]
    assert pop.sizes == 5


def _psy_population(n=4):
    fields = [ea.crtfld('RI', np.zeros(2), np.array([[0] * 2, [1] * 2])),
              ea.crtfld('P', np.ones(3), np.array([[0] * 3, [2] * 3]))]
    pop = ea.PsyPopulation(['RI', 'P'], fields, n)
    pop.initChrom()
    return pop


def test_psy_init_chrom_twice_keeps_the_lengths():
    pop = _psy_population()
    pop.initChrom()
    assert pop.Linds == [2, 3]


def test_psy_slices_and_shuffle():
    pop = _psy_population()
    assert (pop[:].sizes, pop[:-1].sizes, pop[:9].sizes) == (4, 3, 4)
    ea.PsyPopulation(None, None, 3).shuffle()  # no chromosomes to shuffle


def test_psy_population_does_not_share_its_fields():
    pop = _psy_population()
    for derived in (pop[[0, 1]], pop.copy(), pop + pop):
        assert not any(np.shares_memory(a, b) for a, b in zip(pop.Fields, derived.Fields))
