from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

import geatpy


@geatpy.Problem.single
def eval_function(indiv):
    return np.sum(indiv)


@geatpy.Problem.single
def eval_function_with_cv(indiv):
    return np.sum(indiv), np.asarray(indiv)


def test_Problem_single():
    indivs = [[1, 2, 3], [4, 5, 6]]
    assert np.array_equal(eval_function(indivs), [[6], [15]])
    obj, cv = eval_function_with_cv(indivs)
    assert np.array_equal(obj, [[6], [15]])
    assert np.array_equal(cv, [[1, 2, 3], [4, 5, 6]])


def test_Problem_single_with_single_indiv():
    indivs = [1, 2, 3]
    assert np.array_equal(eval_function(indivs), [[6]])
    obj, cv = eval_function_with_cv(indivs)
    assert np.array_equal(obj, [[6]])
    assert np.array_equal(cv, [[1, 2, 3]])


@pytest.fixture
def population_only_with_Phen():
    indivs = np.array([[1, 2, 3], [4, 5, 6]])
    yield geatpy.Population(None, Phen=indivs)


def test_Problem_evaluation_aimFunc_evalVars_invalid(
        population_only_with_Phen):
    pop = population_only_with_Phen
    problem = geatpy.Problem('test',
                             M=1,
                             maxormins=[1],
                             Dim=3,
                             varTypes=np.zeros(3),
                             lb=np.zeros(3),
                             ub=10 * np.ones(3))
    with pytest.raises(RuntimeError):
        problem.evaluation(pop)


def test_Problem_evaluation_with_evalVars(population_only_with_Phen):
    pop = population_only_with_Phen
    problem = geatpy.Problem('test',
                             M=1,
                             maxormins=[1],
                             Dim=3,
                             varTypes=np.zeros(3),
                             lb=np.zeros(3),
                             ub=10 * np.ones(3),
                             evalVars=eval_function)
    problem.evaluation(pop)
    assert np.array_equal(pop.ObjV, [[6], [15]])


def test_Problem_evaluation_with_aimFunc_and_evalVars(
        population_only_with_Phen):

    def aim_func_example(pop):
        pop.ObjV = [[10]]

    pop = population_only_with_Phen
    problem = geatpy.Problem('test',
                             M=1,
                             maxormins=[1],
                             Dim=3,
                             varTypes=np.zeros(3),
                             lb=np.zeros(3),
                             ub=10 * np.ones(3),
                             aimFunc=aim_func_example,
                             evalVars=eval_function)
    problem.evaluation(pop)
    assert np.array_equal(pop.ObjV, [[10]])


def test_reference_cache_is_published_after_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    started = Event()
    finish = Event()
    savetxt = np.savetxt

    def pause_first_write(stream, values, **kwargs):
        if started.is_set():
            return savetxt(stream, values, **kwargs)
        savetxt(stream, values[:2], **kwargs)
        stream.flush()
        started.set()
        assert finish.wait(10)
        savetxt(stream, values[2:], **kwargs)

    monkeypatch.setattr(np, 'savetxt', pause_first_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(geatpy.benchmarks.DTLZ2)
        try:
            assert started.wait(10)
            second = geatpy.benchmarks.DTLZ2()
        finally:
            finish.set()
        np.testing.assert_array_equal(second.ReferObjV, first.result().ReferObjV)
    assert len(list((tmp_path / 'referenceObjV').iterdir())) == 1


def test_failed_reference_cache_write_keeps_previous_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    problem = geatpy.benchmarks.DTLZ2()
    cache, = (tmp_path / 'referenceObjV').iterdir()
    original = cache.read_bytes()

    def fail_write(stream, *_args, **_kwargs):
        stream.write('partial data')
        raise OSError('write failed')

    monkeypatch.setattr(np, 'savetxt', fail_write)
    with pytest.raises(OSError, match='write failed'):
        problem.getReferObjV(reCalculate=True)
    assert cache.read_bytes() == original
    assert list(cache.parent.iterdir()) == [cache]


@pytest.mark.parametrize('values', [[[1], [2], [3]], [[1, 2, 3]]])
def test_reference_cache_preserves_matrix_shape(tmp_path, monkeypatch, values):
    monkeypatch.chdir(tmp_path)
    values = np.array(values, dtype=float)
    objectives = values.shape[1]

    def reference():
        return values

    def create_problem():
        return geatpy.Problem('test', objectives, [1] * objectives, 1,
                              [0], [0], [1], calReferObjV=reference)

    np.testing.assert_array_equal(create_problem().ReferObjV, values)
    np.testing.assert_array_equal(create_problem().ReferObjV, values)

    cache, = (tmp_path / 'referenceObjV').iterdir()
    np.savetxt(cache, np.zeros((2, objectives + 1)), delimiter=',')
    np.testing.assert_array_equal(create_problem().ReferObjV, values)
    np.testing.assert_array_equal(np.loadtxt(cache, delimiter=',', ndmin=2), values)
