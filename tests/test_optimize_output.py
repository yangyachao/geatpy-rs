"""Results and messages of ea.optimize that the Python layer fixes on top of geatpy 2.7.0."""
import contextlib
import io

import numpy as np
import pytest

import geatpy as ea


class ZDT1(ea.Problem):
    def __init__(self, D=10):
        ea.Problem.__init__(self, 'ZDT1', 2, [1, 1], D, [0] * D, [0] * D, [1] * D)

    def evalVars(self, X):
        f1 = X[:, [0]]
        g = 1 + 9 * X[:, 1:].mean(1, keepdims=True)
        return np.hstack([f1, g * (1 - np.sqrt(f1 / g))])


def _optimize(alg, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        res = ea.optimize(alg, seed=1, verbose=False, drawing=0, drawLog=False, saveFlag=False, **kw)
    return res, out.getvalue()


@pytest.mark.parametrize('name', ['moea_NSGA2_archive_templet', 'moea_awGA_templet', 'moea_MOEAD_archive_templet'])
def test_archived_pareto_sets_have_no_duplicate_objectives(name):
    # geatpy-dev/geatpy#376: the global archive was returned with repeated solutions
    res, _ = _optimize(getattr(ea, name)(ZDT1(), ea.Population('RI', NIND=50), MAXGEN=20, logTras=0),
                       outputMsg=False)
    assert len(np.unique(res['ObjV'], axis=0)) == len(res['ObjV'])


def test_no_feasible_solution_reports_the_violated_constraints():
    class Impossible(ea.Problem):
        def __init__(self):
            ea.Problem.__init__(self, 'I', 1, [1], 3, [0] * 3, [0] * 3, [1] * 3)

        def evalVars(self, X):
            return X.sum(1, keepdims=True), np.hstack([2.5 - X[:, [0]] - X[:, [1]], X[:, [2]] - 0.5])

    res, out = _optimize(ea.soea_SEGA_templet(Impossible(), ea.Population('RI', NIND=20), MAXGEN=5, logTras=0))
    assert not res['success']
    assert 'Could not find any feasible solution.' in out
    assert 'CV[:, 0]: 20/20 individuals violate it' in out
    assert 'CV[:, 1]: ' in out and 'needs a tolerance' in out


@pytest.mark.parametrize('with_reference', [False, True])
def test_hypervolume_uses_objective_directions_and_reference(tmp_path, monkeypatch, with_reference):
    monkeypatch.chdir(tmp_path)
    for sign in (1, -1):
        p = ea.Problem(str(sign), 2, [1, sign], 1, [0], [0], [1],
                       evalVars=lambda x: np.hstack([x, sign * (1 - x**2)]))
        reference = np.array([[2., 2.]]) if with_reference else None
        p.ReferObjV = None if reference is None else reference * [1, sign]
        alg = ea.moea_NSGA2_templet(p, ea.Population('RI', NIND=16), MAXGEN=1, logTras=1)
        result, _ = _optimize(alg, outputMsg=False)
        expected = ea.indicator.HV(result['ObjV'] * [1, sign], reference)
        assert result['hv'] == alg.log['hv'][-1] == expected
        p.maxormins = [1, sign]
        if reference is not None:
            p.ReferObjV = (reference * [1, sign]).tolist()
        alg2 = ea.moea_NSGA2_templet(p, ea.Population('RI', NIND=16), MAXGEN=1, logTras=1)
        result2, _ = _optimize(alg2, outputMsg=False)
        assert result2['hv'] == alg2.log['hv'][-1] == expected


def test_same_second_default_outputs_have_separate_directories(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('time.strftime', lambda *_: 'same-second')
    for _ in range(2):
        alg = ea.moea_NSGA2_templet(ZDT1(), ea.Population('RI', NIND=10), MAXGEN=1)
        ea.optimize(alg, verbose=False, drawing=0, drawLog=False, outputMsg=False)
    assert len(list(tmp_path.glob('result of job */result info.txt'))) == 2
