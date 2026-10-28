"""Seconds per generation and peak memory of algorithm templates on problems with 100 000 variables.

Each case runs in its own process, so the peak memory is that of one run:

    python -I scripts/parity/large_scale.py list
    python -I scripts/parity/large_scale.py run SEGA_BG_01 [--dim 500000]

prints one JSON line. Run every case with each geatpy version (official 2.7.0, an older geatpy-rs, this one)
and compare; README's large-scale table was made this way (median of 3 runs). Written for Python 3.6+
so that it also runs with the official 2.7.0 wheels; macOS and Linux only (peak memory comes from `resource`).
"""
import argparse
import json
import resource
import sys
import time

import numpy as np

import geatpy as ea


class Knapsack01(ea.Problem):
    """0-1 problem like geatpy-dev/geatpy#295: linear objective, one linear capacity constraint."""

    def __init__(self, Dim):
        rs = np.random.RandomState(0)
        self.w, self.v = rs.rand(Dim), rs.rand(Dim)
        self.cap = self.w.sum() / 2
        ea.Problem.__init__(self, 'Knapsack01', 1, [-1], Dim, [1] * Dim, [0] * Dim, [1] * Dim)

    def evalVars(self, x):
        return x @ self.v[:, None], x @ self.w[:, None] - self.cap


# name: (template, problem, encoding, NIND, MAXGEN)
CASES = {
    'DE_rand_1_bin_RI': ('soea_DE_rand_1_bin_templet', lambda d: ea.benchmarks.Sphere(d), 'RI', 50, 20),
    'DE_best_1_L_RI': ('soea_DE_best_1_L_templet', lambda d: ea.benchmarks.Sphere(d), 'RI', 50, 20),
    'SEGA_RI': ('soea_SEGA_templet', lambda d: ea.benchmarks.Sphere(d), 'RI', 50, 20),
    'EGA_RI': ('soea_EGA_templet', lambda d: ea.benchmarks.Sphere(d), 'RI', 50, 20),
    'SEGA_BG_01': ('soea_SEGA_templet', lambda d: Knapsack01(d), 'BG', 50, 20),
    'NSGA2_RI': ('moea_NSGA2_templet', lambda d: ea.benchmarks.ZDT1(Dim=d), 'RI', 100, 10),
}


def run(name, dim):
    templet, make_problem, encoding, nind, maxgen = CASES[name]
    np.random.seed(1)
    problem = make_problem(dim)
    algorithm = getattr(ea, templet)(problem, ea.Population(Encoding=encoding, NIND=nind), MAXGEN=maxgen, logTras=0)
    algorithm.verbose, algorithm.drawing = False, 0  # drawing=1 would open a plot window at the end
    evaluation = [0.0]
    evaluate = problem.evaluation

    def timed(pop):
        start = time.perf_counter()
        evaluate(pop)
        evaluation[0] += time.perf_counter() - start

    problem.evaluation = timed
    start = time.perf_counter()
    result = algorithm.run()
    seconds = time.perf_counter() - start
    gens = max(1, algorithm.currentGen)
    if hasattr(algorithm, 'BestIndi'):
        best = float(algorithm.BestIndi.ObjV[0][0]) if algorithm.BestIndi.sizes else float('nan')
    else:
        best = result[0].sizes  # multi-objective: size of the non-dominated set
    # ru_maxrss is in bytes on macOS and in KiB on Linux
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == 'darwin' else 2 ** 10)
    return {'case': name, 'dim': dim, 'version': ea.__version__, 'gens': gens, 'sec_per_gen': seconds / gens,
            'eval_share': evaluation[0] / seconds, 'peak_rss_mb': rss, 'best': best}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['list', 'run'])
    parser.add_argument('case', nargs='?')
    parser.add_argument('--dim', type=int, default=100000)
    args = parser.parse_args()
    if args.command == 'list':
        print(' '.join(CASES))
    else:
        print(json.dumps(run(args.case, args.dim)))
