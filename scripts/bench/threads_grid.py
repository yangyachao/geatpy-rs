# -*- coding: utf-8 -*-
"""Compare one thread with the default thread count over a grid of sizes of the parallel operators.

usage:
  python -I threads_grid.py [OPERATOR] [GAP]

OPERATOR is gd, sp (Spacing), cd (cdist), rs (refselect), rg (refgselect), nd (ndsortESS) or all
(the default); GAP is the idle time in seconds before each call (default 1e-3). With a gap the
pool's workers fall asleep before each call, as they do between two operator calls in an
algorithm, so the hand-off includes waking them; GAP 0 times back-to-back calls. Serial and default
calls alternate, so clock changes hit both alike. Prints the median of each and their ratio, then
the worst ratios and per-operator counts of ratios above 1.10. The thresholds in
crates/geatpy_core/src/parallel.rs were set with this script; run it from outside the repository
(or with -I) so the installed geatpy is used.
"""
import collections
import sys
import time

import numpy as np

import geatpy as ea

GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3


def busy(sec):
    end = time.perf_counter() + sec
    while time.perf_counter() < end:
        pass


def median_us(f, reps, threads):
    xs = []
    for _ in range(reps):
        ea.set_num_threads(threads)
        busy(GAP)
        start = time.perf_counter()
        f()
        xs.append(time.perf_counter() - start)
    return float(np.median(xs)) * 1e6


def cases(which):
    rng = np.random.default_rng(3)
    ns = [int(x) for x in np.geomspace(60, 3000, 16)]
    if which in ('gd', 'all'):
        for m in [2, 3, 8, 12]:
            for n in ns:
                a, pf = rng.random((n, m)), rng.random((n, m))
                yield 'GD n=%d m=%d' % (n, m), lambda a=a, pf=pf: ea.indicator.GD(a, pf)
    if which in ('sp', 'all'):
        for m in [2, 3, 8, 12]:
            for n in ns:
                a = rng.random((n, m))
                yield 'Spacing n=%d m=%d' % (n, m), lambda a=a: ea.indicator.Spacing(a)
    if which in ('cd', 'all'):
        for d in [3, 10, 30]:
            for n in ns:
                a, b = rng.random((n, d)), rng.random((n, d))
                yield 'cdist n=%d d=%d' % (n, d), lambda a=a, b=b: ea.cdist(a, b, 'euclidean')
    if which in ('rs', 'rg', 'all'):
        for m, pts in [(3, 50), (3, 100), (3, 300), (5, 100), (8, 100), (8, 300)]:
            ref = ea.crtup(m, pts)[0]
            for n in ns:
                obj, levels = rng.random((n, m)), np.ones(n)
                if which != 'rg':
                    yield ('refselect n=%d m=%d z=%d' % (n, m, len(ref)),
                           lambda obj=obj, levels=levels, n=n, ref=ref:
                           ea.refselect(obj, levels, 1, max(1, n // 2), ref, None))
                if which != 'rs':
                    yield ('refgselect n=%d m=%d z=%d' % (n, m, len(ref)),
                           lambda obj=obj, ref=ref: ea.refgselect(obj, ref, 2.0, None, None, None))
    if which in ('nd', 'all'):
        for kind in ['rand', 'sphere']:
            for m in [3, 5, 8, 12]:
                for n in [600, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 16000]:
                    x = rng.random((n, m))
                    if kind == 'sphere':
                        x = x / np.linalg.norm(x, axis=1, keepdims=True) * (1 + 0.05 * rng.random((n, 1)))
                    yield 'ndsortESS %s n=%d M=%d' % (kind, n, m), lambda x=x, n=n: ea.ndsortESS(x, n // 2)


def main():
    rows = []
    for name, f in cases(sys.argv[1] if len(sys.argv) > 1 else 'all'):
        ea.set_num_threads(None)
        f()
        ea.set_num_threads(1)
        f()
        first = median_us(f, 3, 1)
        reps = int(min(200, max(21, 0.15 / (GAP + first * 1e-6))))  # about 0.15 s per measurement
        serial, default = [], []
        for _ in range(reps):
            serial.append(median_us(f, 1, 1))
            default.append(median_us(f, 1, None))
        s, p = float(np.median(serial)), float(np.median(default))
        rows.append((name, s, p))
        print('%-34s %10.1f %10.1f %5.2f' % (name, s, p, p / s), flush=True)
    ea.set_num_threads(None)
    print('worst:')
    for name, s, p in sorted(rows, key=lambda r: r[2] / r[1], reverse=True)[:5]:
        print('  %s %.0f -> %.0f us (%.2f)' % (name, s, p, p / s))
    groups = collections.defaultdict(list)
    for name, s, p in rows:
        groups[name.split()[0]].append(p / s)
    for k, v in groups.items():
        print('%-10s cases %3d  >1.10: %2d  best %.2f  median %.2f'
              % (k, len(v), sum(r > 1.10 for r in v), min(v), float(np.median(v))))


if __name__ == '__main__':
    main()
