# -*- coding: utf-8 -*-
"""Randomised differential fuzzing of the geatpy core (original 2.7.0 vs Rust).

usage: python fz.py OUT.jsonl [REPS_SCALE]   (or through drive.sh)
Every case is called REPS times (scaled by REPS_SCALE) on identical inputs. The first result is recorded exactly; if the
results vary between calls, element-wise mean/variance over all calls is recorded as well, so the
comparison can be exact for deterministic operators and statistical for random ones.
Cases already present in OUT.jsonl are skipped; "START <name>" lines let a driver detect hangs.
Python 3.6 compatible.
"""
import json
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import geatpy as ea  # noqa: E402


def enc(x):
    if isinstance(x, np.ndarray):
        a = x.astype(float) if x.dtype.kind in 'biuf' else x
        data = [None if (isinstance(v, float) and v != v) else v for v in np.ravel(a).tolist()]
        return {'t': 'nd', 'dtype': x.dtype.kind + str(x.dtype.itemsize), 'shape': list(x.shape), 'data': data}
    if isinstance(x, (list, tuple)):
        return {'t': type(x).__name__, 'items': [enc(e) for e in x]}
    if isinstance(x, (bool, np.bool_)):
        return {'t': 'bool', 'v': bool(x)}
    if isinstance(x, (int, np.integer)):
        return {'t': 'int', 'v': int(x)}
    if isinstance(x, (float, np.floating)):
        return {'t': 'float', 'v': None if x != x else float(x)}
    if x is None:
        return {'t': 'None'}
    return {'t': type(x).__name__, 'v': str(x)[:200]}


def leaves(x, out, path=''):
    if isinstance(x, np.ndarray):
        out[path] = x
    elif isinstance(x, (list, tuple)):
        for i, e in enumerate(x):
            leaves(e, out, path + '/%d' % i)
    elif isinstance(x, (int, float, np.integer, np.floating, bool, np.bool_)):
        out[path] = np.array(x, dtype=float)


def run(f, reps):
    try:
        r0 = f()
    except BaseException as e:  # noqa
        return {'status': 'error', 'error': '%s: %s' % (type(e).__name__, str(e)[:300])}
    rec = {'status': 'ok', 'value': enc(r0)}
    if reps <= 1:
        return rec
    L0 = {}
    leaves(r0, L0)
    means = {k: np.zeros(np.shape(v)) for k, v in L0.items()}
    m2 = {k: np.zeros(np.shape(v)) for k, v in L0.items()}
    finite_counts = {k: np.zeros(np.shape(v), dtype=int) for k, v in L0.items()}
    special = {}
    varied, shapes_ok, n = False, True, 0
    for _ in range(reps):
        r = f()
        L = {}
        leaves(r, L)
        if set(L) != set(L0) or any(np.shape(L[k]) != np.shape(L0[k]) for k in L0):
            shapes_ok = False
            varied = True
            continue
        n += 1
        for k in L0:
            v = np.asarray(L[k], dtype=float)
            first = L0[k]
            if not varied:
                varied = not np.all((v == first) | (np.isnan(v) & np.isnan(first)))
            finite = np.isfinite(v)
            if not finite.all():
                if k not in special:
                    special[k] = np.zeros((3,) + v.shape, dtype=int)
                special[k] += [np.isnan(v), np.isposinf(v), np.isneginf(v)]
            finite_counts[k] += finite
            # Welford's population variance, conditional on a finite observation.
            delta = np.where(finite, v, means[k]) - means[k]
            means[k] += delta / np.maximum(finite_counts[k], 1)
            m2[k] += delta * (np.where(finite, v, means[k]) - means[k])
    rec['varied'] = varied
    rec['shapes_stable'] = shapes_ok
    if varied and n:
        rec['stat'] = {}
        for k in L0:
            count = finite_counts[k]
            stat = {'n': n, 'mean': means[k].ravel().tolist(),
                    'var': (m2[k] / np.maximum(count, 1)).ravel().tolist()}
            if k in special:
                stat['special'] = [a.ravel().tolist() for a in special[k]]
            rec['stat'][k] = stat
    return rec


def R(seed):
    return np.random.RandomState(seed)


def cases():
    C = []

    def add(name, f, reps=1):
        C.append((name, f, reps))

    # ---------------------------------------------------------------- encoding
    for i in range(60):
        r = R(1000 + i)
        D = r.randint(1, 4)
        code = ['BG', 'RI', 'P'][i % 3]
        vt = (r.rand(D) < 0.5).astype(int)
        lb = np.round(r.uniform(-5, 5, D), r.randint(0, 3))
        span = np.round(r.uniform(-0.5, 6, D), r.randint(0, 3))
        if i % 7 == 0:
            span[0] = 0
        rng = np.vstack([lb, lb + span])
        bor = (r.rand(2, D) < 0.5).astype(int)
        prec = list(r.randint(0, 4, D))
        codes = list(r.randint(0, 2, D))
        scales = list(r.randint(0, 2, D))
        k = i % 4
        if k == 0:
            add('crtfld_%d' % i, lambda c=code, a=vt, b=rng: ea.crtfld(c, a, b))
        elif k == 1:
            add('crtfld_%d' % i, lambda c=code, a=vt, b=rng, d=bor: ea.crtfld(c, a, b, d))
        else:
            add('crtfld_%d' % i, lambda c=code, a=vt, b=rng, d=bor, p=prec, cs=codes, s=scales:
                ea.crtfld(c, a, b, d, p, cs, s))

    for i in range(24):
        r = R(2000 + i)
        D = r.randint(1, 4)
        vt = (r.rand(D) < 0.5).astype(int)
        lb = np.round(r.uniform(-5, 5, D), 1)
        rng = np.vstack([lb, lb + np.round(r.uniform(0.5, 9, D), 1)])
        bor = (r.rand(2, D) < 0.5).astype(int)
        prec = list(r.randint(0, 3, D))
        codes = list(r.randint(0, 2, D))
        scales = list(r.randint(0, 2, D))
        try:
            FD = ea.crtfld('BG', vt, rng, bor, prec, codes, scales)
        except Exception:
            continue
        L = int(np.sum(FD[0]))
        ch = (r.rand(7, L) < 0.5).astype(int)
        fn = ['bs2ri', 'bs2int', 'bs2real'][i % 3]
        add('%s_%d' % (fn, i), lambda f=getattr(ea, fn), c=ch, F=FD: f(c, F))
        if i % 3 == 0:
            phen = ea.bs2ri(ch, FD)
            add('ri2bs_%d' % i, lambda p=phen, F=FD: ea.ri2bs(p, F))

    for i in range(24):
        r = R(3000 + i)
        D = r.randint(1, 5)
        vt = (r.rand(D) < 0.5).astype(int)
        lb = np.round(r.uniform(-5, 5, D), 1)
        span = np.round(r.uniform(0, 8, D), 1)
        if i % 5 == 0:
            span[0] = 0
        FDR = np.vstack([lb, lb + span, vt])
        ch = r.uniform(-12, 12, (6, D))
        if vt.all() and i % 2:
            ch = np.round(ch)
        ft = [None, 1, 2, 3, 4, 0, 5][i % 7]
        reps = 300 if ft == 4 else 1
        if ft is None:
            add('boundfix_%d' % i, lambda c=ch, F=FDR: ea.boundfix('RI', c, F), reps)
        else:
            add('boundfix_%d' % i, lambda c=ch, F=FDR, t=ft: ea.boundfix('RI', c, F, t), reps)

    for D in [2, 3, 4, 5]:
        for num in [D - 1, D, 7, 10, 21, 56, 100]:
            add('crtup_%d_%d' % (D, num), lambda d=D, n=num: ea.crtup(d, n))
    for i in range(10):
        r = R(4000 + i)
        D = r.randint(1, 4)
        vt = (r.rand(D) < 0.5).astype(int)
        lb = np.round(r.uniform(-3, 3, D), 1)
        FDR = np.vstack([lb, lb + np.round(r.uniform(0, 5, D), 1), vt])
        add('crtri_%d' % i, lambda F=FDR: ea.crtri(5, F), 500)
        add('crtip_%d' % i, lambda F=FDR: ea.crtip(5, F), 500)
        add('crtrp_%d' % i, lambda F=FDR: ea.crtrp(5, F), 500)
    for i, (n, l, p) in enumerate([(5, 6, None), (4, 5, 3), (6, 3, 5), (3, 4, 1)]):
        if p is None:
            add('crtbp_%d' % i, lambda n=n, l=l: ea.crtbp(n, l), 500)
        else:
            add('crtbp_%d' % i, lambda n=n, l=l, p=p: ea.crtbp(n, l, p), 500)
    for i, FDR in enumerate([np.array([[0] * 5, [4] * 5, [1] * 5]), np.array([[2] * 4, [8] * 4, [1] * 4]),
                             np.array([[0.] * 3, [2.] * 3, [1] * 3])]):
        add('crtpp_%d' % i, lambda F=FDR: ea.crtpp(6, F), 500)
    for i, code in enumerate(['BG', 'RI', 'P']):
        F = {'BG': ea.crtfld('BG', np.array([0, 1]), np.array([[0, 0], [3, 5]])),
             'RI': np.array([[0., -1], [2, 4], [0, 1]]), 'P': np.array([[0] * 5, [4] * 5, [1] * 5])}[code]
        add('crtpc_%d' % i, lambda c=code, F=F: ea.crtpc(c, 5, F), 500)

    # ---------------------------------------------------------------- selection / fitness
    for i in range(12):
        r = R(5000 + i)
        N = r.randint(2, 12)
        fit = np.round(r.rand(N, 1) * 5, 1 if i % 2 else 3)
        if i % 4 == 3:
            fit[r.randint(0, N)] = fit[0]
        nsel = [N, N // 2 + 1, 2 * N, 1, 3][i % 5]
        for fn in ['dup', 'ecs', 'tour', 'etour', 'rws', 'sus', 'urs', 'rcs', 'rps', 'otos']:
            add('%s_%d' % (fn, i), lambda f=getattr(ea, fn), a=fit, n=nsel: f(a, n, None), 2000)
        add('tour3_%d' % i, lambda a=fit, n=nsel: ea.tour(a, n, 3), 2000)
        add('etour4_%d' % i, lambda a=fit, n=nsel: ea.etour(a, n, 4), 2000)
        add('selecting_%d' % i, lambda a=fit, n=nsel, s=['tour', 'rws', 'sus', 'ecs'][i % 4]: ea.selecting(s, a, n), 2000)
    for i in range(10):
        r = R(5500 + i)
        N, M = r.randint(2, 9), r.randint(1, 3)
        ov = np.round(r.rand(N, M) * 4, 1)
        cv = (np.round(r.rand(N, 2) - 0.6, 2) if i % 2 else None)
        mm = (np.array([1] * M) if i % 3 == 0 else np.array([-1] + [1] * (M - 1)))
        o1, m1 = np.ascontiguousarray(ov[:, :1]), mm[:1]
        add('ranking_%d' % i, lambda o=o1, c=cv, m=m1: ea.ranking(o, c, m))
        add('scaling_%d' % i, lambda o=o1, c=cv, m=m1: ea.scaling(o, c, m))
        add('powing_%d' % i, lambda o=o1, c=cv, m=m1: ea.powing(o, c, m))
        add('indexing_%d' % i, lambda o=o1, c=cv, m=m1: ea.indexing(o, c, m))
        add('ranking_nl_%d' % i, lambda o=o1, c=cv, m=m1: ea.ranking(o, c, m, 1, 2))
        add('awGA_%d' % i, lambda o=ov, c=cv, m=mm: ea.awGA(o, c, m))
        add('rwGA_%d' % i, lambda o=ov, c=cv, m=mm: ea.rwGA(o, c, m), 1000)
    add('ranking_rm', lambda: ea.ranking(np.array([[3.], [1.], [2.], [1.]]), None, None, [2, 1]))
    add('ranking_sp', lambda: ea.ranking(np.array([[3.], [1.], [2.], [1.]]), None, None, 0, 1.7))
    add('scaling_smul', lambda: ea.scaling(np.array([[3.], [1.], [2.], [5.]]), None, None, 1.5))
    add('powing_k', lambda: ea.powing(np.array([[3.], [1.], [2.], [5.]]), None, None, 3))
    add('indexing_b', lambda: ea.indexing(np.array([[3.], [1.], [2.], [5.]]), None, None, 0.5))

    # ---------------------------------------------------------------- recombination
    def pop(kind, N, L, seed):
        r = R(seed)
        if kind == 'real':
            return np.round(r.uniform(-3, 3, (N, L)), 3)
        if kind == 'bin':
            return (r.rand(N, L) < 0.5).astype(int)
        if kind == 'int':
            return r.randint(0, 9, (N, L))
        return np.array([r.permutation(L) for _ in range(N)])

    idx = 0
    for fn, kind in [('recsbx', 'real'), ('recint', 'real'), ('reclin', 'real'), ('recndx', 'real'),
                     ('recdis', 'real'), ('xovsp', 'bin'), ('xovdp', 'int'), ('xovud', 'bin'),
                     ('xovsh', 'int'), ('xovsec', 'int'), ('xovbd', 'int'), ('xovexp', 'int'),
                     ('xovpmx', 'perm'), ('xovox', 'perm')]:
        f = getattr(ea, fn)
        for N, L in [(6, 7), (5, 6), (8, 1), (2, 9)]:
            ch = pop(kind, N, L, 6000 + idx)
            idx += 1
            for hn in [False, True]:
                for p in [None, 0.5, 1]:
                    if p is None:
                        add('%s_%dx%d_h%d_pd' % (fn, N, L, hn), lambda f=f, c=ch, h=hn: f(c, 0.7, h), 1500)
                    else:
                        add('%s_%dx%d_h%d_p%s' % (fn, N, L, hn, p), lambda f=f, c=ch, h=hn, p=p: f(c, p, h), 1500)
        if fn in ('xovsp', 'xovdp', 'xovud', 'xovsh', 'xovsec', 'xovbd', 'xovexp', 'recdis'):
            ch = pop(kind, 6, 8, 6500 + idx)
            gid = np.array([3, 3, 1, 1, 2, 2, 5, 5])
            add('%s_gid' % fn, lambda f=f, c=ch, g=gid: f(c, 1, False, g), 1500)
            gid2 = np.array([2, 1, 2, 1, 3, 3, 1, 2])
            add('%s_gid2' % fn, lambda f=f, c=ch, g=gid2: f(c, 1, True, g), 1500)
    ch = pop('perm', 6, 8, 6999)
    for m in [1, 2]:
        for p in [0.3, 1]:
            for hn in [False, True]:
                add('xovpmx_m%d_p%s_h%d' % (m, p, hn), lambda c=ch, p=p, h=hn, m=m: ea.xovpmx(c, p, h, m), 1500)
    chr_ = pop('real', 6, 5, 7000)
    add('recsbx_n', lambda: ea.recsbx(chr_, 1, False, 5), 1500)
    add('recsbx_n_h', lambda: ea.recsbx(chr_, 0.6, True, 40), 1500)
    add('recint_a', lambda: ea.recint(chr_, 1, False, 0.25), 1500)
    add('reclin_a', lambda: ea.reclin(chr_, 1, True, 0.25), 1500)
    add('recndx_a', lambda: ea.recndx(chr_, 1, False, 1.2), 1500)
    add('recombin_xovdp', lambda: ea.recombin('xovdp', pop('int', 6, 5, 7001), 1, True), 1500)

    # ---------------------------------------------------------------- mutation
    for i in range(10):
        r = R(8000 + i)
        D = r.randint(1, 6)
        vt = (r.rand(D) < [0.5, 0.0, 1.0][i % 3]).astype(int)
        lb = np.round(r.uniform(-4, 4, D), 1)
        span = np.round(r.uniform(0.5, 9, D), 1)
        if i == 9:
            span[0] = 0
        FDR = np.vstack([lb, lb + span, vt])
        ch = lb + r.rand(6, D) * span
        ch[:, vt == 1] = np.round(ch[:, vt == 1])
        ch = np.clip(ch, lb, lb + span)
        for fn, extra in [('mutpolyn', [None, None]), ('mutpolyn', [1, 5]), ('mutgau', [None, None]),
                          ('mutgau', [0.9, 1.0]), ('mutgau', [0.9, 1.0, False, 3]), ('mutbga', [None, None]),
                          ('mutbga', [0.8, 0.2, 10]), ('mutuni', [None, None]), ('mutuni', [0.8, 0.3]),
                          ('mutuni', [0.8, np.array([0.1] * D)])]:
            args = [a for a in extra]
            while args and args[-1] is None:
                args.pop()
            add('%s_%d_%s' % (fn, i, len(C)), lambda f=getattr(ea, fn), c=ch, F=FDR, a=args: f('RI', c, F, *a), 1500)
        xr = [np.arange(6), np.roll(np.arange(6), 1), np.roll(np.arange(6), 2)]
        add('mutde_%d' % i, lambda c=ch, F=FDR, x=xr: ea.mutde('RI', c, F, x, 0.5), 1500)
        add('mutde_ft_%d' % i, lambda c=ch, F=FDR, x=xr: ea.mutde('RI', c, F, x, 0.7, 2), 1500)
        add('mutde_m_%d' % i, lambda c=ch, F=FDR, x=xr, m=(np.arange(6 * D).reshape(6, D) % 2 == 0): ea.mutde('RI', c, F, x, 0.5, 1, m), 1500)
        # random donors (the cases above fix every index); squared outputs also compare the spread,
        # which tells drawing with replacement from refilling the pool in small populations
        if i < 4:
            for tag, x, f, extra in [('none', None, 0.5, []), ('r0', [np.arange(6)], 0.5, []),
                                     ('two', [None] * 5, [0.5, 0.3], []), ('fnone', [None] * 3, None, []),
                                     ('fx', [None] * 5, [0.4, None], []), ('ft3', [None] * 5, 0.9, [3]),
                                     ('ft4', [None] * 5, 0.9, [4]), ('mask4', None, 0.5, [1, 4]),
                                     ('mask9', [None] * 5, 0.5, [1, 9])]:
                add('mutde_rnd_%s_%d' % (tag, i), lambda c=ch, F=FDR, x=x, f=f, e=extra: ea.mutde('RI', c, F, x, f, *e) ** 2, 1500)
            for n in (2, 3, 4):
                add('mutde_small_%d_%d' % (n, i), lambda c=ch[:n], F=FDR: ea.mutde('RI', c, F, [None] * 5, [0.5, 0.5]) ** 2, 1500)
        for fn in ['mutinv', 'mutmove', 'mutswap']:
            add('%s_RI_%d' % (fn, i), lambda f=getattr(ea, fn), c=ch, F=FDR: f('RI', c, F, 1), 1500)
    for L in [1, 2, 5, 9]:
        P = np.array([R(9000 + L + k).permutation(L) for k in range(6)])
        for fn in ['mutinv', 'mutmove', 'mutswap']:
            add('%s_P_%d' % (fn, L), lambda f=getattr(ea, fn), c=P: f('P', c, None, 1), 1500)
            add('%s_P_%d_pm' % (fn, L), lambda f=getattr(ea, fn), c=P: f('P', c, None, 0.4), 1500)
        add('mutinv_P_%d_len' % L, lambda c=P: ea.mutinv('P', c, None, 1, 3), 1500)
        add('mutmove_P_%d_len' % L, lambda c=P: ea.mutmove('P', c, None, 1, 3, 0.8), 1500)
    B = (R(9100).rand(6, 9) < 0.5).astype(int)
    add('mutbin_d', lambda: ea.mutbin('BG', B), 1500)
    add('mutbin_pm', lambda: ea.mutbin('BG', B, None, 0.3), 1500)
    FP = np.array([[0] * 6, [5] * 6, [1] * 6])
    PP = np.array([R(9200 + k).permutation(6) for k in range(6)])
    add('mutpp_d', lambda: ea.mutpp('P', PP, FP), 1500)
    add('mutpp_n', lambda: ea.mutpp('P', PP, FP, 1, 2), 1500)

    # ---------------------------------------------------------------- multi-objective
    for i in range(16):
        r = R(10000 + i)
        N, M = r.randint(1, 25), r.randint(1, 4)
        ov = np.round(r.rand(N, M) * [3, 1][i % 2], [1, 3][i % 2])
        if i % 3 == 0 and N > 3:
            ov[1] = ov[0]
            ov[2, 0] = np.inf
        cv = np.round(r.rand(N, 2) - 0.7, 2) if i % 4 == 1 else None
        mm = [None, np.array([1] * M), np.array([-1] * M), np.array([1] + [-1] * (M - 1))][i % 4]
        need = [None, N // 2 + 1, 1, N][i % 4]
        for fn in ['ndsortESS', 'ndsortTNS', 'ndsortDED']:
            add('%s_%d' % (fn, i), lambda f=getattr(ea, fn), o=ov, n=need, c=cv, m=mm: f(o, n, None, c, m))
            add('%s_lv_%d' % (fn, i), lambda f=getattr(ea, fn), o=ov: f(o, None, 2))
        lv = ea.ndsortESS(ov)[0] if N else None
        add('crowdis_%d' % i, lambda o=ov, l=lv: ea.crowdis(o, l))
        add('crowdis_none_%d' % i, lambda o=ov: ea.crowdis(o, np.ones(len(o))))
        add('cdist_%d' % i, lambda o=ov: ea.cdist(o, np.ascontiguousarray(o[::-1])))
        add('cdist_m_%d' % i, lambda o=ov, k=['euclidean', 'cityblock', 'chebyshev', 'cosine'][i % 4]: ea.cdist(o, o[:3], k))
        add('crtidp_%d' % i, lambda o=ov, c=cv, m=mm: ea.crtidp(o, c, m))
        add('crtidp_rev_%d' % i, lambda o=ov, c=cv, m=mm: ea.crtidp(o, c, m, None, True))
        w = ea.crtup(M, max(N, M))[0][:N] if N >= 1 else None
        if w is not None and len(w) == N:
            add('tcheby_%d' % i, lambda o=ov, w=w: ea.tcheby(o, w, np.min(o, 0)))
            add('tcheby_cv_%d' % i, lambda o=ov, w=w, c=cv, m=mm: ea.tcheby(o, w, np.min(o, 0), c, m))
            add('pbi_%d' % i, lambda o=ov, w=w: ea.pbi(o, w, np.min(o, 0)))
            add('pbi_t_%d' % i, lambda o=ov, w=w, c=cv, m=mm: ea.pbi(o, w, np.min(o, 0), c, m, 3))
        if N >= 2 and M >= 2:
            ref = ea.crtup(M, 10)[0]
            add('refselect_%d' % i, lambda o=ov, l=lv, ref=ref, n=max(1, N // 2): ea.refselect(o, l, int(np.max(l)), n, ref), 800)
            add('refgselect_%d' % i, lambda o=ov, ref=ref, c=cv: ea.refgselect(o, ref, 2.0, c), 1)
            add('GD_%d' % i, lambda o=ov: ea.indicator.GD(o, o[: max(1, len(o) // 2)]))
            add('IGD_%d' % i, lambda o=ov: ea.indicator.IGD(o, o[: max(1, len(o) // 2)]))
            add('Spacing_%d' % i, lambda o=ov: ea.indicator.Spacing(o))
            if M <= 3:
                add('HV_%d' % i, lambda o=ov: ea.indicator.HV(o / 3.0, np.ones((1, o.shape[1]))))
    for i in range(6):
        r = R(11000 + i)
        cv = np.round(r.rand(8, r.randint(1, 4)) - 0.5, 2)
        add('mergecv_%d' % i, lambda c=cv: ea.mergecv(c))
    for i, (sizes, mg, st, sel, rep) in enumerate([([6, 6, 6], 0.2, 0, 0, 0), ([6, 6, 6], 0.4, 1, 1, 1),
                                                   ([5, 7, 4, 6], 0.3, 2, 0, 1), ([8, 8], 0.5, 0, 1, 0),
                                                   ([6, 6, 6], 0.2, 1, 0, 2)]):
        fv = [np.arange(s).reshape(-1, 1).astype(float) for s in sizes]
        add('migrate_%d' % i, lambda s=sizes, m=mg, t=st, se=sel, rp=rep, f=fv: ea.migrate(s, m, t, se, rp, f), 1500)
    return C


# Inputs that crash the original 2.7.0 core (segfault on a chromosome of length 1 or a 1-D input).
# They are recorded as crashes without running them there, so that no crash reports pile up.
ORIGINAL = getattr(ea, '__version__', '') == '2.7.0'
KNOWN_CRASH = re.compile(r'^(xovsp|xovsh)_\d+x1_|^(rec(sbx|int|lin|dis)|xov[a-z]+)_1d$')


def main():
    out = sys.argv[1]
    reps_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    done = set()
    try:
        for line in open(out):
            if line.startswith('{'):
                done.add(json.loads(line)['name'])
            elif line.startswith('START '):
                done.add(line.split()[1])
    except IOError:
        pass
    fh = open(out, 'a')
    for name, f, reps in cases():
        if name in done:
            continue
        if ORIGINAL and KNOWN_CRASH.match(name):
            fh.write(json.dumps({'name': name, 'status': 'crash', 'error': 'known crash of the original core'}) + '\n')
            continue
        fh.write('START %s\n' % name)
        fh.flush()
        np.random.seed(0)
        rec = run(f, 1 if reps <= 1 else max(2, int(reps * reps_scale)))
        rec['name'] = name
        fh.write(json.dumps(rec) + '\n')
        fh.flush()


if __name__ == '__main__':
    main()
