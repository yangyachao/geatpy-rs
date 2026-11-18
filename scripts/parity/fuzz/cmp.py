# -*- coding: utf-8 -*-
"""Compare two fz.py result files (original core vs Rust core).

usage: python cmp.py ORIG.jsonl RUST.jsonl [Z]

Reports, per category: cases whose status differs (error/crash vs ok), return structure or dtype
differences, deterministic-vs-random mismatches, value differences of deterministic cases
(relative tolerance 1e-9) and random cases whose element-wise means differ by more than Z
(default 6) standard errors.
"""
import json, sys, math, collections
import numpy as np
def load(p):
    d = {}
    for l in open(p):
        if l.startswith('{'):
            r = json.loads(l); d[r['name']] = r
    return d
A, B = load(sys.argv[1]), load(sys.argv[2])
Z = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
def struct(v):
    if v['t'] == 'nd': return ('nd', v['dtype'], tuple(v['shape']))
    if v['t'] in ('list', 'tuple'): return (v['t'], tuple(struct(e) for e in v['items']))
    return (v['t'],)
def vals(v):
    if v['t'] == 'nd': return [np.nan if x is None else x for x in v['data']]
    if v['t'] in ('list', 'tuple'): return sum((vals(e) for e in v['items']), [])
    if v['t'] in ('int', 'float', 'bool'): return [np.nan if v['v'] is None else float(v['v'])]
    return [v.get('v')]
def close(a, b):
    if len(a) != len(b): return False
    for x, y in zip(a, b):
        if isinstance(x, str) or isinstance(y, str):
            if x != y: return False
            continue
        if x is None or y is None:
            if x is not y: return False
            continue
        if np.isnan(x) and np.isnan(y): continue
        if x == y: continue
        if not (math.isfinite(x) and math.isfinite(y)): return False
        if not abs(x - y) <= 1e-9 * max(1, abs(x), abs(y)): return False
    return True
out = collections.defaultdict(list)
for n, a in A.items():
    b = B.get(n)
    if b is None: out['missing'].append(n); continue
    if a['status'] != 'ok' or b['status'] != 'ok':
        different_error = (a['status'] == b['status'] == 'error' and
                           a.get('error', '').split() != b.get('error', '').split())
        if a['status'] != b['status'] or different_error:
            out['status'].append('%s: orig=%s rust=%s' % (n, a.get('error', a['status'])[:110], b.get('error', b['status'])[:110]))
        continue
    sa, sb = struct(a['value']), struct(b['value'])
    va, vb = a.get('varied', False), b.get('varied', False)
    if sa != sb:
        out['struct'].append('%s: orig=%s rust=%s' % (n, str(sa)[:120], str(sb)[:120]))
    if va != vb:
        out['determinism'].append('%s: orig varied=%s rust varied=%s' % (n, va, vb)); continue
    if not va:
        if not close(vals(a['value']), vals(b['value'])):
            x, y = vals(a['value']), vals(b['value'])
            out['value'].append('%s:\n     orig=%s\n     rust=%s' % (n, x[:16], y[:16]))
        continue
    if not (a.get('shapes_stable') and b.get('shapes_stable')):
        if a.get('shapes_stable') != b.get('shapes_stable'):
            out['shape-stability'].append(n)
        continue
    for k in a['stat']:
        p, q = a['stat'][k], b['stat'].get(k)
        if q is None or len(p['mean']) != len(q['mean']): out['stat-shape'].append(n); break
        m1, m2 = np.array(p['mean']), np.array(q['mean'])
        se = np.sqrt(np.array(p['var']) / p['n'] + np.array(q['var']) / q['n']) + 1e-12
        z = np.abs(m1 - m2) / se
        zero_var = (np.array(p['var']) < 1e-15) & (np.array(q['var']) < 1e-15) & (np.abs(m1 - m2) > 1e-9)
        if z.max() > Z or zero_var.any():
            i = int(np.argmax(z if not zero_var.any() else zero_var))
            out['stat'].append('%s%s: max z=%.1f at %d (orig mean %.4f, rust %.4f) [%d/%d elems > %g]' % (n, k, z.max(), i, m1[i], m2[i], int((z > Z).sum()), len(z), Z))
            break
print('cases', len(A), 'rust', len(B))
for cat in ['missing', 'status', 'struct', 'determinism', 'value', 'shape-stability', 'stat-shape', 'stat']:
    print('\n#### %s (%d)' % (cat, len(out[cat])))
    for s in out[cat]: print('  ', s)
