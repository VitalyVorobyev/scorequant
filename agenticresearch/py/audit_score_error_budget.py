"""Independent O8 audit: exact falsification, exhaustive partitions, reproduced evidence.

Only elementary rational matrix primitives are shared with the original instrument.
The audit checks projection identities independently and tests statements its random
selftest omitted. Float eigenvalue/log checks are explicitly screening, never exact proof.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from decimal import Decimal, localcontext
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import random

import score_error_budget as old

OUT = Path(__file__).resolve().parents[1] / 'WORK/artifacts/AUDIT-SCORE-ERROR-BUDGET-001'


def partitions(n, k):
    def visit(z):
        if len(z) == n:
            if max(z) + 1 == k:
                yield z
            return
        for b in range(min(k - 1, max(z) + 1) + 1):
            yield from visit(z + [b])
    yield from visit([0])


def expectation(rows, weights):
    return [sum(w * row[j] for w, row in zip(weights, rows)) for j in range(len(rows[0]))]


def cross(x, y, weights):
    return [[sum(w * a[i] * b[j] for w, a, b in zip(weights, x, y))
             for j in range(len(y[0]))] for i in range(len(x[0]))]


def conditional(rows, weights, z):
    result = []
    for b in z:
        mass = sum(w for w, a in zip(weights, z) if a == b)
        result.append([sum(w * row[j] for w, row, a in zip(weights, rows, z) if a == b) / mass
                       for j in range(len(rows[0]))])
    return result


def exact_search():
    rng = random.Random(650022)
    counts = {'laws': 0, 'partitions': 0, 'positive_retained': 0, 'projection_checks': 0,
              'log_screen_checks': 0, 'sandwich_checks': 0, 'simplex_pairs': 0}
    for d in (1, 2, 3):
        for variant in range(4):
            n = 6 if d < 3 else 7
            s = [[F(rng.randrange(-3, 4), 100 if variant == 3 and j == 0 else 1)
                  for j in range(d)] for _ in range(n)]
            if variant == 1:
                s[-1] = s[0][:]
            raw = [F(1, 1000) if variant == 2 and i == 0 else F(rng.randrange(1, 5)) for i in range(n)]
            weights = [w / sum(raw) for w in raw]
            mean = expectation(s, weights)
            s = [[x - m for x, m in zip(row, mean)] for row in s]
            v = cross(s, s, weights)
            if old.det(v) == 0:
                continue
            vi = old.inv(v)
            e = [[F(rng.randrange(-2, 3), 40) for _ in range(d)] for _ in s]
            ef = cross(e, e, weights)
            eps2 = old.trace(old.matmul(vi, ef))
            counts['laws'] += 1
            for z in partitions(n, d + 1):
                counts['partitions'] += 1
                c = conditional(s, weights, z)
                ez = conditional(e, weights, z)
                i = cross(c, c, weights)
                eb = cross(ez, ez, weights)
                assert old.is_psd(old.msub(v, i)) and old.is_psd(old.msub(ef, eb))
                assert cross(s, c, weights) == i
                counts['projection_checks'] += 3
                zh = z[:]
                zh[0] = (zh[0] + 1) % (d + 1)
                cp = conditional(s, weights, zh)
                ip = cross(cp, cp, weights)
                # Independently form cross-label centroid predictors, zero on unoccupied cells.
                for a, b, ca, ia, ib in ((z, zh, c, i, ip), (zh, z, cp, ip, i)):
                    lookup = {label: row for label, row in zip(a, ca)}
                    gamma = [[F(0) for _ in range(d)] for _ in range(d)]
                    for row, w, za, zb in zip(s, weights, a, b):
                        if za != zb:
                            g = lookup.get(zb, [F(0)] * d)
                            gamma = old.madd(gamma, old.mscale(old.madd(old.outer(row, row), old.outer(g, g)), 2*w))
                    assert old.is_psd(old.madd(old.msub(ib, ia), gamma))
                    counts['sandwich_checks'] += 1
                if old.det(i) == 0:
                    continue
                counts['positive_retained'] += 1
                ii = old.inv(i)
                h = old.msub(ii, vi)
                assert old.is_psd(h)
                between = sum(w * old.dot(a, old.matvec(h, b)) for w, a, b in zip(weights, ez, c))
                residual_e = [[a-b for a,b in zip(x,y)] for x,y in zip(e,ez)]
                residual_s = [[a-b for a,b in zip(x,y)] for x,y in zip(s,c)]
                within = sum(w * old.dot(a, old.matvec(vi,b)) for w,a,b in zip(weights,residual_e,residual_s))
                loss = d-old.trace(old.matmul(vi,i))
                ez2 = old.trace(old.matmul(vi,eb))
                # Stronger directional projection core, with no floating eigendecomposition.
                assert between**2 <= old.trace(old.matmul(h,eb))*loss
                assert within**2 <= (eps2-ez2)*loss
                counts['projection_checks'] += 2
                sh = [[a+b for a,b in zip(x,y)] for x,y in zip(s,e)]
                ch = conditional(sh,weights,z)
                for a, at, err in ((v,cross(sh,sh,weights),ef),(i,cross(ch,ch,weights),eb)):
                    x2 = old.trace(old.matmul(old.inv(a),err))
                    if x2 >= 1 or old.det(at) <= 0:
                        continue
                    with localcontext() as ctx:
                        ctx.prec = 60
                        dec = lambda f: Decimal(f.numerator)/Decimal(f.denominator)
                        x = dec(x2).sqrt()
                        u = 2*Decimal(d).sqrt()*x+x*x
                        ratio = old.det(at)/old.det(a)
                        tr = old.trace(old.matmul(old.inv(a),old.msub(at,a)))
                        rem = dec(ratio).ln()-dec(tr)
                        assert -u*u/(2*(1-x)**2)-Decimal('1e-50') <= rem <= Decimal('1e-50')
                    counts['log_screen_checks'] += 1
    # Include simplex vertices and priors/fractions close to the boundary.
    for m in (2,3,4):
        for trial in range(60):
            normal = lambda xs: [F(x)/sum(xs) for x in xs]
            prior = normal([1]+[rng.randrange(2,100) for _ in range(m-1)])
            theta = normal([rng.randrange(1,50) for _ in range(m)])
            eta = [F(int(j == trial % m)) for j in range(m)]
            hat = normal([rng.randrange(0,10)+int(j==0) for j in range(m)])
            phi = old.mixture_score_map(eta,theta,prior)
            ph = old.mixture_score_map(hat,theta,prior)
            dm = min(t/p for t,p in zip(theta,prior)); q = max(t/p for t,p in zip(theta,prior))
            for j in range(m):
                assert abs(ph[j]-phi[j]) <= (abs(hat[j]-eta[j])/prior[j]+sum(t/p*abs(a-b) for t,p,a,b in zip(theta,prior,eta,hat))/theta[j])/dm
                # Inverse coordinate bound: |Delta eta_j| <= Q(pi_j |Delta Phi_j| + sum pi |Delta Phi|).
                assert abs(hat[j]-eta[j]) <= q*(prior[j]*abs(ph[j]-phi[j])+sum(p*abs(a-b) for p,a,b in zip(prior,ph,phi)))
            counts['simplex_pairs'] += 1
    return counts


def boundary_fixtures():
    specs = []
    def add(name, claim, scores, weights, labels, quantities):
        specs.append({'id':name,'criterion':'D','level':'information_accounting','claim_falsified':claim,
                      'scores':scores,'weights':weights,'K':max(labels)+1,'labels_before':labels,
                      'labels_after_or_optimum':None,'poi_indices':list(range(len(scores[0]))),'nuisance_indices':[],
                      'objective_before':None,'objective_after':None,'exact_quantities':quantities,
                      'verification':{'method':'exact_formula','notes':'Rational witness; independently recomputed in tests/test_research_claims.py. See audit for the corrected theorem.'},
                      'source':'AUDITS/AUDIT-SCORE-ERROR-BUDGET-001.md','audit':'AUDITS/AUDIT-SCORE-ERROR-BUDGET-001.md','date':'2026-09-08'})
    add('CE-SCORE-ERROR-LOG-LOWER-001','Lemma 4 scalar inequality with positive lambda_min; the final epsilon-based curvature bound survives.',
        [['-2'],['2']],['1/2']*2,[0,1],{'proxy_scores':[['-3'],['3']],'lambda':'5/4','claimed_log_lower':'65/72','eps2':'1/4'})
    # Minimal N for centered d=2, K=3 and positive within-cell loss.
    s = [['-1','-9'],['1','-9'],['-1','1'],['1','1']]
    z = [0,1,2,2]
    add('CE-SCORE-ERROR-ALIGNMENT-ORDER-001','The trace-loss alignment bound is always <= the crude bound; both separately bound |T1|.',s,['1/20','1/20','9/20','9/20'],z,
        {'errors':[['0',str(F(row[1])/10)] for row in s],'rho_min':'1/10','eps2':'1/100','eps_R2':'1/100','trace_bound_squared':'81/1000','crude_bound_squared':'2/25','T1':'0'})
    add('CE-SCORE-ERROR-TRANSLATION-001','Uncentred determinant retention is invariant under affine translations.',[['-1'],['0'],['1']],['1/3']*3,[0,0,1],
        {'proxy_scores':[['0'],['1'],['2']],'eta_before':'3/4','eta_after':'9/10'})
    add('CE-SCORE-ERROR-SINGULAR-LS-001','The least-squares map A* is always an admissible nonsingular reporting reparameterization.',[['-1'],['1']],['1/2']*2,[0,0],
        {'proxy_scores':[['1'],['1']],'A_star':'0','V':'1','V_tilde':'1','eps_linear2':'1','transformed_V':'0'})
    add('CE-CLASSIFIER-CALIBRATION-CHART-001','Positive classifier reliability lower-bounds score error for an arbitrary fixed linear score chart.',[['1'],['-1']],['1/2']*2,[0,1],
        {'prior':['1/3']*3,'theta0':['1/3']*3,'chart':[['1','-1','0']],
         'posterior':[['1/2','1/6','1/3'],['1/6','1/2','1/3']],
         'proxy_posterior':[['7/12','1/4','1/6'],['1/4','7/12','1/6']],
         'reliability':'1/24','eps2':'0','eta_true':'1','eta_reported':'1'})
    add('CE-CLASSIFIER-CALIBRATION-RETENTION-001','Bad calibration necessarily distorts reported retention, even with a full identifiable fraction chart.',[['1'],['-1']],['1/2']*2,[0,1],
        {'prior':['1/2']*2,'theta0':['1/2']*2,'chart':[['1','-1']],
         'posterior':[['3/4','1/4'],['1/4','3/4']],
         'proxy_posterior':[['5/8','3/8'],['3/8','5/8']],
         'reliability':'1/32','eps2':'1/4','eta_true':'1','eta_reported':'1'})
    for record in specs:
        path = old.FIXTURES/(record['id']+'.json')
        path.write_text(json.dumps(record,indent=2)+'\n')
    return [r['id'] for r in specs]


def compare_measurements(reference, replay):
    """Check every recorded numeric field; provenance is intentionally different."""
    deltas = []
    def visit(a, b):
        if isinstance(a, dict):
            for key, value in a.items():
                if key != 'provenance':
                    visit(value, b[key])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                visit(x, y)
        elif isinstance(a, float):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), (a, b)
            deltas.append(abs(a-b))
        else:
            assert a == b, (a, b)
    visit(reference, replay)
    return {'numeric_fields':len(deltas),'max_absolute_difference':max(deltas, default=0)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['exact','fixtures','replicate'])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='exact':
        record={'provenance':old.provenance('independent-audit',{'seed':650022}), 'counts':exact_search(),
                'limitations':'12 finite laws, exhaustive K=d+1 partitions, N=6/7; not all N<=10 laws. Rational projection/sandwich/simplex checks; 60-digit Decimal log screening, not a proof.'}
    elif args.mode=='fixtures':
        record={'fixtures':boundary_fixtures()}
    else:
        record={'provenance':old.provenance('audit-replication',{}),'selftest':old.stage_selftest(),
                'door3':old.stage_door3(),'synthetic2d':old.stage_synthetic2d()}
        assert not record['selftest']['failures']
        assert all(r['within_bracket'] for r in record['synthetic2d']['rows'])
        record['comparison'] = {mode:compare_measurements(json.loads((old.ARTIFACTS/(mode+'.json')).read_text()), record[mode]) for mode in ('door3','synthetic2d')}
    record['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/(args.mode+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record.get('counts',{'mode':args.mode,'artifact':str(OUT/(args.mode+'.json'))})))

if __name__=='__main__':
    main()
