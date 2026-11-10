"""
Experiment for Fig. 1 and Fig. S1(a),(b): every modification of the equation of one species moves the community along
the same line.

Five-species community (seed 3, gamma = 0.9); species k = 3 is modified in four ways: a press on r_3, a change of the
self-regulation A_33, a change of the interaction A_31, and a mixed change (r_3 + 0.4 sin 2 theta, A_31 + theta^2,
A_35 - 0.5 theta). For each modification the script computes
  * the equilibria with all species present along a sweep of theta,
  * the effective press s in two ways: from the displacement along the line, and from its definition, Eq. (3) of the
    paper, s = h(x*; theta) (the change of the per-capita growth rate of species k, evaluated at the equilibrium),
  * the distance of the equilibria from the line x0* + s b_k, and the singular values of all points (x*, 1).
Second part, Fig. S1(b): the same four modifications when the effects of species 1 and 2 saturate (type II),
x_j -> x_j / (1 + 0.25 x_j) in every per-capita growth rate. The equilibria then lie on one curve, not on a line.

Sign convention: d below is column k of A^{-1}, which is -b_k in the paper (B = -A^{-1}). The stored effective press s
is the paper's: x* = x0* + s b_k = x0* - s d.

Run:  python3 -B code/exp_collinear.py      (about 1 second).   Output: data/exp_collinear.npz
"""
import os
import numpy as np
from scipy.optimize import fsolve
from glv_tools import make_community, interior, sweep, line_residual_affine
from ml_core import numerical_rank

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
os.makedirs(OUT, exist_ok=True)

if __name__ == '__main__':
    n = 5
    r, A = make_community(n, seed=3, sigma=0.9)
    x0 = interior(r, A); k, l, m = 2, 0, 4         # species k=3 (1-based) is modified
    B = np.linalg.inv(A); d = B[:, k]              # d = -b_k
    print("community: r =", np.round(r, 3), "\nA =\n", np.round(A, 3), "\nx0 =", np.round(x0, 3),
          "\ncolumn k of A^-1 (= -b_k) =", np.round(d, 3))

    press, selfreg, inter, mixed = f'press r_{k + 1}', f'self-regulation A_{k + 1}{k + 1}', \
        f'interaction A_{k + 1}{l + 1}', 'mixed'
    mods = {
        press:   lambda th: (r + th * np.eye(n)[k], A),
        selfreg: lambda th: (r, A + th * np.outer(np.eye(n)[k], np.eye(n)[k])),
        inter:   lambda th: (r, A + th * np.outer(np.eye(n)[k], np.eye(n)[l])),
        mixed:   lambda th: (r + 0.4 * np.sin(2 * th) * np.eye(n)[k],
                             A + np.outer(np.eye(n)[k], th ** 2 * np.eye(n)[l] - 0.5 * th * np.eye(n)[m])),
    }
    ranges = {press: (-0.35, 0.35), selfreg: (-0.5, 1.2), inter: (-1.0, 1.0), mixed: (-0.8, 0.8)}
    results = {}
    allX = []
    for name, mod in mods.items():
        thetas = np.linspace(*ranges[name], 81)
        mult = []
        X, SUPP = sweep(r, A, mod, thetas, multiplicity=mult)
        full = [q for q in range(len(thetas)) if SUPP[q] == tuple([1] * n)]
        X = X[:, full]; thetas = thetas[full]
        # effective press from the displacement: s such that x = x0 + s b_k = x0 - s d (least squares)
        s = -((X - x0[:, None]).T @ d) / (d @ d)
        resid = np.max(np.abs(X - x0[:, None] + np.outer(d, s)))                      # largest component
        dist = np.max(np.linalg.norm(X - x0[:, None] + np.outer(d, s), axis=0))      # largest Euclidean distance
        # effective press from its definition, Eq. (3): the change of g_k evaluated at the equilibrium
        rt_At = [mod(th) for th in thetas]
        s_identity = np.array([((rt[k] - r[k]) + (At[k] - A[k]) @ X[:, q]) for q, (rt, At) in enumerate(rt_At)])
        print(f"{name:22s}: {len(full)} equilibria with all species present (of {len(mult)} grid points; at "
              f"{int(np.sum(np.array(mult) == 0))} there is no saturated equilibrium, at "
              f"{len(mult) - len(full) - int(np.sum(np.array(mult) == 0))} a species is absent); "
              f"largest distance from the line x0 + s b_k = {dist:.1e} (largest component {resid:.1e}); "
              f"max |s - s of Eq. (3)| = {np.max(np.abs(s - s_identity)):.1e}; s range [{s.min():.3f},{s.max():.3f}]; "
              f"grid points with more than one saturated equilibrium: {int(np.sum(np.array(mult) > 1))} of {len(mult)}")
        results[name] = dict(thetas=thetas, X=X, s=s, sid=s_identity, mult=np.array(mult), resid=resid, dist=dist)
        allX.append(X)
    allX = np.hstack(allX)
    rk, sv = numerical_rank(np.vstack([allX, np.ones(allX.shape[1])]), rtol=1e-12)
    print(f"all {allX.shape[1]} equilibria of the four modifications: rank of (x, 1) = {rk}; singular values:",
          np.array2string(sv / sv[0], precision=2))

    # ---- type II: saturating effects of species 1 and 2 on every species (Fig. S1(b)) ----
    # per-capita growth g_i = r_i + sum_j A_ij f_j(x_j), f_j(x) = x/(1 + h x) for j in {0,1}, f_j = x otherwise.
    # Species k keeps the same four modifications. The equilibria are confined to the solution CURVE of the n-1
    # unmodified equations (not a line any more).
    h = 0.25
    def f(x):
        y = x.copy(); y[0] = x[0] / (1 + h * x[0]); y[1] = x[1] / (1 + h * x[1]); return y
    def solve_eq(rt, At, guess):
        sol, info, ier, msg = fsolve(lambda x: rt + At @ f(x), guess, xtol=1e-13, full_output=True)
        return sol, (ier == 1 and np.all(sol > 1e-6) and np.max(np.abs(rt + At @ f(sol))) < 1e-9)
    xn0, ok = solve_eq(r, A, x0); assert ok
    ranges2 = {press: (-0.3, 0.3), selfreg: (-0.5, 0.5), inter: (-0.8, 0.8), mixed: (-0.7, 0.7)}
    curves = {}
    for name, mod in mods.items():
        th0 = np.linspace(0, ranges2[name][1], 41); th1 = np.linspace(0, ranges2[name][0], 41)
        pts = {}
        for ths in (th0, th1):                       # continuation outwards from theta = 0 in both directions
            guess = xn0.copy()
            for th in ths:
                rt, At = mod(th); xe, ok = solve_eq(rt, At, guess)
                if not ok:
                    break
                pts[th] = xe; guess = xe
        P = np.array([pts[t] for t in sorted(pts)]).T
        curves[name] = P
    others = [i for i in range(n) if i != k]
    t2_eqres, t2_bend, t2_dev = [], [], []
    for name, P in curves.items():
        res = np.max(np.abs(np.array([r[others] + A[others] @ f(P[:, q]) for q in range(P.shape[1])])))
        bend = line_residual_affine(P)
        t2_eqres.append(res); t2_bend.append(bend)
        print(f"type II, {name:22s}: {P.shape[1]} equilibria, max residual of the unmodified equations = {res:.1e}, "
              f"deviation from a straight line (relative second singular value of the centered data) = {bend:.2e}")
    # the common curve: parametrize by x_k and check that x_i(x_k) coincides across the modifications (interpolation)
    ref = curves[press]
    order = np.argsort(ref[k]); xk_ref, Xref = ref[k][order], ref[:, order]
    for name, P in curves.items():
        if name == press:
            t2_dev.append(np.nan)
            continue
        inside = (P[k] >= xk_ref.min()) & (P[k] <= xk_ref.max())
        dev = max(np.max(np.abs(np.interp(P[k][inside], xk_ref, Xref[i]) - P[i][inside])) for i in others)
        t2_dev.append(dev)
        print(f"   {name:22s}: max deviation from the curve of the press (interpolated in x_k) = {dev:.1e} "
              f"over {inside.sum()} equilibria")
    np.savez(os.path.join(OUT, 'exp_collinear.npz'), r=r, A=A, x0=x0, d=d, k=k, l=l, m=m, h=h, xn0=xn0,
             **{f'X_{i}': results[nm]['X'] for i, nm in enumerate(mods)},
             **{f'th_{i}': results[nm]['thetas'] for i, nm in enumerate(mods)},
             **{f's_{i}': results[nm]['s'] for i, nm in enumerate(mods)},
             **{f'C_{i}': curves[nm] for i, nm in enumerate(mods)}, names=np.array(list(mods)),
             # added for verify_sm_numbers.py: s from Eq. (3), distance from the line, numbers of saturated
             # equilibria, type-II summary
             resid=np.array([results[nm]['resid'] for nm in mods]),
             dist=np.array([results[nm]['dist'] for nm in mods]),
             **{f'sid_{i}': results[nm]['sid'] for i, nm in enumerate(mods)},
             **{f'mult_{i}': results[nm]['mult'] for i, nm in enumerate(mods)},
             t2_eqres=np.array(t2_eqres), t2_bend=np.array(t2_bend), t2_dev=np.array(t2_dev))
    print('saved data/exp_collinear.npz')
