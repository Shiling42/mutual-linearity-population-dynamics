"""
Experiments for Fig. S2 (uses of the lines) and Fig. S4 (several modified species and other models).

Uses, on the community and the sweep of Fig. 2 (read from data/exp_stepwise_example.npz, so run exp_stepwise.py first):
  (4a) prediction of all abundances on a branch from two measurements;
  (4b) the changes of composition, seen as kinks in the relation between two monitored abundances;
  (4c) the range of the effective press on the branch with all species, and the pole of s(theta);
  (4d) slopes of Eq. (5) from the sweep of A_23 against ratios of press responses from an actual press on species 2.
Scope:
  (5a) seven-species community (seed 11, gamma = 0.9): 300 random simultaneous modifications of the intrinsic growth
       rate and of all interaction coefficients of 1, 2 or 3 species; the draws that keep all seven species give
       points (x*, 1) of rank |K| + 1;
  (5b) MacArthur consumer-resource model, 4 resources and 3 consumers: mortality of consumer 1 (one equation), and
       the consumption rate u_11 (shared by two equations);
  (5c) replicator equation with 4 strategies: one payoff entry.

Sign convention: B below is A^{-1} and dk its k-th column, which is -b_k in the paper; s is the paper's effective
press, x* = x0* + s b_k = x0* - s dk.

Run:  python3 -B code/exp_use_and_scope.py      (about 2 seconds).   Output: data/exp_use.npz
"""
import os
import numpy as np
from glv_tools import make_community, interior, segments
from ml_core import numerical_rank, glv_saturated_equilibrium

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))

if __name__ == '__main__':
    n = 6; k, l = 1, 2
    r, A = make_community(n, seed=28, sigma=1.1)
    d = np.load(os.path.join(OUT, 'exp_stepwise_example.npz'))
    X, thetas, SUPP = d['X'], d['thetas'], [tuple(s) for s in d['SUPP']]
    segs = segments(SUPP)
    # ---- (4a) prediction from two measurements on the longest branch (species 4 absent) ----
    br = max(segs, key=lambda s: s[1] - s[0]); s0, s1 = br
    i1, i2 = s0 + 5, s0 + 25                       # two 'measured' control settings
    P1, P2 = X[:, i1], X[:, i2]
    dirn = P2 - P1
    # predict every species from species 1 alone: x_i = P1_i + (x_1 - P1_1) * dirn_i / dirn_1
    pred = P1[:, None] + np.outer(dirn / dirn[0], X[0, s0:s1] - P1[0])
    err = np.max(np.abs(pred - X[:, s0:s1]))
    print(f"(4a) branch theta in [{thetas[s0]:.2f},{thetas[s1-1]:.2f}], two measurements at theta={thetas[i1]:.2f},{thetas[i2]:.2f}: "
          f"max prediction error for all species over the whole branch = {err:.1e}")
    # ---- (4b) hidden changes of composition: the grid values at which the composition changes ----
    kinks = [thetas[c0] for (c0, c1) in segs[1:]]
    print(f"(4b) kinks in the relation between two monitored abundances at theta = {np.round(kinks, 3)} (species that change: "
          f"{[ [j+1 for j in range(n) if SUPP[c0][j] != SUPP[c0-1][j]] for (c0,c1) in segs[1:]]})")
    # ---- (4c) range of the effective press on the branch with all species ----
    full = [q for q in range(len(thetas)) if SUPP[q] == tuple([1] * n)]
    x0 = interior(r, A); B = np.linalg.inv(A); dk = B[:, k]
    # along x(s) = x0 - s dk, species i reaches zero at s_i = x0_i / dk_i (positive and negative sides)
    s_th = x0 / dk
    s_plus = min([s for s in s_th if s > 0]); s_minus = max([s for s in s_th if s < 0])
    seg_ends = (x0 - s_minus * dk, x0 - s_plus * dk)
    print(f"(4c) branch with all species: s in ({s_minus:.3f},{s_plus:.3f}); first extinction at s<0: species {np.argmax(s_th==s_minus)+1}, "
          f"at s>0: species {np.argmax(s_th==s_plus)+1}; attainable range of x_1: [{min(seg_ends[0][0],seg_ends[1][0]):.3f},{max(seg_ends[0][0],seg_ends[1][0]):.3f}]")
    thp = -1.0 / B[l, k]
    print(f"    pole of s(theta) at theta = {thp:.3f}; theta at the ends of the branch = [{thetas[full[0]]:.3f},{thetas[full[-1]]:.3f}]")
    # ---- (4d) slopes = ratios of press responses ----
    slopes_int = dirn[1:] / dirn[0]
    # on the branch (species 4 absent), compare with an actual press on species k in the reduced community
    S = [j for j in range(n) if SUPP[s0][j]]
    rS, AS = r[S], A[np.ix_(S, S)]
    xS0 = interior(rS, AS); xSp = interior(rS + 0.05 * np.eye(len(S))[S.index(k)], AS)
    dp = np.zeros(n); dp[S] = xSp - xS0
    print(f"(4d) slopes dx_i/dx_1 from the sweep of A_{k+1}{l+1}: {np.round(slopes_int,4)}; from an actual press on species {k+1}: {np.round(dp[1:]/dp[0],4)}"
          f"; largest difference {np.max(np.abs(slopes_int - dp[1:]/dp[0])):.1e}")

    # ---- (5a) several modified species -> rank |K| + 1 of (x,1) ----
    r5, A5 = make_community(7, seed=11, sigma=0.9)
    rng = np.random.default_rng(5)
    n_draws = 300
    dim_kept, dim_rank, dim_sv, dim_mult = [], [], [], []
    for K in [[0], [0, 3], [0, 3, 5]]:
        pts = []; mult = []
        for t in range(n_draws):
            rt, At = r5.copy(), A5.copy()
            for kk in K:
                rt[kk] += rng.uniform(-0.2, 0.2); At[kk] += rng.uniform(-0.25, 0.25, size=7)
            xs, alive = glv_saturated_equilibrium(rt, At, multiplicity=mult)
            if xs is not None and alive.all():
                pts.append(xs)
        P = np.array(pts).T
        rk, sv = numerical_rank(np.vstack([P, np.ones(P.shape[1])]), rtol=1e-10)
        dim_kept.append(P.shape[1]); dim_rank.append(rk); dim_sv.append(sv / sv[0]); dim_mult.append(mult)
        print(f"(5a) equations of species {[q+1 for q in K]} modified at random: {P.shape[1]} of the {n_draws} draws keep all seven "
              f"species; rank of (x,1) = {rk} = |K|+1; singular values = {np.array2string(sv[:5]/sv[0], precision=1)}; "
              f"draws with more than one saturated equilibrium: {int(np.sum(np.array(mult) > 1))}")
    # ---- (5b) MacArthur consumer-resource: dR_j/dt = R_j(kappa_j - c_j R_j - sum_i u_ij N_i), dN_i/dt = N_i(eps_i sum_j u_ij R_j - m_i) ----
    m_res, n_con = 4, 3
    rng = np.random.default_rng(2)
    while True:
        U = rng.uniform(0.2, 1.0, size=(n_con, m_res)); eps = rng.uniform(0.3, 0.8, size=n_con)
        s_ = rng.uniform(1.0, 2.0, size=m_res); c_ = rng.uniform(0.5, 1.0, size=m_res); m_ = rng.uniform(0.3, 0.6, size=n_con)
        # GLV form on y=(R,N): r=(kappa, -m), A = [[-diag(c), -U^T],[diag(eps) U, 0]]   (s_ is kappa)
        rr = np.concatenate([s_, -m_]); AA = np.block([[-np.diag(c_), -U.T], [np.diag(eps) @ U, np.zeros((n_con, n_con))]])
        y = np.linalg.solve(AA, -rr)
        if np.all(y > 0.02) and np.all(np.linalg.eigvals(np.diag(y) @ AA).real < 0):
            break
    N = m_res + n_con
    # mortality of consumer 1 (the equation of N_1 only): line
    pts = []; mult_cr = []
    for th in np.linspace(-0.15, 0.15, 31):
        rt = rr.copy(); rt[m_res] -= th
        ys, alive = glv_saturated_equilibrium(rt, AA, multiplicity=mult_cr)
        if ys is not None and alive.all(): pts.append(ys)
    P = np.array(pts).T; rk, sv = numerical_rank(np.vstack([P, np.ones(P.shape[1])]), rtol=1e-10)
    print(f"(5b) MacArthur consumer-resource ({m_res} resources, {n_con} consumers), mortality of consumer 1 varied "
          f"({P.shape[1]} of 31 settings keep all species): rank of (y,1) = {rk}, singular values = {np.array2string(sv[:4]/sv[0], precision=1)}")
    # consumption rate u_11: enters the equation of consumer 1 (eps u R) AND that of resource 1 (-u N): two equations -> plane
    pts = []
    for th in np.linspace(-0.15, 0.3, 31):
        At = AA.copy(); At[0, m_res] -= th; At[m_res, 0] += eps[0] * th
        ys, alive = glv_saturated_equilibrium(rr, At, multiplicity=mult_cr)
        if ys is not None and alive.all(): pts.append(ys)
    Pcr2 = np.array(pts).T; rk2, sv2 = numerical_rank(np.vstack([Pcr2, np.ones(Pcr2.shape[1])]), rtol=1e-10)
    print(f"     consumption rate u_11 varied (shared by two equations; {Pcr2.shape[1]} of 31 settings keep all species): "
          f"rank of (y,1) = {rk2}, singular values = {np.array2string(sv2[:4]/sv2[0], precision=1)}")
    # ---- (5c) replicator equation: interior rest point under one payoff entry ----
    rng = np.random.default_rng(1)
    while True:
        Pay = rng.uniform(0, 1, size=(4, 4)); xe = np.linalg.solve(Pay, np.ones(4))
        if np.all(xe > 0): break
    Xr = []
    for th in np.linspace(-0.6, 0.6, 61):
        Pt = Pay.copy(); Pt[0, 1] += th; xe = np.linalg.solve(Pt, np.ones(4))
        if np.all(xe > 0): Xr.append(xe / xe.sum())
    Xr = np.array(Xr).T; rk3, sv3 = numerical_rank(Xr, rtol=1e-10)
    print(f"(5c) replicator, 4 strategies, payoff entry a_12 varied ({Xr.shape[1]} of 61 settings have an interior rest point): "
          f"rank of the rest points = {rk3}, singular values = {np.array2string(sv3/sv3[0], precision=1)}")
    np.savez(os.path.join(OUT, 'exp_use.npz'), P1=P1, P2=P2, i1=i1, i2=i2, s0=s0, s1=s1, pred=pred, seg_ends=np.array(seg_ends),
             s_minus=s_minus, s_plus=s_plus, slopes_int=slopes_int, dp=dp, x0=x0, dk=dk,
             Pcr=P, P2cr=Pcr2, Xr=Xr, rr=rr, AA=AA, Pay=Pay,
             # added for verify_sm_numbers.py: the dimension counting of (5a) and the numbers of saturated equilibria
             dim_r=r5, dim_A=A5, dim_draws=n_draws, dim_kept=np.array(dim_kept), dim_rank=np.array(dim_rank),
             **{f'dim_sv_{q + 1}': dim_sv[q] for q in range(3)},
             **{f'dim_mult_{q + 1}': np.array(dim_mult[q]) for q in range(3)}, cr_mult=np.array(mult_cr))
    print('saved data/exp_use.npz')
