"""
Experiment for Sec. S2 ("Equilibria along a sweep") and Sec. S4 ("Extinction versus divergence"): the same 120 random
sweeps as exp_stepwise.py (same random stream, so the same communities and the same coefficients A_kl).

1. Check of the saturated equilibrium against direct integration. At five random values of theta per sweep (600
   settings) the GLV equations are integrated from a random initial condition, and the set of surviving species is
   compared with the composition of the saturated equilibrium.
2. Sweeps without a saturated equilibrium over the whole range. On the last branch before the gap, with present
   species S and k, l in S, the effective press is s = (theta - theta0) x_l / (1 - (theta - theta0) B_lk), which
   diverges at the pole theta0 + 1/B_lk. Sign criterion: if every entry of b_k on S is zero or has the sign of B_lk,
   every abundance grows toward the pole and the equilibrium diverges there. A sweep that fails the criterion on its
   last grid branch is resolved inside the last grid step.
3. Branches on which k and l are present and which end inside the range of theta: how they end.

Sign convention: Bc below is A_SS^{-1}, which is -B of the paper. The numbers that are printed and stored use the
paper's B = -A_SS^{-1} and b_k (its k-th column).

Run:  python3 -B code/exp_divergence_stats.py      (about 25 seconds).   Output: data/exp_divergence_stats.npz
"""
import os
import numpy as np
from glv_tools import make_community, sweep, segments
from ml_core import glv_saturated_equilibrium, glv_attractor

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
os.makedirs(OUT, exist_ok=True)


def resolve_last_step(rr, AA, kk, ll, th0, th1, x_th0, S):
    """A sweep has a saturated equilibrium at the grid value th0 (present species S) and none at the next one, th1,
    but its last grid branch fails the sign criterion. Follow the branch from th0: find the species e that reaches
    zero before the pole and where, then the pole of the branch without e and the sign criterion on that branch.
    Returns a dict, or None if the branch from th0 does not run toward its pole."""
    nn = len(rr)
    At = AA.copy(); At[kk, ll] += th0
    Bp = -np.linalg.inv(At[np.ix_(S, S)]); ik, il = S.index(kk), S.index(ll)
    b, Blk, xS = Bp[:, ik], Bp[il, ik], x_th0[S]
    if Blk <= 0:
        return None
    s_zero = [(-xS[i] / b[i], S[i]) for i in range(len(S)) if b[i] < 0]      # s at which species S[i] reaches zero
    if not s_zero:
        return None
    s_e, e = min(s_zero)
    th_e = th0 + s_e / (xS[il] + s_e * Blk)                                   # inverts s(theta); lies before the pole
    S2 = [i for i in S if i != e]
    At2 = AA.copy(); At2[kk, ll] += th_e
    Bp2 = -np.linalg.inv(At2[np.ix_(S2, S2)]); ik2, il2 = S2.index(kk), S2.index(ll)
    pole2 = th_e + 1.0 / Bp2[il2, ik2]
    one_signed2 = bool(np.all(Bp2[:, ik2] * np.sign(Bp2[il2, ik2]) >= 0))
    # check by enumeration: half-way between the extinction and the new pole the saturated equilibrium is the one
    # without e; between the new pole and th1 there is none
    mod = lambda th: (rr, AA + th * np.outer(np.eye(nn)[kk], np.eye(nn)[ll]))
    x_mid, alive_mid = glv_saturated_equilibrium(*mod(0.5 * (th_e + pole2)))
    mid_ok = x_mid is not None and [i for i in range(nn) if alive_mid[i]] == S2
    x_after, _ = glv_saturated_equilibrium(*mod(0.5 * (pole2 + th1))) if pole2 < th1 else (0, 0)
    return dict(e=e, th_e=th_e, pole_first=th0 + 1.0 / Blk, pole2=pole2, one_signed2=one_signed2,
                mid_ok=bool(mid_ok), none_after=x_after is None, in_step=bool(th0 < th_e < pole2 <= th1))


if __name__ == '__main__':
    rng = np.random.default_rng(0); rng_ic = np.random.default_rng(12345)   # same community stream as exp_stepwise.py
    n_sweeps = 0; n_div = 0; n_complete = 0
    br_complete = 0; br_incomplete_comp = 0; br_incomplete_div = 0           # branches (k, l present) ending inside
    sign_ok = 0; sign_total = 0; pole_in_step = 0; gap_to_right_end = 0
    checks = 0; agree = 0; max_err_all = 0.0
    exceptions = []; resolved = []
    for nn in [5, 8]:
        for trial in range(60):
            seed = int(rng.integers(1e9))
            rr, AA = make_community(nn, seed=seed, sigma=1.0)
            kk, ll = rng.choice(nn, 2, replace=False)
            ths = np.linspace(-3, 3, 121)
            mod = lambda th: (rr, AA + th * np.outer(np.eye(nn)[kk], np.eye(nn)[ll]))
            X, SUPP = sweep(rr, AA, mod, ths)
            n_sweeps += 1
            nan = np.isnan(X[0])
            segs = segments(SUPP)
            # 1. direct integration at 5 random settings that have a saturated equilibrium
            good = np.where(~nan)[0]
            for q in rng_ic.choice(good, size=min(5, len(good)), replace=False):
                rt, At = mod(ths[q])
                x0 = rng_ic.uniform(0.05, 2.0, size=nn)
                xo, alive = glv_attractor(rt, At, x0, t_max=10000)
                err = np.max(np.abs(xo - X[:, q])); checks += 1
                sat = np.array(SUPP[q], bool)
                max_err_all = max(max_err_all, err)
                if np.all(alive == sat):
                    agree += 1
                else:
                    missing = [i for i in range(nn) if sat[i] and not alive[i]]
                    extra = [i for i in range(nn) if alive[i] and not sat[i]]
                    x_missing = max([X[i, q] for i in missing]) if missing else np.nan
                    # the nearest change of composition along the sweep, and the species that change there
                    changes = [c0 for (c0, c1) in segs[1:]]              # first grid index of each new composition
                    steps, at_extinction = -1, 0
                    if changes:
                        dist = [c0 - q if q < c0 else q - c0 + 1 for c0 in changes]
                        steps = min(dist); c0 = changes[int(np.argmin(dist))]
                        changed = [i for i in range(nn) if SUPP[c0][i] != SUPP[c0 - 1][i]]
                        at_extinction = int(bool(missing) and not extra and missing == changed)
                    exceptions.append([nn, seed, kk, ll, ths[q], steps, len(missing), len(extra), x_missing, err,
                                       at_extinction])
                    print(f"   composition differs: n={nn}, seed={seed}, A_{kk+1}{ll+1}, theta={ths[q]:.2f}: the "
                          f"saturated equilibrium has species {[i+1 for i in missing]} (abundance {x_missing:.2e}) "
                          f"that integration ends without; species present only after integration: "
                          f"{[i+1 for i in extra]}; nearest change of composition: {steps} grid step(s) away"
                          + (", where exactly these species go extinct" if at_extinction else ""), flush=True)
            # 2. sweeps without a saturated equilibrium over the whole range
            if nan.any():
                n_div += 1
                first_nan = int(np.argmax(nan))
                gap_to_right_end += int(nan[first_nan:].all())
                # the branch just before the first gap: does it run into its pole with all abundances growing?
                if first_nan > 0 and not nan[first_nan - 1]:
                    S = [i for i in range(nn) if SUPP[first_nan - 1][i]]
                    if kk in S and ll in S:
                        th0 = ths[first_nan - 1]; At = AA.copy(); At[kk, ll] += th0
                        Bc = np.linalg.inv(At[np.ix_(S, S)]); ik, il = S.index(kk), S.index(ll)
                        pole = th0 - 1.0 / Bc[il, ik]                    # = th0 + 1/B_lk in the paper's convention
                        one_signed = np.all(Bc[:, ik] <= 0)              # every entry of b_k = -Bc[:, ik] is >= 0
                        sign_total += 1; sign_ok += int(one_signed and pole > th0)
                        if one_signed and pole > th0:
                            pole_in_step += int(th0 < pole <= ths[first_nan])
                        else:
                            res = resolve_last_step(rr, AA, kk, ll, th0, ths[first_nan], X[:, first_nan - 1], S)
                            if res is None:
                                print(f"   not resolved: n={nn}, seed={seed}, A_{kk+1}{ll+1}, last grid value {th0:.2f}")
                                resolved.append([nn, seed, kk, ll, th0, ths[first_nan]] + [np.nan] * 8)
                            else:
                                resolved.append([nn, seed, kk, ll, th0, ths[first_nan], res['e'], res['th_e'],
                                                 res['pole_first'], res['pole2'], res['one_signed2'], res['mid_ok'],
                                                 res['none_after'], res['in_step']])
                                print(f"   sweep that fails the sign criterion on its last grid branch: n={nn}, "
                                      f"seed={seed}, A_{kk+1}{ll+1}. Between the grid values {th0:.2f} and "
                                      f"{ths[first_nan]:.2f}: species {res['e']+1} reaches zero at theta = "
                                      f"{res['th_e']:.4f} (pole of that branch: {res['pole_first']:.4f}); on the branch "
                                      f"without it the sign criterion holds: {res['one_signed2']}, and its pole is at "
                                      f"theta = {res['pole2']:.4f}. Enumeration: saturated equilibrium without species "
                                      f"{res['e']+1} half-way to that pole: {res['mid_ok']}; none beyond it: "
                                      f"{res['none_after']}", flush=True)
            else:
                n_complete += 1
            # 3. branches with k, l present that end inside the range of theta
            for (a, b) in segs:
                if nan[a]:
                    continue
                S = [i for i in range(nn) if SUPP[a][i]]
                if kk not in S or ll not in S:
                    continue
                if b < len(ths) and nan[b]:
                    br_incomplete_div += 1
                elif b < len(ths):
                    if nan.any():
                        br_incomplete_comp += 1
                    else:
                        br_complete += 1
    exceptions = np.array(exceptions, float).reshape(-1, 11); resolved = np.array(resolved, float).reshape(-1, 14)
    n_resolved = int(np.sum((resolved[:, 10] == 1) & (resolved[:, 11] == 1) & (resolved[:, 12] == 1)
                            & (resolved[:, 13] == 1))) if len(resolved) else 0
    print(f"sweeps: {n_sweeps}; with a saturated equilibrium at every theta: {n_complete}; without one beyond some "
          f"theta: {n_div} (in {gap_to_right_end} of them from that theta to the end of the range)")
    print(f"sign criterion on the last branch of these sweeps ({sign_total} have k, l present there): every entry of "
          f"b_k has the sign of B_lk and the pole is ahead in {sign_ok}; in {pole_in_step} of them the pole lies within "
          f"the next grid step. Sweeps that fail the criterion: {len(resolved)}; of these, with an extinction and then "
          f"the pole within the next grid step: {n_resolved}")
    print(f"branches with k, l present that end inside the range: in the {n_complete} complete sweeps {br_complete}, all "
          f"by a change of composition; in the other {n_div} sweeps {br_incomplete_comp} by a change of composition and "
          f"{br_incomplete_div} by divergence at the pole  (all 120 sweeps: {br_complete + br_incomplete_comp} and "
          f"{br_incomplete_div})")
    print(f"direct integration: {checks} settings; the surviving species are those of the saturated equilibrium at "
          f"{agree} (where they are, the abundances agree by construction: both solve the same linear equations)")
    np.savez(os.path.join(OUT, 'exp_divergence_stats.npz'), n_sweeps=n_sweeps, n_complete=n_complete, n_div=n_div,
             gap_to_right_end=gap_to_right_end, sign_total=sign_total, sign_ok=sign_ok, pole_in_step=pole_in_step,
             br_complete=br_complete, br_incomplete_comp=br_incomplete_comp, br_incomplete_div=br_incomplete_div,
             checks=checks, agree=agree, max_err_all=max_err_all,
             # one row per setting at which integration and enumeration differ:
             # n, seed, k, l, theta, grid steps to the nearest change of composition, number of species of the
             # saturated equilibrium that integration ends without, number present only after integration, largest
             # saturated abundance of the missing species, max |x_integrated - x_saturated|, 1 if the missing species
             # are exactly those that go extinct at that nearest change of composition
             exceptions=exceptions,
             # one row per sweep that fails the sign criterion on its last grid branch:
             # n, seed, k, l, last grid value with a saturated equilibrium, next grid value, species that reaches zero,
             # theta of that extinction, pole of the last grid branch, pole of the branch after the extinction, sign
             # criterion on that branch, enumeration agrees before the pole, no saturated equilibrium after it, both
             # events within the grid step
             resolved=resolved)
    print('saved data/exp_divergence_stats.npz')
