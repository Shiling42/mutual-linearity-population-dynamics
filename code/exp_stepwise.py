"""
Experiment for Fig. 2 and for the statistics of Sec. S4: piecewise mutual linearity across changes of composition.

(a) The example of Fig. 2: six-species community (seed 28, gamma = 1.1) in which A_23 is changed by theta in
    [-0.6, 3.0]; the saturated equilibrium at 361 values of theta, the branches of fixed composition, and the
    direction after each extinction compared with Eq. (7) of the paper.
(b) 120 random communities (60 with n = 5, 60 with n = 8, gamma = 1): one random coefficient A_kl is changed by
    theta in [-3, 3] in 121 steps. For the sweeps that have a saturated equilibrium at every step: the number of
    changes of composition per sweep, and the direction after every loss of a single species compared with Eq. (7).
    The sweeps without a saturated equilibrium over the whole range are analysed in exp_divergence_stats.py, which
    draws the same communities.

Run:  python3 -B code/exp_stepwise.py      (about 25 seconds)
Output: data/exp_stepwise_example.npz, data/exp_stepwise_stats.npz
"""
import os
import numpy as np
from glv_tools import make_community, sweep, segments, kink_prediction
from ml_core import numerical_rank

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
os.makedirs(OUT, exist_ok=True)


def analyse_sweep(r, A, k, l, thetas, multiplicity=None):
    n = len(r)
    mod = lambda th: (r, A + th * np.outer(np.eye(n)[k], np.eye(n)[l]))
    X, SUPP = sweep(r, A, mod, thetas, multiplicity=multiplicity)
    segs = segments(SUPP)
    return X, SUPP, segs


if __name__ == '__main__':
    # (a) the example
    n = 6; k, l = 1, 2
    r, A = make_community(n, seed=28, sigma=1.1)
    thetas = np.linspace(-0.6, 3.0, 361)
    mult_ex = []
    X, SUPP, segs = analyse_sweep(r, A, k, l, thetas, multiplicity=mult_ex)
    print("example: compositions along the sweep (1 = present):")
    sv3 = []
    for (s0, s1) in segs:
        idx = list(range(s0, s1))
        Xs = np.vstack([X[:, idx], np.ones(len(idx))])
        rk, sv = numerical_rank(Xs, rtol=1e-9)
        sv3.append(sv[2] / sv[0])
        print(f"   theta in [{thetas[s0]:.3f},{thetas[s1-1]:.3f}]  species present {SUPP[s0]}  rank of (x,1) = {rk}  "
              f"third singular value / first = {sv[2]/sv[0]:.1e}")
    # kink check in the example
    for q in range(1, len(segs)):
        (p0, p1), (c0, c1) = segs[q - 1], segs[q]
        Sb = [i for i in range(n) if SUPP[p0][i]]; Sa = [i for i in range(n) if SUPP[c0][i]]
        lost = set(Sb) - set(Sa); gained = set(Sa) - set(Sb)
        d_obs = X[:, c1 - 1] - X[:, c0]
        if lost and not gained and k in Sa and l in Sa:
            e = lost.pop(); d_pred = kink_prediction(A, k, Sb, e)
            cos = abs(d_obs @ d_pred) / np.linalg.norm(d_obs) / np.linalg.norm(d_pred)
            print(f"   change at theta={thetas[c0]:.3f}: species {e+1} lost; direction after the kink, Eq. (7) against "
                  f"observed: 1 - |cos| = {1 - cos:.1e}")
        elif gained:
            print(f"   change at theta={thetas[c0]:.3f}: species {[g+1 for g in gained]} invade")
    print(f"   grid points with more than one saturated equilibrium: {int(np.sum(np.array(mult_ex) > 1))} of {len(mult_ex)}")
    np.savez(os.path.join(OUT, 'exp_stepwise_example.npz'), X=X, thetas=thetas, SUPP=np.array(SUPP), r=r, A=A, k=k, l=l,
             mult=np.array(mult_ex), sv3=np.array(sv3))      # mult, sv3: added for verify_sm_numbers.py

    # (b) statistics over random communities
    rng = np.random.default_rng(0)
    nkinks, cosines, nsweeps = [], [], 0
    no_effect = 0
    single_loss_len = []          # number of grid points of the branch after every loss of a single species
    mult_all = []                 # number of saturated equilibria at every grid point of all 120 sweeps
    sizes = []                    # number of species of each of the 120 communities
    GAMMA = 1.0
    for nn in [5, 8]:
        for trial in range(60):
            seed = int(rng.integers(1e9))
            rr, AA = make_community(nn, seed=seed, sigma=GAMMA)
            kk, ll = rng.choice(nn, 2, replace=False)
            ths = np.linspace(-3, 3, 121)
            sizes.append(nn)
            Xr, SUPPr, segr = analyse_sweep(rr, AA, kk, ll, ths, multiplicity=mult_all)
            if np.any(np.isnan(Xr)):
                continue
            nsweeps += 1
            nkinks.append(len(segr) - 1)
            for q in range(1, len(segr)):
                (p0, p1), (c0, c1) = segr[q - 1], segr[q]
                Sb = [i for i in range(nn) if SUPPr[p0][i]]; Sa = [i for i in range(nn) if SUPPr[c0][i]]
                lost = set(Sb) - set(Sa); gained = set(Sa) - set(Sb)
                if len(lost) == 1 and not gained and kk in Sa and ll in Sa:
                    single_loss_len.append(c1 - c0)
                # the direction after the kink is measured between the first and the last grid point of the
                # following branch, which must contain at least three grid points
                if len(lost) == 1 and not gained and kk in Sa and ll in Sa and c1 - c0 >= 3:
                    e = lost.pop(); d_pred = kink_prediction(AA, kk, Sb, e)
                    d_obs = Xr[:, c1 - 1] - Xr[:, c0]
                    cosines.append(abs(d_obs @ d_pred) / np.linalg.norm(d_obs) / np.linalg.norm(d_pred))
                if (kk not in Sa or ll not in Sa):
                    no_effect += 1
    nkinks = np.array(nkinks); mult_all = np.array(mult_all); single_loss_len = np.array(single_loss_len)
    print(f"\nrandom communities (n=5 and 8): 120 sweeps of theta in [-3,3], of which {nsweeps} have a saturated "
          f"equilibrium at every step")
    print(f"   changes of composition per sweep in these {nsweeps}: mean {nkinks.mean():.2f}, max {nkinks.max()}, "
          f"fraction with >=1: {np.mean(nkinks>=1):.3f}, with >=2: {np.mean(nkinks>=2):.3f}")
    print(f"   losses of a single species with k and l still present: {len(single_loss_len)}, of which "
          f"{len(cosines)} are followed by a branch of at least three grid points")
    print(f"   direction after these {len(cosines)} kinks, Eq. (7) against observed: largest 1 - |cos| = "
          f"{1 - min(cosines):.1e}")
    print(f"   branches on which the control has no effect (k or l absent): {no_effect}")
    print(f"   grid points of all 120 sweeps: {len(mult_all)}; with one saturated equilibrium: "
          f"{int(np.sum(mult_all == 1))}; with none: {int(np.sum(mult_all == 0))}; with more than one: "
          f"{int(np.sum(mult_all > 1))}")
    np.savez(os.path.join(OUT, 'exp_stepwise_stats.npz'), nkinks=nkinks, cosines=np.array(cosines),
             # added for verify_sm_numbers.py: numbers of saturated equilibria, lengths of the branches after a
             # loss, and the parameters of the sweeps
             mult=mult_all, single_loss_len=single_loss_len, sizes=np.array(sizes), gamma=GAMMA, thetas=ths)
    print('saved data/exp_stepwise_example.npz and data/exp_stepwise_stats.npz')
