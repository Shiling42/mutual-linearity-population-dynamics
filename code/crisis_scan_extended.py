"""
Experiment for Sec. S6 ("Compositions and the crisis") and for the two bands of Fig. S3(c): where does the chaotic
attractor of the community of Vano et al. (2006) end under a_13 -> a_13 + theta?

1. The two-species state (algebra, printed first). The equilibrium of species 3 and 4 alone does not depend on
   theta. The script prints it, the growth rates with which species 1 and 2 would invade it, and the value of theta
   above which species 1 cannot invade. Above that value this state and the chaotic attractor can coexist.
2. The scan. For theta = 0.15, 0.16, ..., 0.30, twelve trajectories are started near the interior equilibrium (every
   abundance multiplied by 1 + 0.1 u, u uniform in [-1, 1], from the generator seeded with (2026, round(1000 theta),
   index)) and integrated for 40 000 time units (DOP853, rtol 1e-10, atol 1e-13). A trajectory has escaped to the
   two-species state when x_1 or x_2 falls below 1e-9; the time of that event is recorded.

Run:  python3 -B code/crisis_scan_extended.py [workers]      (about 1.5 minutes with 4 processes, the default)
Output: data/crisis_scan_extended.csv, one line per trajectory (theta, index of the initial condition, stayed 1/0,
time of the escape or -1), sorted by theta and index. figure_scripts/figS3_chaos.py reads the two bands of
Fig. S3(c) from this file.
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp
from glv_tools import VANO_R as r, VANO_A as a0, vano_rhs, boundary_state

warnings.filterwarnings('ignore')
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
THETAS = np.round(np.arange(0.15, 0.3001, 0.01), 3)
N_IC, T_END, K, L = 12, 40000.0, 0, 2


def two_species_state():
    """The equilibrium of species 3 and 4 alone and its invasion growth rates. Returns x (four abundances), a
    function rate(theta, i) for the invasion growth rate of species i (0-based), and the value of theta at which the
    invasion growth rate of species 1 changes sign."""
    x34, g0 = boundary_state(r, a0, [2, 3])

    def rate(theta, i):
        a = a0.copy(); a[K, L] += theta
        return boundary_state(r, a, [2, 3])[1][i]

    theta_c = g0[K] / (r[K] * x34[L])            # r_1 (1 - sum_j a_1j x_j - theta x_3) = 0
    return x34, rate, theta_c


def collapse(t, x, r_, a_):
    return min(x[0], x[1]) - 1e-9


collapse.terminal = True
collapse.direction = -1


def one(job):
    th, q = job
    a = a0.copy()
    a[K, L] += th
    xeq = np.linalg.solve(a, np.ones(4))
    rng = np.random.default_rng([2026, int(round(1000 * th)), q])
    x0 = xeq * (1 + 0.1 * rng.uniform(-1, 1, size=4))
    t0 = time.time()
    sol = solve_ivp(vano_rhs, (0, T_END), x0, args=(r, a), method='DOP853', rtol=1e-10, atol=1e-13, events=collapse)
    stayed = sol.t_events[0].size == 0
    return th, q, int(stayed), (-1.0 if stayed else float(sol.t_events[0][0])), time.time() - t0


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    os.makedirs(OUT, exist_ok=True)
    x34, rate, theta_c = two_species_state()
    J34 = -np.diag(r[2:] * x34[2:]) @ a0[2:, 2:]
    print(f'equilibrium of species 3 and 4 alone: (x3, x4) = ({x34[2]:.4f}, {x34[3]:.4f}); eigenvalues within these two '
          f'species: {np.round(np.linalg.eigvals(J34).real, 3)}')
    print(f'invasion growth rate of species 1: {rate(0.0, 0):+.4f} at theta = 0, {rate(0.1, 0):+.4f} at theta = 0.1; '
          f'it is positive for theta < {theta_c:.4f}')
    print(f'invasion growth rate of species 2: {rate(0.0, 1):+.4f} at every theta (at theta = 0.3: {rate(0.3, 1):+.4f})')
    jobs = [(float(th), q) for th in THETAS for q in range(N_IC)]
    rows = []
    with Pool(workers) as pool:
        for n, (th, q, stayed, tc, sec) in enumerate(pool.imap_unordered(one, jobs), 1):
            rows.append((th, q, stayed, tc))
            print(f'{n}/{len(jobs)}: theta = {th:.2f}, ic {q}: ' + ('stayed' if stayed else f'escape at t = {tc:.0f}')
                  + f'  ({sec:.0f} s)', flush=True)
    path = os.path.join(OUT, 'crisis_scan_extended.csv')
    with open(path, 'w') as fh:
        fh.write('theta,ic,stayed,t_collapse\n')
        for th, q, stayed, tc in sorted(rows):
            fh.write(f'{th:.2f},{q},{stayed},{tc:.1f}\n')
    print('trajectories that stay for the whole run, of 12:')
    for th in THETAS:
        sub = [row for row in rows if abs(row[0] - th) < 1e-9]
        esc = [row[3] for row in sub if not row[2]]
        print(f'   theta = {th:.2f}: {sum(row[2] for row in sub):2d}'
              + (f'   (escapes at t = {min(esc):.0f} to {max(esc):.0f})' if esc else ''))
    print('saved data/crisis_scan_extended.csv')
