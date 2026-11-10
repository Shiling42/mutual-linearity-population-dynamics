"""
Experiment for Fig. 3, Fig. S3 and Table S1: time averages on cycles and chaotic attractors stay on the line of
interior equilibria.

The four-species competitive community of Vano et al. (2006), dx_i/dt = rho_i x_i (1 - sum_j a_ij x_j), with the
control a_13 -> a_13 + theta (species 1 is modified; A_13 = -a_13 because rho_1 = 1). For nine values of theta:
  * integration with an eighth-order Runge-Kutta method (DOP853, rtol 1e-9, atol 1e-12) from the interior equilibrium
    with the abundances changed by +-10%, a transient of 3000 time units, then 40 000 units sampled every 0.05 for
    the time average;
  * the largest Lyapunov exponent over 3000 time units (fourth-order Runge-Kutta steps of 0.01 for the state, Euler
    steps for the tangent vector), started at the end of the trajectory;
  * the regime: fixed point if the interior equilibrium is stable; otherwise limit cycle if the successive maxima of
    x_2 over the last 800 time units repeat, and chaos if they do not.
The interior equilibria x*(theta) lie on the line of Eqs. (4) and (6) of the paper.

The trajectories at theta >= 0 are chaotic, so their digits depend on the machine and on the library versions (see
README.md). The stored data are those of the run described there.

Run:  python3 -B code/exp_chaos.py      (about 1.5 minutes).   Output: data/exp_chaos.npz
"""
import os
import warnings
import numpy as np
from glv_tools import VANO_R as r, VANO_A as a0, METHOD, RTOL, ATOL, time_average, lyapunov, repeating_maxima

warnings.filterwarnings('ignore')
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
os.makedirs(OUT, exist_ok=True)
k, l = 0, 2
T_TRANSIENT, T_AVERAGE, DT_SAMPLE = 3000, 40000, 0.05      # time units: transient, averaging window, sampling
LYAP_STEPS, LYAP_DT = 300000, 0.01                           # Lyapunov exponent: 300 000 steps of 0.01

if __name__ == '__main__':
    thetas = np.array([-0.45, -0.38, -0.30, -0.20, -0.10, 0.0, 0.08, 0.15, 0.20])
    res = {}
    for th in thetas:
        a = a0.copy(); a[k, l] += th
        xeq = np.linalg.solve(a, np.ones(4))
        xa, xmin, xmax, X, t = time_average(r, a, xeq * (1 + 0.1 * np.array([1, -1, 1, -1])), t_transient=T_TRANSIENT,
                                            t_avg=T_AVERAGE, dt=DT_SAMPLE)
        ly = lyapunov(r, a, X[:, -1], n_steps=LYAP_STEPS, dt=LYAP_DT)
        J = -np.diag(r * xeq) @ a; mx = np.max(np.linalg.eigvals(J).real)
        tail = X[:, -16000:]
        regime = 'fixed point' if mx < 0 else ('limit cycle' if repeating_maxima(tail[1]) else 'chaos')
        # smallest abundance on the attractor: at a fixed point that of the equilibrium (near the Hopf bifurcation
        # the averaging window still contains the decaying transient), otherwise the minimum over the window
        minx = xeq.min() if regime == 'fixed point' else xmin.min()
        res[th] = dict(xeq=xeq, xavg=xa, xmin=xmin, xmax=xmax, lyap=ly, remax=mx, tail=tail, ttail=t[-16000:],
                       regime=regime, minx=minx)
        print(f"theta={th:+.2f}: {regime:11s}  Re(eig)max={mx:+.4f}, Lyapunov={ly:+.4f}, smallest abundance on the "
              f"attractor={minx:.1e} (in the averaging window: {xmin.min():.1e}; species 3: {xmin[2]:.1e}), "
              f"max|<x>-x*|={np.max(np.abs(xa-xeq)):.1e}, x*={np.round(xeq,4)}", flush=True)
    # straight line of equilibria: x*(theta) = x0 - theta x0_l / (1 + theta C_lk) C_.k with C = a0^{-1}; this is
    # Eqs. (4) and (6) of the paper with theta -> -theta, because A_13 = -a_13
    C = np.linalg.inv(a0); x0 = C @ np.ones(4)
    line_err = max(np.max(np.abs(res[th]['xeq'] - (x0 - th * x0[l] / (1 + th * C[l, k]) * C[:, k]))) for th in thetas)
    print(f"max deviation of the interior equilibria x*(theta) from the line of Eqs. (4) and (6): {line_err:.1e}")
    np.savez(os.path.join(OUT, 'exp_chaos.npz'), thetas=thetas, C=C, x0=x0,
             **{f'xeq_{i}': res[th]['xeq'] for i, th in enumerate(thetas)},
             **{f'xavg_{i}': res[th]['xavg'] for i, th in enumerate(thetas)},
             **{f'xmin_{i}': res[th]['xmin'] for i, th in enumerate(thetas)},
             **{f'lyap_{i}': res[th]['lyap'] for i, th in enumerate(thetas)},
             **{f'remax_{i}': res[th]['remax'] for i, th in enumerate(thetas)},
             **{f'tail_{i}': res[th]['tail'] for i, th in enumerate(thetas)},
             **{f'ttail_{i}': res[th]['ttail'] for i, th in enumerate(thetas)},
             # added for verify_sm_numbers.py: the regime and the smallest abundance on the attractor (Table S1)
             regime=np.array([res[th]['regime'] for th in thetas]),
             minx=np.array([res[th]['minx'] for th in thetas]), line_err=line_err,
             # the parameters of the integration
             method=METHOD, t_transient=T_TRANSIENT, t_avg=T_AVERAGE, dt_sample=DT_SAMPLE, rtol=RTOL, atol=ATOL,
             lyap_time=LYAP_STEPS * LYAP_DT, lyap_dt=LYAP_DT)
    print('saved data/exp_chaos.npz')
