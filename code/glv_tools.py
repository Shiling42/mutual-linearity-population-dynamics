"""
glv_tools.py -- tools for "Mutual Linearity in Nonlinear Population Dynamics": random GLV communities, their saturated
equilibria along a sweep of a control parameter, the press response, the direction after a kink, and time averages
and Lyapunov exponents on cycles and chaotic attractors.

Sign convention. The paper writes B = -A^{-1} for the press-response matrix and b_k for its k-th column, so that a
press r_k -> r_k + p moves the coexistence equilibrium to x0* + p b_k. Some scripts work with A^{-1} itself, whose
column k is -b_k; each says so where it takes the inverse. The effective press s stored by the experiments is the
paper's s, with x* = x0* + s b_k.
"""
import numpy as np
from scipy.integrate import solve_ivp
from ml_core import glv_saturated_equilibrium


def make_community(n=6, seed=28, sigma=1.1, rmin=0.5, rmax=1.5, xmin=0.05):
    """Random community: A_ii = -1, A_ij = sigma * N(0,1) / sqrt(n), r_i uniform in [rmin, rmax]; draws are repeated
    until the interior equilibrium has all abundances above xmin and is locally stable. (sigma is gamma in the paper.)"""
    rng = np.random.default_rng(seed)
    while True:
        A = -np.eye(n) + sigma * rng.normal(size=(n, n)) / np.sqrt(n)
        np.fill_diagonal(A, -1.0)
        r = rng.uniform(rmin, rmax, size=n)
        x = np.linalg.solve(A, -r)
        if np.all(x > xmin) and np.all(np.linalg.eigvals(np.diag(x) @ A).real < 0):
            return r, A


def interior(r, A):
    return np.linalg.solve(A, -r)


def sweep(r, A, modify, thetas, multiplicity=None):
    """
    modify(theta) -> (r_theta, A_theta).  Returns X (n x T) and SUPP (list of tuples: 1 for a present species).
    At each theta the saturated equilibrium is found by enumeration (ml_core.glv_saturated_equilibrium); where none
    exists the column of X is NaN. Uniqueness of the saturated equilibrium is not assumed: if a list is passed as
    `multiplicity`, the number of saturated equilibria found at each theta is appended to it. The experiments store
    these numbers; they are 1 at every point of every sweep of the paper, and 0 where the equilibrium has diverged.
    """
    X, SUPP = [], []
    for th in thetas:
        rt, At = modify(th)
        xs, alive = glv_saturated_equilibrium(rt, At, multiplicity=multiplicity)
        if xs is None:
            xs = np.full(len(r), np.nan); alive = np.zeros(len(r), bool)
        X.append(xs); SUPP.append(tuple(int(v) for v in alive))
    return np.array(X).T, SUPP


def segments(SUPP):
    """Index ranges (start, stop) of the branches of fixed composition along a sweep."""
    segs, start = [], 0
    for m in range(1, len(SUPP) + 1):
        if m == len(SUPP) or SUPP[m] != SUPP[start]:
            segs.append((start, m)); start = m
    return segs


def line_residual_affine(X):
    """Relative residual of an affine line fit to columns of X (ratio of 2nd to 1st singular value after centering)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s[1] / s[0] if s[0] > 0 else 0.0


def kink_prediction(A, k, S_before, e):
    """Predicted direction after species e leaves the set S_before of present species: Eq. (7) of the paper,
    b_k' = b_k - (B_ek / B_ee) b_e. (The formula has the same form for A^{-1} and for B = -A^{-1}.)"""
    S = list(S_before)
    B = np.linalg.inv(A[np.ix_(S, S)])
    ik, ie = S.index(k), S.index(e)
    d = B[:, ik]
    dnew = d - (d[ie] / B[ie, ie]) * B[:, ie]
    out = np.zeros(A.shape[0]); out[S] = dnew
    return out


# ---------------------------------------------------------------------------------------------
# time averages on cycles and chaotic attractors (form of Vano et al. 2006: dx_i/dt = r_i x_i (1 - sum_j a_ij x_j))
# ---------------------------------------------------------------------------------------------
VANO_R = np.array([1.0, 0.72, 1.53, 1.27])
VANO_A = np.array([[1.0, 1.09, 1.52, 0.0],
                   [0.0, 1.0, 0.44, 1.36],
                   [2.33, 0.0, 1.0, 0.47],
                   [1.21, 0.51, 0.35, 1.0]])


METHOD, RTOL, ATOL = 'DOP853', 1e-9, 1e-12       # integration in time_average: eighth-order Runge-Kutta, tolerances


def vano_rhs(t, x, r, a):
    return r * x * (1.0 - a @ x)


def time_average(r, a, x0, t_transient=2000.0, t_avg=20000.0, dt=0.02):
    """Long-time average of a trajectory after a transient; also returns min abundance and the trajectory tail."""
    sol = solve_ivp(vano_rhs, (0, t_transient + t_avg), x0, args=(r, a), method=METHOD,
                    rtol=RTOL, atol=ATOL, dense_output=True)
    t = np.arange(t_transient, t_transient + t_avg, dt)
    X = sol.sol(t)
    return X.mean(axis=1), X.min(axis=1), X.max(axis=1), X, t


def lyapunov(r, a, x0, n_steps=400000, dt=0.01, seed=0):
    """Largest Lyapunov exponent: fourth-order Runge-Kutta steps for the state, Euler steps for the tangent vector."""
    def f(x): return r * x * (1 - a @ x)
    def jac(x): return np.diag(r * (1 - a @ x)) - np.diag(r * x) @ a
    x = x0.copy(); v = np.random.default_rng(seed).normal(size=len(x0)); v /= np.linalg.norm(v); lam = 0.0
    for _ in range(n_steps):
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        v = v + dt * (jac(x) @ v); nv = np.linalg.norm(v); lam += np.log(nv); v /= nv
    return lam / (n_steps * dt)


def repeating_maxima(x, tol=1e-4):
    """True if the successive maxima of the time series x repeat (a cycle): fewer distinct values than half the
    number of maxima. Used to tell a limit cycle from chaos; the figure scripts apply the same rule."""
    m = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    return 1 + np.sum(np.diff(np.sort(x[m])) > tol) <= len(m) / 2


def boundary_state(r, a, present):
    """Equilibrium of the species in `present` alone, and the per-capita growth rates r_i (1 - sum_j a_ij x_j) of all
    species there (the invasion growth rates of the absent ones)."""
    S = list(present)
    x = np.zeros(len(r)); x[S] = np.linalg.solve(a[np.ix_(S, S)], np.ones(len(S)))
    return x, r * (1.0 - a @ x)
