"""
ml_core.py -- numerical core for "Mutual Linearity in Nonlinear Population Dynamics".

Contents
  * numerical_rank: numerical rank and singular values of a data matrix;
  * the generalized Lotka-Volterra (GLV) model  dx_i/dt = x_i (r_i + sum_j A_ij x_j):
      - glv_saturated_equilibria / glv_saturated_equilibrium: saturated equilibria by enumeration of all sets of
        present species;
      - glv_attractor: the state reached by direct integration (used as an independent check of the enumeration).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import svd


def numerical_rank(D, rtol=1e-8):
    """Numerical rank of data matrix D (columns = samples) and its singular values."""
    s = svd(D, compute_uv=False)
    return int(np.sum(s > rtol * s[0])), s


def glv_rhs(x, r, A):
    return x * (r + A @ x)


def glv_attractor(r, A, x0, t_max=4000.0):
    """Integrate GLV to large time; prune extinct species; polish interior equilibrium of survivors."""
    n = len(r)
    sol = solve_ivp(lambda t, x: glv_rhs(np.maximum(x, 0), r, A), (0, t_max), np.asarray(x0, float),
                    method='LSODA', rtol=1e-11, atol=1e-14)
    x = np.maximum(sol.y[:, -1], 0.0)
    alive = x > 1e-7
    xs = np.zeros(n)
    if alive.any():
        idx = np.where(alive)[0]
        xs[idx] = np.linalg.solve(A[np.ix_(idx, idx)], -r[idx])
        if np.any(xs < -1e-9):  # polishing failed -> fall back to integrated value
            xs = x
    return xs, alive


def glv_saturated_equilibria(r, A, tol=1e-10):
    """
    All saturated equilibria, by enumeration of the 2^n - 1 sets S of present species. A set is kept if
      (i)   the equilibrium on S, x_S = -A_SS^{-1} r_S, is positive;
      (ii)  no absent species can invade (its per-capita growth rate r_e + sum_j A_ej x_j is not positive);
      (iii) the equilibrium is locally stable within S (no eigenvalue of diag(x_S) A_SS with positive real part).
    Returns a list of (x, alive): the abundances (zero for absent species) and the boolean mask of present species.
    The list is empty if no such equilibrium exists (unbounded growth, or a cycle or chaotic attractor).
    """
    n = len(r)
    found = []
    for mask in range(1, 2 ** n):
        idx = [q for q in range(n) if (mask >> q) & 1]
        As = A[np.ix_(idx, idx)]
        try:
            xs = np.linalg.solve(As, -r[idx])
        except np.linalg.LinAlgError:
            continue
        if np.any(xs <= tol):
            continue
        x = np.zeros(n); x[idx] = xs
        growth = r + A @ x
        out = [q for q in range(n) if q not in idx]
        if any(growth[q] > tol for q in out):
            continue
        J = np.diag(xs) @ As
        if np.any(np.linalg.eigvals(J).real > tol):
            continue
        found.append((x, np.array([q in idx for q in range(n)])))
    return found


def glv_saturated_equilibrium(r, A, tol=1e-10, multiplicity=None):
    """
    The saturated equilibrium (see glv_saturated_equilibria). Returns (x*, alive), or (None, None) if there is none.
    If several exist, the one with the most species present is returned. The communities of the paper never have
    several: pass a list as `multiplicity` and the number found is appended to it, so that the caller can check.
    """
    found = glv_saturated_equilibria(r, A, tol)
    if multiplicity is not None:
        multiplicity.append(len(found))
    best = None
    for cand in found:
        if best is None or int(cand[1].sum()) > int(best[1].sum()):
            best = cand
    return best if best is not None else (None, None)
