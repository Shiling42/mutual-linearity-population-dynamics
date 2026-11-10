"""Fig. 3 -- time averages on cycles and chaos stay on the line (one column, 3.375 in).

Run:  python3 -B figure_scripts/prl_fig3.py      (from any directory; needs pdftocairo)
Data: data/exp_chaos.npz (written by code/exp_chaos.py): the four-species competitive Lotka-Volterra community of
Vano et al. (2006), dx_i/dt = rho_i x_i (1 - sum_j a_ij x_j), with a_13 -> a_13 + theta (species 1 is modified).
(a) attractors at two values of theta in the (x2, x4) plane (limit cycle at theta = -0.20, chaos at theta = 0.08),
the line of interior equilibria and the time averages at all nine values of theta (inset: zoom); (b) time averages
against theta and the equilibria x*(theta).
Output: figures/fig3.pdf and a 300-dpi PNG. No TikZ.

The regime is read off the successive maxima of x2 on the stored tail (they repeat on a cycle); a threshold on the
Lyapunov exponent would class the weak chaos at theta = 0.15 as a limit cycle. The line of interior equilibria in (a)
is drawn over its whole feasible range, x0 - s C_.k with all abundances positive, as in Fig. S3(b). Panel (a) uses the
regime colours of Fig. S3(b): limit cycle light blue, chaos orange (Okabe-Ito C[5], C[4], colours that no species
uses in this four-species community); the markers of the time averages are filled likewise.
"""
import os
import numpy as np
from scipy.optimize import brentq
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from prl_style import plt, C, DATA, ROOT, ONE_COL, panel_label, OUT
from glv_tools import VANO_R as r_, VANO_A as a0, repeating_maxima
import subprocess

d = np.load(os.path.join(DATA, 'exp_chaos.npz'))
thetas, Cinv, x0 = d['thetas'], d['C'], d['x0']
k, l = 0, 2
nT = len(thetas)
xavg = np.array([d[f'xavg_{i}'] for i in range(nT)])
lyap = np.array([float(d[f'lyap_{i}']) for i in range(nT)])
remax = np.array([float(d[f'remax_{i}']) for i in range(nT)])


regime = ['fixed point' if remax[i] < 0 else ('limit cycle' if repeating_maxima(d[f'tail_{i}'][1]) else 'chaos')
          for i in range(nT)]
SHOW = [0, 3, 6]                       # theta = -0.45 (fixed point), -0.20 (limit cycle), +0.08 (chaos)
REG_MK = {'fixed point': 'o', 'limit cycle': 's', 'chaos': '^'}
REG_COL = {'fixed point': 'k', 'limit cycle': C[5], 'chaos': C[4]}           # marker fill
TRAJ_COL = {'limit cycle': C[5], 'chaos': C[4]}                                  # trajectory colour
TRAJ_LW = {'limit cycle': 0.8, 'chaos': 0.4}
TRAJ_ALPHA = {'limit cycle': 0.95, 'chaos': 0.55}


def line(th):
    """Interior equilibrium x*(theta): a straight line in abundance space [Eqs. (4) and (6) with theta -> -theta]."""
    return x0 - th * x0[l] / (1 + th * Cinv[l, k]) * Cinv[:, k]


dev = max(np.max(np.abs(xavg[i] - line(thetas[i]))) for i in range(nT))
print('max |<x> - x*(theta)| over the nine values of theta:', f'{dev:.1e}')


def remax_of(th):
    a = a0.copy(); a[k, l] += th
    xe = np.linalg.solve(a, np.ones(4))
    return np.max(np.linalg.eigvals(-np.diag(r_ * xe) @ a).real)


TH_HOPF = brentq(remax_of, -0.45, -0.30)
print(f'Hopf bifurcation at theta = {TH_HOPF:.4f}')

W, H = ONE_COL, 3.36
AX_A = [0.43, 1.83, 2.87, 1.38]
AX_B = [0.43, 0.33, 2.87, 1.08]
fig = plt.figure(figsize=(W, H))

# ---------------- (a) attractors and their time averages in the (x2, x4) plane
ax = fig.add_axes([AX_A[0] / W, AX_A[1] / H, AX_A[2] / W, AX_A[3] / H])
i2, i4 = 1, 3
cvec = Cinv[:, k]                      # the whole feasible line x0 - s C_.k (all abundances positive)
s_lo = max(x0[j] / cvec[j] for j in range(4) if cvec[j] < 0)
s_hi = min(x0[j] / cvec[j] for j in range(4) if cvec[j] > 0)
L = np.array([x0 - s * cvec for s in np.linspace(s_lo, s_hi, 400)])
ok = np.ones(len(L), bool)
ax.plot(L[ok, i2], L[ok, i4], color='k', lw=0.9, zorder=2)
for i in SHOW[1:]:
    tail = d[f'tail_{i}']
    ax.plot(tail[i2], tail[i4], color=TRAJ_COL[regime[i]], lw=TRAJ_LW[regime[i]], alpha=TRAJ_ALPHA[regime[i]],
            zorder=1 if regime[i] == 'chaos' else 1.5)
for i in range(nT):
    ax.scatter(xavg[i, i2], xavg[i, i4], marker=REG_MK[regime[i]], s=11, color=REG_COL[regime[i]],
               edgecolor='k', linewidth=0.35, zorder=4)
ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.66)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_xlabel(r'$x_2$', labelpad=1); ax.set_ylabel(r'$x_4$')
hand = [Line2D([], [], color=TRAJ_COL.get(g, 'k'), lw=1.0, marker=REG_MK[g], ms=3.4, mfc=REG_COL[g], mec='k',
               mew=0.35, label=g) for g in ['fixed point', 'limit cycle', 'chaos']]
hand[0].set_linestyle('none')
hand.insert(0, Line2D([], [], color='k', lw=0.9, label=r'equilibria $\mathbf{x}^\ast(\theta)$'))
ax.legend(handles=hand, loc='lower left', handletextpad=0.4, labelspacing=0.15, borderaxespad=0.1,
          handlelength=1.3, fontsize=6.5)
# inset: zoom on the time averages
x_lo, x_hi, y_lo, y_hi = 0.425, 0.525, 0.305, 0.38
axi = fig.add_axes([(AX_A[0] + 1.98) / W, (AX_A[1] + 0.80) / H, 0.86 / W, 0.56 / H])
axi.plot(L[ok, i2], L[ok, i4], color='k', lw=0.8, zorder=1)
for i in range(nT):
    axi.scatter(xavg[i, i2], xavg[i, i4], marker=REG_MK[regime[i]], s=10, color=REG_COL[regime[i]],
                edgecolor='k', linewidth=0.35, zorder=3)
axi.set_xlim(x_lo, x_hi); axi.set_ylim(y_lo, y_hi)
axi.set_xticks([]); axi.set_yticks([])
for s_ in axi.spines.values():
    s_.set_visible(True); s_.set_linewidth(0.5); s_.set_color('0.5')
ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fill=False, lw=0.5, ec='0.5', zorder=5))
for (xa, ya), (xb, yb) in [((x_hi, y_hi), (0, 1)), ((x_hi, y_lo), (0, 0))]:
    fig.add_artist(ConnectionPatch((xa, ya), (xb, yb), coordsA=ax.transData, coordsB=axi.transAxes,
                                   lw=0.4, color='0.6'))
axi.text(0.03, 0.06, 'time averages', fontsize=6.5, ha='left', va='bottom', transform=axi.transAxes, color='0.3')
panel_label(fig, 0.02, AX_A[1] + AX_A[3] + 0.04, 'a', W, H)

# ---------------- (b) time averages against theta, across the bifurcations
ax = fig.add_axes([AX_B[0] / W, AX_B[1] / H, AX_B[2] / W, AX_B[3] / H])
tt = np.linspace(-0.5, 0.235, 300)
LL = np.array([line(t) for t in tt])
for sp in range(4):
    ax.plot(tt, LL[:, sp], color=C[sp], lw=0.9, zorder=2)
    for i in range(nT):
        ax.scatter(thetas[i], xavg[i, sp], marker=REG_MK[regime[i]], s=10, facecolor=C[sp], edgecolor='k',
                   linewidth=0.35, zorder=4)
ax.axvline(TH_HOPF, color='0.45', lw=0.6, ls=(0, (1, 1.6)), zorder=1)
ax.text(TH_HOPF + 0.008, 0.585, 'Hopf', fontsize=6.5, color='0.35', ha='left', va='top')
ax.set_xlim(-0.5, 0.25); ax.set_ylim(0.05, 0.6)
ax.set_xticks([-0.4, -0.2, 0, 0.2]); ax.set_xticklabels(['−0.4', '−0.2', '0.0', '0.2'])
ax.set_yticks([0.1, 0.3, 0.5])
ax.set_xlabel(r'$\theta$ (change of $a_{13}$)', labelpad=1); ax.set_ylabel(r'$\langle x_i\rangle$')
for sp, dy in {0: -0.035, 1: 0.038, 2: 0.035, 3: -0.035}.items():
    ax.text(0.245, line(0.235)[sp] + dy, f'sp. {sp + 1}', color=C[sp], fontsize=7, ha='right', va='center')
panel_label(fig, 0.02, AX_B[1] + AX_B[3] + 0.05, 'b', W, H)

if __name__ == '__main__':
    out = os.path.join(OUT, 'fig3.pdf')
    fig.savefig(out)
    subprocess.run(['pdftocairo', '-png', '-r', '300', '-singlefile', out, out[:-4]], check=True)
    print('saved', os.path.relpath(out, ROOT))
    print('regimes:', list(zip(thetas, regime, np.round(lyap, 4))))
