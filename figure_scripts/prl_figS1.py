"""Fig. S1 -- the effective press, nonlinear functional responses, and the universality of the press path
(two columns, 7.0 in).

Run:  python3 -B figure_scripts/prl_figS1.py      (from any directory; needs pdftocairo)
(a) effective press s against the control for the four modifications of species 3 of Fig. 1 (data/exp_collinear.npz);
(b) the same modifications in the variant with saturating (type II) effects of species 1 and 2 (same npz): one curve;
(c) the six-species community of Fig. 2 (data/exp_stepwise_example.npz): five different modifications of species 2
    -- a press on r_2, changes of A_22, A_23 and A_21, and a mixed change -- all trace the press path of species 2,
    kinks included. The sweeps of (c) are computed here with the saturated-equilibrium enumeration of code/ml_core.py.
Output: figures/figS1.pdf (7.0 in wide) and a 300-dpi PNG.
"""
import os
import subprocess
import numpy as np
from matplotlib.lines import Line2D
from prl_style import plt, C, DATA, ROOT, TWO_COL, OUT
from glv_tools import sweep
from ml_core import glv_saturated_equilibrium

T_COL = ['k', '0.35', '0.15', '0.55']          # modification types: greys + shapes (colour = species elsewhere)
T_LS = ['-', (0, (4, 1.5)), (0, (1, 1.2)), (0, (6, 1.5, 1, 1.5))]
T_MK = ['o', 's', '^', 'D']

# ------------------------------------------------------------------ (a), (b): community of Fig. 1
d = np.load(os.path.join(DATA, 'exp_collinear.npz'), allow_pickle=True)
k1 = int(d['k'])
TH = [d[f'th_{i}'] for i in range(4)]
S = [d[f's_{i}'] for i in range(4)]
CV = [d[f'C_{i}'] for i in range(4)]
xn0 = d['xn0']
LAB1 = [r'press $r_3$', r'self-regulation $A_{33}$', r'interaction $A_{31}$', 'mixed']

# ------------------------------------------------------------------ (c): community of Fig. 2
e = np.load(os.path.join(DATA, 'exp_stepwise_example.npz'))
r, A = e['r'], e['A']
n, k, l = len(r), int(e['k']), int(e['l'])
E = np.eye(n)
MODS = {
    r'press $r_2$': (lambda th: (r + th * E[k], A), np.linspace(-1.0, 1.95, 60)),
    r'self-regulation $A_{22}$': (lambda th: (r, A + th * np.outer(E[k], E[k])), np.linspace(-0.9, 3.0, 120)),
    r'interaction $A_{23}$': (lambda th: (r, A + th * np.outer(E[k], E[l])), np.linspace(-0.6, 3.0, 361)),
    r'interaction $A_{21}$': (lambda th: (r, A + th * np.outer(E[k], E[0])), np.linspace(-3.0, 3.0, 120)),
    'mixed': (lambda th: (r + 0.3 * np.sin(2 * th) * E[k],
                          A + np.outer(E[k], th ** 2 * E[4] - 0.7 * th * E[5])), np.linspace(-2.0, 2.0, 120)),
}
SW = {}
SVALS = []
worst = 0.0
for name, (mod, ths) in MODS.items():
    X, SUPP = sweep(r, A, mod, ths)
    keep = ~np.isnan(X[0])
    print(f'(c) {name}: {int(keep.sum())} of {len(keep)} values of theta have a saturated equilibrium and are drawn')
    X, ths = X[:, keep], ths[keep]
    for q in range(X.shape[1]):                 # check: x*(theta) = x*_press(s(theta)), s = h(x*; theta)
        rt, At = mod(ths[q])
        s = (rt[k] - r[k]) + (At[k] - A[k]) @ X[:, q] if X[k, q] > 0 else -10.0
        if X[k, q] > 0:
            SVALS.append(s)
        xp, _ = glv_saturated_equilibrium(r + s * E[k], A)
        worst = max(worst, np.max(np.abs(xp - X[:, q])))
    SW[name] = X
print(f'(c) max |x*(theta) - x*_press(s(theta))| over all five modifications = {worst:.1e}')
PGRID = np.linspace(min(SVALS), max(SVALS), 400)
PPATH = np.array([glv_saturated_equilibrium(r + p * E[k], A)[0] for p in PGRID]).T

W, H = TWO_COL, 2.25
fig = plt.figure(figsize=(W, H))


def head(x_in, letter, title):
    """Panel label "(a)" in regular weight + short title, as in Figs. S2-S4."""
    fig.text(x_in / W, TOPY / H, f'({letter.lower()})', fontsize=8.5, fontweight='normal', ha='left', va='baseline')
    fig.text((x_in + 0.24) / W, TOPY / H, title, fontsize=7.5, ha='left', va='baseline')


AX = [[0.45, 0.36, 1.75, 1.66], [2.85, 0.36, 1.75, 1.66], [5.20, 0.36, 1.75, 1.66]]
TOPY = 0.36 + 1.66 + 0.07

ax = fig.add_axes([AX[0][0] / W, AX[0][1] / H, AX[0][2] / W, AX[0][3] / H])
ax.axhline(0, color='0.85', lw=0.5, zorder=0); ax.axvline(0, color='0.85', lw=0.5, zorder=0)
for q in [2, 1, 0, 3]:
    ax.plot(TH[q], S[q], color=T_COL[q], ls=T_LS[q], marker=T_MK[q], ms=2.8, markevery=8, lw=1.0, label=LAB1[q],
            mfc=T_COL[q] if q != 3 else 'white')
ax.set_xlim(-0.85, 1.1); ax.set_ylim(-1.0, 3.0)
ax.set_xlabel(r'control parameter $\theta$'); ax.set_ylabel(r'effective press $s$')
h_, l_ = ax.get_legend_handles_labels()
order = [2, 1, 0, 3]
ax.legend([h_[order.index(q)] for q in range(4)], [l_[order.index(q)] for q in range(4)], loc='upper left',
          handlelength=1.5, labelspacing=0.2, borderaxespad=0.1, fontsize=6.5)
head(AX[0][0] - 0.43, 'A', 'Effective press')

ax = fig.add_axes([AX[1][0] / W, AX[1][1] / H, AX[1][2] / W, AX[1][3] / H])
for q in range(4):
    xx, yy = CV[q][k1], CV[q][0]
    t = np.hypot((xx - xn0[k1]) / 2.6, (yy - xn0[0]) / 5.0) * np.sign(xx - xn0[k1])
    _, idx = np.unique(np.floor((t + q / 4 * 0.06) / 0.06), return_index=True)     # thin for legibility
    ax.scatter(xx[idx], yy[idx], s=11, marker=T_MK[q], facecolor=T_COL[q] if q != 3 else 'white',
               edgecolor='k', linewidth=0.4, zorder=3 + (q == 0))
P = CV[2]; i0, i1 = np.argmin(P[k1]), np.argmax(P[k1])
ax.plot([P[k1][i0], P[k1][i1]], [P[0][i0], P[0][i1]], color='0.5', lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.scatter([xn0[k1]], [xn0[0]], marker='*', s=40, color='k', edgecolor='white', linewidth=0.3, zorder=6)
ax.text(1.05, 5.4, 'straight chord', color='0.4', fontsize=6.5, rotation=34, ha='center', va='bottom')
ax.set_xlabel(r'$x_3^\ast$ (modified species)'); ax.set_ylabel(r'$x_1^\ast$')
head(AX[1][0] - 0.43, 'B', 'Type II responses: one curve')

ax = fig.add_axes([AX[2][0] / W, AX[2][1] / H, AX[2][2] / W, AX[2][3] / H])
names = list(MODS)


def thin(x, y, dmin=0.11):
    """Keep points at least dmin apart along the path (data units), so that markers do not pile up."""
    keep, last = [], None
    for q in np.argsort(x):
        if last is None or np.hypot(x[q] - x[last], y[q] - y[last]) > dmin:
            keep.append(q); last = q
    return np.array(keep)


MK = {names[2]: ('x', 7), names[1]: ('s', 8), names[3]: ('^', 9), names[4]: ('D', 6)}
for sp in [2, 4]:
    ax.plot(PPATH[0], PPATH[sp], color=C[sp], lw=1.6, alpha=0.45, zorder=2, solid_joinstyle='miter')
    for j, name in enumerate([names[2], names[1], names[3], names[4]]):
        X = SW[name]
        q = thin(X[0], X[sp], dmin=0.13 + 0.02 * j)
        mk, ms = MK[name]
        if mk == 'x':
            ax.scatter(X[0][q], X[sp][q], s=ms, marker=mk, color=C[sp], linewidth=0.6, zorder=3)
        else:
            ax.scatter(X[0][q], X[sp][q], s=ms, marker=mk, facecolor='none', edgecolor=C[sp], linewidth=0.55,
                       zorder=3)
    ax.text(2.95, np.interp(2.95, PPATH[0], PPATH[sp]) + 0.22, f'$x_{sp + 1}$', color=C[sp], fontsize=7.5,
            ha='center', va='bottom')
ax.set_xlim(0.45, 3.25); ax.set_ylim(-0.15, 4.15)
ax.set_xlabel(r'$x_1^\ast$'); ax.set_ylabel(r'$x_i^\ast$')
hand = [Line2D([], [], color='0.45', lw=1.6, alpha=0.6, label=r'press path ($r_2$)')]
hand += [Line2D([], [], ls='none', marker=MK[names[j]][0], ms=3.2, mfc='none', mec='0.2', mew=0.6, label=names[j])
         for j in [2, 1, 3, 4]]
ax.legend(handles=hand, loc='upper right', labelspacing=0.15, borderaxespad=0.05, handlelength=1.2, fontsize=6.2,
          handletextpad=0.3)
head(AX[2][0] - 0.43, 'C', 'All modifications: one kinked path')

if __name__ == '__main__':
    out = os.path.join(OUT, 'figS1.pdf')
    fig.savefig(out)
    subprocess.run(['pdftocairo', '-png', '-r', '300', '-singlefile', out, out[:-4]], check=True)
    print('saved', os.path.relpath(out, ROOT))
