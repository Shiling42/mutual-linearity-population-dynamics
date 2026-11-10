"""Fig. 2 -- extinctions and invasions kink the line; every modification of species k follows the same path
(two columns, 7.0 in).

Run:  python3 -B figure_scripts/prl_fig2.py      (from any directory; needs pdflatex and pdftocairo)
Data: data/exp_stepwise_example.npz (written by code/exp_stepwise.py).
  1. matplotlib draws panels (a) and (b) into figure_scripts/build_fig2/fig2_panels.pdf and leaves the slot of panel
     (c) empty. Panel (a) carries a composition strip on top (one row per species, coloured while present, gray while
     absent; the rows are labelled sp. 1 ... sp. 6 in their colours). In panel (b) each change of composition is a
     circle in the colour of the species that changes, filled for an invasion and open for an extinction; a light
     band in each species' colour under its line is the press path of species 2 (saturated equilibria under
     r_2 -> r_2 + p on a dense grid of presses, so that the band follows the kinks), computed here with the same
     enumeration routine and never copied from the sweep of A_23;
  2. it compiles prl_fig2_sketch.tex (panel c, geometry of a kink) and places it with prl_assemble.tex;
  3. the result is figures/fig2.pdf (7.0 in wide) and a 300-dpi PNG for checking.
The kink directions are predicted by Eq. (7) of the paper, b_k' = b_k - (B_ek/B_ee) b_e with B = -A_SS^{-1} on the
set S of species present before the extinction.
"""
import os
import numpy as np
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from prl_style import plt, C, DATA, HERE, TWO_COL, panel_label, pdflatex, finish
from glv_tools import segments
from ml_core import glv_saturated_equilibrium

BUILD = os.path.join(HERE, 'build_fig2')
os.makedirs(BUILD, exist_ok=True)

d = np.load(os.path.join(DATA, 'exp_stepwise_example.npz'))
X, TH = d['X'], d['thetas']
SUPP = [tuple(int(v) for v in s) for s in d['SUPP']]
r, A, k, l = d['r'], d['A'], int(d['k']), int(d['l'])
n = X.shape[0]
segs = segments(SUPP)

KINKS = []
for q in range(1, len(segs)):
    (p0, p1), (c0, c1) = segs[q - 1], segs[q]
    p, u = X[:, p0], X[:, p1 - 1] - X[:, p0]
    c, v = X[:, c0], X[:, c1 - 1] - X[:, c0]
    K = p.copy() if np.linalg.norm(u) < 1e-12 else p + np.linalg.lstsq(np.column_stack([u, -v]), c - p, rcond=None)[0][0] * u
    th_K = -(r[k] + A[k] @ K) / K[l]
    Sb = [i for i in range(n) if SUPP[p0][i]]
    Sa = [i for i in range(n) if SUPP[c0][i]]
    kink = dict(theta=th_K, K=K, gained=sorted(set(Sa) - set(Sb)), lost=sorted(set(Sb) - set(Sa)))
    if len(kink['lost']) == 1 and not kink['gained']:
        e = kink['lost'][0]
        Bs = -np.linalg.inv(A[np.ix_(Sb, Sb)])
        bk = np.zeros(n); bk[Sb] = Bs[:, Sb.index(k)]
        be = np.zeros(n); be[Sb] = Bs[:, Sb.index(e)]
        bnew = bk - Bs[Sb.index(e), Sb.index(k)] / Bs[Sb.index(e), Sb.index(e)] * be
        d_obs = X[:, c1 - 1] - X[:, c0]
        kink['pred'] = np.sign(d_obs @ bnew) * bnew / np.linalg.norm(bnew)
    KINKS.append(kink)

TP = [TH[segs[0][0]:segs[0][1]]]
XP = [X[:, segs[0][0]:segs[0][1]]]
for q, kink in enumerate(KINKS):
    c0, c1 = segs[q + 1]
    TP += [[kink['theta']], TH[c0:c1]]
    XP += [kink['K'][:, None], X[:, c0:c1]]
TP, XP = np.concatenate(TP), np.hstack(XP)
TB = [TH[0]] + [kk['theta'] for kk in KINKS] + [TH[-1]]

present = X[k] > 0
s_sweep = TH * X[l]
P_GRID = np.linspace(s_sweep[present].min(), s_sweep[present].max(), 600)
PRESS = np.array([glv_saturated_equilibrium(r + p * np.eye(n)[k], A)[0] for p in P_GRID]).T

# ----------------------------------------------------------------------------------------------- layout
W, H = TWO_COL, 2.50
AX_BAR = [0.40, 1.83, 1.98, 0.50]           # composition strip
AX_A = [0.40, 0.34, 1.98, 1.40]
AX_B = [2.90, 0.34, 1.98, 1.99]
SLOT = dict(x=5.04, y=0.02, w=W - 5.04, h=H - 0.04)
TOPY = AX_B[1] + AX_B[3] + 0.06
fig = plt.figure(figsize=(W, H))
CHANGE = dict(color='0.45', lw=0.6, ls=(0, (1, 1.6)))
PRED_GREY = '0.68'
ABSENT = '0.90'

# ---------------- composition strip (species 1 on top)
axs = fig.add_axes([AX_BAR[0] / W, AX_BAR[1] / H, AX_BAR[2] / W, AX_BAR[3] / H])
BAR = 0.70
row = lambda sp: n - 1 - sp
for sp in range(n):
    states = [SUPP[s0][sp] for (s0, s1) in segs]
    start = 0
    for q in range(1, len(states) + 1):
        if q == len(states) or states[q] != states[start]:
            axs.broken_barh([(TB[start], TB[q] - TB[start])], (row(sp) - BAR / 2, BAR),
                            facecolors=C[sp] if states[start] else ABSENT, edgecolor='none', zorder=2)
            start = q
axs.set_xlim(TH[0], TH[-1]); axs.set_ylim(-0.55, n - 0.45)
axs.set_yticks([row(sp) for sp in range(n)])
axs.set_yticklabels([f'sp. {sp + 1}' for sp in range(n)], fontsize=6.5)
for sp, lab in enumerate(axs.get_yticklabels()):
    lab.set_color(C[sp])                         # labels are listed in the order the ticks were given
axs.tick_params(axis='y', length=0, pad=2.0)
axs.tick_params(axis='x', bottom=False, labelbottom=False)
for side in axs.spines.values():
    side.set_visible(False)

# ---------------- (a) abundances against the control
ax = fig.add_axes([AX_A[0] / W, AX_A[1] / H, AX_A[2] / W, AX_A[3] / H])
for sp in range(n):
    ax.plot(TP, XP[sp], color=C[sp], lw=1.15, zorder=3 if sp == k else 2)
ax.set_xlim(TH[0], TH[-1]); ax.set_ylim(-0.1, 4.1)
ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlabel(r'$\theta$ (change of $A_{23}$)'); ax.set_ylabel(r'abundance $x_i^\ast$')
# dotted change lines: behind the curves in (a), behind the bars in the strip, and across the gap between them
for kink in KINKS:
    ax.axvline(kink['theta'], zorder=1, **CHANGE)
    axs.axvline(kink['theta'], zorder=1, **CHANGE)
    xf = fig.transFigure.inverted().transform(ax.transData.transform((kink['theta'], 0)))[0]
    fig.add_artist(Line2D([xf, xf], [(AX_A[1] + AX_A[3]) / H, AX_BAR[1] / H], transform=fig.transFigure, **CHANGE))
panel_label(fig, AX_A[0] - 0.38, TOPY, 'a', W, H)

# ---------------- (b) path in abundance space
ax = fig.add_axes([AX_B[0] / W, AX_B[1] / H, AX_B[2] / W, AX_B[3] / H])
PAIRS = [1, 2, 4, 5]
for sp in PAIRS:
    ax.plot(XP[0], XP[sp], color=C[sp], lw=1.15, zorder=3, solid_joinstyle='miter')
    ax.plot(PRESS[0], PRESS[sp], color=C[sp], lw=3.4, alpha=0.22, solid_capstyle='round', solid_joinstyle='round',
            zorder=1)                    # press path: light band under the line
ax.set_xlim(0.45, 2.85); ax.set_ylim(-0.12, 4.25)
ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5]); ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlabel(r'abundance $x_1^\ast$'); ax.set_ylabel(r'abundance $x_i^\ast$')


def page_dir(p, dvec):
    q0 = ax.transData.transform(p); q1 = ax.transData.transform(np.asarray(p) + 1e-3 * np.asarray(dvec))
    dq = q1 - q0
    return dq / np.linalg.norm(dq)


def offset_points(xy, dx_pt, dy_pt):
    disp = ax.transData.transform(xy)
    return ax.transData.inverted().transform(disp + np.array([dx_pt, dy_pt]) * fig.dpi / 72.0)


for q, kink in enumerate(KINKS):
    if 'pred' not in kink:
        continue
    K, dp = kink['K'], kink['pred']
    branch_end = X[:, segs[q + 1][1] - 1]
    for sp in PAIRS:
        p = np.array([K[0], K[sp]])
        u = page_dir(p, [dp[0], dp[sp]])
        avail = np.linalg.norm(ax.transData.transform([branch_end[0], branch_end[sp]]) - ax.transData.transform(p))
        tip = offset_points(p, *(min(0.30 * 72, 0.75 * avail * 72.0 / fig.dpi) * u))
        ax.add_patch(FancyArrowPatch(p, tip, arrowstyle=ArrowStyle('simple', head_length=0.66, head_width=0.66,
                                                                    tail_width=0.34),
                                     mutation_scale=8.5, color=PRED_GREY, lw=0, zorder=2, shrinkA=0, shrinkB=0))
# kinks: circle in the colour of the species that changes; filled = invades, open = lost
for kink in KINKS:
    K = kink['K']
    sp_c = (kink['gained'] or kink['lost'])[0]
    invades = bool(kink['gained'])
    ax.scatter([K[0]] * len(PAIRS), [K[sp] for sp in PAIRS], s=17, zorder=6,
               facecolor=C[sp_c] if invades else 'white', edgecolor=C[sp_c] if not invades else 'k',
               linewidth=1.1 if not invades else 0.5)
end = XP[:, -1]
for sp, dy in {1: 0.13, 5: -0.13, 2: 0.0, 4: 0.0}.items():
    ax.text(end[0] + 0.05, end[sp] + dy, f'$x_{sp + 1}$', color=C[sp], fontsize=7.5, ha='left', va='center')
handles = [Line2D([], [], color='0.3', lw=1.15, label=r'change of $A_{23}$'),
           Line2D([], [], color='0.45', lw=3.4, alpha=0.22, solid_capstyle='round', label=r'press on $r_2$'),
           FancyArrowPatch((0, 0), (1, 0), color=PRED_GREY, lw=0, label='predicted'),
           Line2D([], [], ls='none', marker='o', ms=4.2, mfc='0.55', mec='k', mew=0.5, label='invasion'),
           Line2D([], [], ls='none', marker='o', ms=4.2, mfc='white', mec='0.4', mew=1.1, label='extinction')]
def legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    """The 'predicted' legend entry: the same grey arrow as in the panel."""
    y = height / 2 - ydescent
    return FancyArrowPatch((-xdescent, y), (width - xdescent, y),
                           arrowstyle=ArrowStyle('simple', head_length=0.66, head_width=0.66, tail_width=0.34),
                           mutation_scale=7.0, color=PRED_GREY, lw=0, shrinkA=0, shrinkB=0)


ax.legend(handles=handles, handler_map={FancyArrowPatch: HandlerPatch(patch_func=legend_arrow)},
          loc='upper left', bbox_to_anchor=(0.30, 1.02), handletextpad=0.4, labelspacing=0.2,
          borderaxespad=0.1, handlelength=1.3, fontsize=6.5)
panel_label(fig, AX_B[0] - 0.42, TOPY, 'b', W, H)
panel_label(fig, SLOT['x'] + 0.02, TOPY, 'c', W, H)


def write_layout():
    cm = 2.54
    with open(os.path.join(HERE, 'prl_fig2_layout.tex'), 'w') as fh:
        fh.write('% GENERATED by prl_fig2.py\n'
                 f'\\def\\slotW{{{SLOT["w"] * cm:.4f}}}\\def\\slotH{{{SLOT["h"] * cm:.4f}}}% cm\n')
    with open(os.path.join(HERE, 'prl_assemble_layout.tex'), 'w') as fh:
        fh.write('% GENERATED by prl_fig2.py\n'
                 f'\\def\\figW{{{W:.4f}}}\\def\\figH{{{H:.4f}}}\\def\\slotX{{{SLOT["x"]:.4f}}}'
                 f'\\def\\slotY{{{SLOT["y"]:.4f}}}\n'
                 '\\def\\panels{build_fig2/fig2_panels.pdf}\\def\\sketch{build_fig2/prl_fig2_sketch.pdf}\n')


if __name__ == '__main__':
    fig.savefig(os.path.join(BUILD, 'fig2_panels.pdf'))
    write_layout()
    pdflatex('prl_fig2_sketch.tex', BUILD)
    pdflatex('prl_assemble.tex', BUILD)
    finish(os.path.join(BUILD, 'prl_assemble.pdf'), 'fig2')
