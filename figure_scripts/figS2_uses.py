"""Fig. S2 -- uses of the lines: prediction from two measurements, kinks that reveal changes in unmonitored species,
extinction before divergence, press responses without pressing (two columns, 7.0 in).

Run:  python3 -B figure_scripts/figS2_uses.py      (from any directory)
Data: data/exp_stepwise_example.npz (code/exp_stepwise.py) and data/exp_use.npz (code/exp_use_and_scope.py).
Output: figures/figS2_uses.pdf (one vector file, 7.0 in wide) and a 300-dpi PNG, rendered with pdftocairo
(matplotlib's own PNG is written instead if pdftocairo is not installed). No TikZ.

Layout (7.0 x 4.30 in, 2 x 2): top row (a) | (b), bottom row (c) | (d). Species i -> C[i-1] throughout.

Only the presentation is done here; the data come unchanged from the two experiments. For drawing only: (i) as in
Fig. 2, each change of composition is located exactly between two grid values of theta (the kink point is the
intersection of the two straight branches that meet there), so the path of (b) is drawn continuous through its kinks
and the predicted lines of (a) span the whole branch; (ii) in (c) the branch is drawn through the exact points where
it ends (species 2 reaches zero; species 6 invades), evaluated with the same ratio of affine functions of theta that
gives the dashed continuation (it reproduces the data on the branch to 1e-15).
"""
import os
import sys
import shutil
import subprocess
import numpy as np
from scipy.optimize import brentq
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))                      # the folder that holds code/, data/, figures/
sys.path.insert(0, os.path.join(ROOT, 'code'))                         # glv_tools.py, ml_core.py
from figstyle import plt, C, TWO_COL  # noqa: E402
from glv_tools import segments  # noqa: E402

DATA = os.path.join(ROOT, 'data')                                       # data, read only
OUT = os.path.join(ROOT, 'figures')                                     # the figures
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------------------------- data
d = np.load(os.path.join(DATA, 'exp_stepwise_example.npz'))
X, TH = d['X'], d['thetas']
SUPP = [tuple(int(v) for v in s) for s in d['SUPP']]
r, A, k, l = d['r'], d['A'], int(d['k']), int(d['l'])      # k = 1 (species 2, modified), l = 2 (species 3)
u = np.load(os.path.join(DATA, 'exp_use.npz'))
P1, P2, i1, i2, s0, s1 = u['P1'], u['P2'], int(u['i1']), int(u['i2']), int(u['s0']), int(u['s1'])
slopes_int, dp = u['slopes_int'], u['dp']
n = X.shape[0]
segs = segments(SUPP)                                       # branches of fixed composition (index ranges)

# ---------------- exact change points (presentation only; as in prl_fig2.py)
KINKS = []
for q in range(1, len(segs)):
    (p0, p1), (c0, c1) = segs[q - 1], segs[q]
    p, du = X[:, p0], X[:, p1 - 1] - X[:, p0]
    c, dv = X[:, c0], X[:, c1 - 1] - X[:, c0]
    if np.linalg.norm(du) < 1e-12:                          # the control has no effect on this branch: a point
        K = p.copy()
    else:
        a_, _ = np.linalg.lstsq(np.column_stack([du, -dv]), c - p, rcond=None)[0]
        K = p + a_ * du
    assert np.linalg.norm(c + (K - c) @ dv / (dv @ dv) * dv - K) < 1e-9
    th_K = -(r[k] + A[k] @ K) / K[l]                        # g_k(K) = 0 with A_kl -> A_kl + theta
    Sb = [i for i in range(n) if SUPP[p0][i]]
    Sa = [i for i in range(n) if SUPP[c0][i]]
    KINKS.append(dict(theta=th_K, K=K, gained=sorted(set(Sa) - set(Sb)), lost=sorted(set(Sb) - set(Sa))))
TP = [TH[segs[0][0]:segs[0][1]]]                            # the whole path with the exact kinks inserted
XP = [X[:, segs[0][0]:segs[0][1]]]
for q, kink in enumerate(KINKS):
    c0, c1 = segs[q + 1]
    TP += [[kink['theta']], TH[c0:c1]]
    XP += [kink['K'][:, None], X[:, c0:c1]]
TP, XP = np.concatenate(TP), np.hstack(XP)


def event_text(kink):
    return f'sp. {kink["gained"][0] + 1} invades' if kink['gained'] else f'sp. {kink["lost"][0] + 1} lost'


# ---------------- A: the branch of the two measurements (species 4 absent) and its exact ends
qA = segs.index((s0, s1))
KA0, KA1 = KINKS[qA - 1]['K'], KINKS[qA]['K']               # sp. 4 lost ... sp. 5 lost
SP_A = [j for j in range(n) if SUPP[s0][j] and j != 0]      # predicted species: 2, 3, 5, 6
slope_A = (P2 - P1) / (P2[0] - P1[0])                        # the line through the two measurements
predict = lambda x1, j: P1[j] + (x1 - P1[0]) * slope_A[j]
x1_line = np.array([KA0[0], KA1[0]])
err_A = np.max(np.abs(np.array([predict(X[0, s0:s1], j) for j in range(n)]) - X[:, s0:s1]))

# ---------------- C: x*(theta) on the branch without species 6 (a ratio of affine functions), its pole and exact ends
a, b = segs[1]
S = [j for j in range(n) if SUPP[a][j]]
th0 = TH[b - 1]
At = A.copy(); At[k, l] += th0
Bm = np.linalg.inv(At[np.ix_(S, S)]); x0S = X[S, b - 1]
ik, il = S.index(k), S.index(l)
POLE = th0 - 1.0 / Bm[il, ik]


def x_branch(t, sp):
    j = S.index(sp)
    f = (t - th0) * x0S[il] / (1 + (t - th0) * Bm[il, ik])
    return x0S[j] - f * Bm[j, ik]


err_C = max(np.max(np.abs(x_branch(TH[a:b], sp) - X[sp, a:b])) for sp in S)
TH_EXT = brentq(lambda t: x_branch(t, 1), POLE + 1e-6, TH[b - 1], xtol=1e-14)   # species 2 reaches zero (moving left)
TH_INV = KINKS[1]['theta']                                  # species 6 invades (right end of the branch)
assert abs(TH_EXT - KINKS[0]['theta']) < 1e-9

# ---------------- D: slopes from the interaction sweep and ratios of press responses from an actual press
SP_D = [j for j in range(n) if SUPP[s0][j] and j != 0]
sl_int = slopes_int[[j - 1 for j in SP_D]]
sl_press = (dp[1:] / dp[0])[[j - 1 for j in SP_D]]

# ----------------------------------------------------------------------------------------------- layout (inches)
W, H = TWO_COL, 4.30
LEFT_X, L_W = 0.47, 2.90                   # (a), (c): left edge as in Figs. S3 and S4
RIGHT_X = 3.97                             # (b), (d)
R_W = W - 0.12 - RIGHT_X
BOT, ROW_H = 0.40, 1.50
TOP_Y = BOT + ROW_H + 0.64
LETTER_DX, TITLE_DX, HEAD_DY = -0.44, 0.24, 0.10   # as in Figs. S3 and S4
LINE_W = 1.2                               # species lines (Fig. 2)
SPECIES_FS, NOTE_FS, KEY_FS = 7.5, 6.5, 7.0       # line-end species labels, notes, keys
NOTE_GREY = '0.15'
CIRCLE = dict(s=15, facecolor='white', edgecolor='k', linewidth=0.6, zorder=6)   # changes of composition (Fig. 2)

fig = plt.figure(figsize=(W, H))


def add_axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def head(x_axes, y_axes_top, letter, title):
    """Panel label "(a)" in regular weight + short sentence-case title on one baseline."""
    y = (y_axes_top + HEAD_DY) / H
    fig.text((x_axes + LETTER_DX) / W, y, f'({letter.lower()})', fontsize=8.5, fontweight='normal', ha='left',
             va='baseline')                                   # "(a)" in regular weight
    fig.text((x_axes + LETTER_DX + TITLE_DX) / W, y, title, fontsize=7.5, ha='left', va='baseline')


def offset_points(ax, xy, dx_pt, dy_pt):
    disp = ax.transData.transform(xy)
    return ax.transData.inverted().transform(disp + np.array([dx_pt, dy_pt]) * fig.dpi / 72.0)


# ---------------- A: two measurements predict the whole branch ----------------
axA = add_axes(LEFT_X, TOP_Y, L_W, ROW_H)
TRUE_EVERY = 8
TRUE_PHASE = {1: 0, 2: 0, 4: 0, 5: TRUE_EVERY // 2}         # x2 and x6 nearly coincide: their true values alternate
AY = (-0.1, 3.3)
for P in (P1, P2):                                          # the two settings at which everything is measured:
    axA.plot([P[0]] * 2, [AY[0], max(P[SP_A]) + 0.22], color='0.80', lw=0.6, zorder=0)   # drop lines to x1
for j in SP_A:
    axA.plot(x1_line, predict(x1_line, j), color=C[j], lw=LINE_W, zorder=2, solid_capstyle='butt')
    sel = slice(s0 + TRUE_PHASE[j], s1, TRUE_EVERY)
    axA.scatter(X[0, sel], X[j, sel], s=11, facecolor='white', edgecolor=C[j], linewidth=0.7, zorder=3)
for j in SP_A:
    axA.scatter([P1[0], P2[0]], [P1[j], P2[j]], s=22, color='k', edgecolor='white', linewidth=0.5, zorder=5)
axA.set_xlim(1.0, 2.38); axA.set_ylim(*AY)
axA.set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.2]); axA.set_yticks([0, 1, 2, 3])
axA.set_xlabel(r'abundance $x_1$ (measured)'); axA.set_ylabel(r'abundance $x_i$')
A_END = {1: 0.10, 5: -0.10, 2: 0.0, 4: 0.0}               # x2 and x6 meet at the end of the branch: split them
for j in SP_A:
    axA.text(x1_line[1] + 0.03, predict(x1_line[1], j) + A_END[j], f'$x_{j + 1}$', color=C[j],
             fontsize=SPECIES_FS, ha='left', va='center')
hA = [Line2D([], [], ls='none', marker='o', ms=4.7, mfc='k', mec='white', mew=0.5),
      Line2D([], [], color='0.3', lw=LINE_W),
      Line2D([], [], ls='none', marker='o', ms=3.3, mfc='white', mec='0.3', mew=0.7)]
axA.legend(hA, ['two measurements', r'predicted from $x_1$ alone', 'true value'], loc='upper left', fontsize=KEY_FS,
           handlelength=1.5, handletextpad=0.5, labelspacing=0.25, borderaxespad=0.1)
head(LEFT_X, TOP_Y + ROW_H, 'A', 'Prediction from two measurements')

# ---------------- B: kinks in a monitored pair reveal changes in unmonitored species ----------------
axB = add_axes(RIGHT_X, TOP_Y, R_W, ROW_H)
axB.plot(XP[0], XP[2], color=C[2], lw=LINE_W, zorder=3, solid_joinstyle='miter')
for kink in KINKS:
    axB.scatter([kink['K'][0]], [kink['K'][2]], **CIRCLE)
axB.set_xlim(0.45, 2.75); axB.set_ylim(0.4, 4.0)
axB.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5]); axB.set_yticks([1, 2, 3])
axB.set_xlabel(r'abundance $x_1$ (monitored)'); axB.set_ylabel(r'abundance $x_3$ (monitored)')
B_EVENT = {0: (9, 0, 'left', 'center'), 1: (7, 0, 'left', 'center'), 2: (7, 2.5, 'left', 'bottom'),
           3: (0, 7, 'center', 'bottom')}                   # offsets in points from the kink
for q, (dx, dy, ha, va) in B_EVENT.items():
    K = KINKS[q]['K']
    axB.annotate(event_text(KINKS[q]), (K[0], K[2]), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va,
                 fontsize=NOTE_FS, color=NOTE_GREY)
# direction of the sweep (invades / lost refer to increasing theta): arrowheads half-way along two branches
for qa, qb in [(0, 1), (2, 3)]:
    pa_, pb_ = KINKS[qa]['K'][[0, 2]], KINKS[qb]['K'][[0, 2]]
    mid, step = 0.5 * (pa_ + pb_), 0.02 * (pb_ - pa_)
    axB.annotate('', xy=mid + step, xytext=mid - step, zorder=4,
                 arrowprops=dict(arrowstyle='-|>,head_length=0.55,head_width=0.28', color=C[2], lw=0,
                                 shrinkA=0, shrinkB=0, mutation_scale=9))
K2, K3 = KINKS[2]['K'][[0, 2]], KINKS[3]['K'][[0, 2]]
q0_, q1_ = axB.transData.transform(K2), axB.transData.transform(K3)
ang = np.degrees(np.arctan2(q1_[1] - q0_[1], q1_[0] - q0_[0]))
pos = offset_points(axB, 0.5 * (K2 + K3), 6.0 * np.sin(np.radians(ang)), -6.0 * np.cos(np.radians(ang)))
axB.text(*pos, r'increasing $\theta$', rotation=ang, rotation_mode='anchor', ha='center', va='top',
         fontsize=NOTE_FS, color=NOTE_GREY)
head(RIGHT_X, TOP_Y + ROW_H, 'B', 'Kinks reveal changes in unmonitored species')

# ---------------- C: hyperbolic along a branch; the extinction comes before the divergence ----------------
axC = add_axes(LEFT_X, BOT, L_W, ROW_H)
CX, C_END = (-1.1, 0.52), 0.40
tt = np.linspace(POLE + 1e-4, C_END, 4000)
tb = np.r_[TH_EXT, TH[a:b], TH_INV]
axC.axhline(0, color='0.75', lw=0.5, zorder=0)
axC.axvline(POLE, color='0.45', lw=0.7, ls=(0, (1, 1.6)), zorder=1)
for sp in [1, 4]:
    axC.plot(tt, x_branch(tt, sp), color=C[sp], lw=0.9, ls=(0, (3, 2)), zorder=2)
    yb = np.r_[x_branch(TH_EXT, sp), X[sp, a:b], x_branch(TH_INV, sp)]
    axC.plot(tb, yb, color=C[sp], lw=2.4, zorder=3, solid_capstyle='butt')
axC.scatter([TH_EXT], [0.0], **CIRCLE)
axC.set_xlim(*CX); axC.set_ylim(-1.5, 7.0)
axC.set_xticks([-1.0, -0.5, 0.0]); axC.set_xticklabels(['−1', '−0.5', '0'])
axC.set_yticks([0, 2, 4, 6])
axC.set_xlabel(r'$\theta$ (change of $A_{23}$)'); axC.set_ylabel(r'abundance $x_i$')
axC.text(POLE + 0.03, 6.9, 'asymptote\n(divergence)', ha='left', va='top', fontsize=NOTE_FS, color=NOTE_GREY,
         linespacing=1.15)
axC.annotate('sp. 2 reaches zero first', (TH_EXT, 0.0), xytext=(-6, 3), textcoords='offset points', ha='right',
             va='bottom', fontsize=NOTE_FS, color=NOTE_GREY)
axC.text(-0.15, 1.75, 'branch\n(sp. 6 absent)', ha='center', va='center', fontsize=NOTE_FS, color=NOTE_GREY,
         linespacing=1.15)
for sp in [1, 4]:
    axC.text(C_END + 0.025, x_branch(C_END, sp), f'$x_{sp + 1}$', color=C[sp], fontsize=SPECIES_FS, ha='left',
             va='center')
head(LEFT_X, BOT + ROW_H, 'C', 'Extinction precedes divergence')

# ---------------- D: press responses without pressing ----------------
axD = add_axes(RIGHT_X, BOT, R_W, ROW_H)
xD = np.arange(len(SP_D))
BW, BGAP = 0.27, 0.035                    # bar width and gap within a pair (data units)
for q, j in enumerate(SP_D):
    axD.bar(xD[q] - BW / 2 - BGAP / 2, sl_int[q], width=BW, color=C[j], edgecolor=C[j], lw=1.0, zorder=2)
    axD.bar(xD[q] + BW / 2 + BGAP / 2, sl_press[q], width=BW, color='white', edgecolor=C[j], lw=1.0, zorder=2)
axD.axhline(0, color='0.3', lw=0.6, zorder=3)
axD.set_xlim(-0.6, len(SP_D) - 0.4); axD.set_ylim(-1.5, 2.1)
axD.set_xticks(xD); axD.set_xticklabels([f'$x_{j + 1}$' for j in SP_D], fontsize=SPECIES_FS)
for t, j in zip(axD.get_xticklabels(), SP_D):
    t.set_color(C[j])
axD.tick_params(axis='x', length=0, pad=3)
axD.set_yticks([-1, 0, 1])
axD.set_ylabel(r'slope $\Delta x_j/\Delta x_1$')
hD = [Patch(facecolor='0.35', edgecolor='0.35', lw=1.0), Patch(facecolor='white', edgecolor='0.35', lw=1.0)]
axD.legend(hD, ['interaction sweep', 'actual press on sp. 2'], loc='upper center', ncol=2, fontsize=KEY_FS,
           handlelength=1.2, handleheight=0.9, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.1)
head(RIGHT_X, BOT + ROW_H, 'D', 'Press responses without pressing')

# ----------------------------------------------------------------------------------------------- output
if __name__ == '__main__':
    out_pdf = os.path.join(OUT, 'figS2_uses.pdf')
    fig.savefig(out_pdf)
    if shutil.which('pdftocairo'):
        subprocess.run(['pdftocairo', '-png', '-r', '300', '-singlefile', out_pdf, out_pdf[:-4]], check=True)
    else:
        fig.savefig(out_pdf[:-4] + '.png', dpi=300)
    print('saved', os.path.relpath(out_pdf, ROOT), 'and', os.path.relpath(out_pdf[:-4] + '.png', ROOT))
    print(f'A: two measurements at theta = {TH[i1]:.2f}, {TH[i2]:.2f}; max prediction error on the branch '
          f'{err_A:.1e}')
    print(f'C: pole (asymptote) at theta = {POLE:.4f}; species 2 reaches zero at theta = {TH_EXT:.4f}; branch ends '
          f'at theta = {TH_INV:.4f} (sp. 6 invades); x*(theta) of the branch against the data: {err_C:.1e}')
    print('D: slopes (interaction sweep) =', np.round(sl_int, 4), '; press ratios =', np.round(sl_press, 4),
          '; max difference =', f'{np.max(np.abs(sl_int - sl_press)):.1e}')
    for kink in KINKS:
        print('  change at theta = %+.4f: %s; kink at (x1, x3) = (%.4f, %.4f)' % (kink['theta'], event_text(kink),
                                                                                kink['K'][0], kink['K'][2]))
