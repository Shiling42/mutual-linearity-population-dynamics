"""Fig. S3 -- the chaotic community of Fig. 3 in detail: time averages of cycles and chaos stay on the line
(two columns, 7.0 in).

Run:  python3 -B figure_scripts/figS3_chaos.py      (from any directory)
Data: data/exp_chaos.npz (code/exp_chaos.py) and data/crisis_scan_extended.csv (code/crisis_scan_extended.py), from
which the limits of the two bands of panel (c) are read.
Output: figures/figS3_chaos.pdf (one vector file, 7.0 in wide) and a 300-dpi PNG, rendered with pdftocairo
(matplotlib's own PNG is written instead if pdftocairo is not installed). No TikZ.

Layout (7.0 x 4.56 in): top row (a) (three stacked time series, shared time axis) | (b) (attractors in the (x2, x4)
plane, zoom inset in the empty upper right); bottom row (c) (averages against theta, full width). Encodings:
species i -> C[i-1] in (a) and (c); the regime of the dynamics is shown by the symbol shape (circle: fixed point,
square: limit cycle, triangle: chaos) and, in (a) and (b), by the Okabe-Ito colours that no species uses here (black,
sky blue, orange); in (c) the symbols carry the species colours and a small key explains the shapes and the two
backgrounds.

Only the presentation is done here; the data come unchanged from the two experiments. For drawing only: the Hopf
point (dotted line in c) is located as the zero of the largest real part of the eigenvalues of the Jacobian at the
interior equilibrium (theta_H = -0.3738), the two-species equilibrium of species 3 and 4 (c) is solved from the
model, and in (b) the line of interior equilibria is drawn over its whole feasible range (all abundances positive),
which in the (x2, x4) plane runs from axis to axis. The regime is read off the successive maxima of x2 on the stored
tail (they repeat on a cycle); a threshold on the Lyapunov exponent would class the weak chaos at theta = 0.15 as a
limit cycle.
"""
import os
import sys
import shutil
import subprocess
import numpy as np
from scipy.optimize import brentq
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch, Patch
from matplotlib.legend_handler import HandlerTuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))                      # the folder that holds code/, data/, figures/
sys.path.insert(0, os.path.join(ROOT, 'code'))                         # glv_tools.py, ml_core.py
from figstyle import plt, C, TWO_COL  # noqa: E402
from glv_tools import VANO_R as r, VANO_A as a0, repeating_maxima  # noqa: E402

DATA = os.path.join(ROOT, 'data')                                       # data, read only
OUT = os.path.join(ROOT, 'figures')                                     # the figures
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------------------------- data
d = np.load(os.path.join(DATA, 'exp_chaos.npz'))
thetas, Cinv, x0 = d['thetas'], d['C'], d['x0']
k, l = 0, 2                                # a_13 -> a_13 + theta: species 1 is modified, through species 3
nT = len(thetas)
xavg = np.array([d[f'xavg_{i}'] for i in range(nT)])
lyap = np.array([float(d[f'lyap_{i}']) for i in range(nT)])
remax = np.array([float(d[f'remax_{i}']) for i in range(nT)])


regime = ['fixed point' if remax[i] < 0 else ('limit cycle' if repeating_maxima(d[f'tail_{i}'][1]) else 'chaos')
          for i in range(nT)]
SHOW = [0, 3, 6]                           # theta = -0.45 (fixed point), -0.20 (limit cycle), +0.08 (chaos)


def line(th):
    """Interior equilibrium x*(theta) on the straight line [Eqs. (4) and (6) with theta -> -theta, as in exp_chaos.py]."""
    return x0 - th * x0[l] / (1 + th * Cinv[l, k]) * Cinv[:, k]


def remax_of(th):
    a = a0.copy(); a[k, l] += th
    xeq = np.linalg.solve(a, np.ones(4))
    return np.max(np.linalg.eigvals(-np.diag(r * xeq) @ a).real)


TH_HOPF = brentq(remax_of, thetas[1], thetas[2], xtol=1e-10)          # -0.3738 (presentation only)
X34 = np.r_[0.0, 0.0, np.linalg.solve(a0[2:, 2:], np.ones(2))]         # (0, 0, 0.634, 0.778); a_13 does not enter
# Limits of the two bands of panel (c), read from the extended scan (crisis_scan_extended.py: 12 initial conditions
# per theta, theta = 0.15 ... 0.30): the hatched band starts at the first theta at which a trajectory escapes to the
# two-species state, the shaded band at the first theta from which every trajectory escapes.
SCAN = np.genfromtxt(os.path.join(DATA, 'crisis_scan_extended.csv'), delimiter=',', names=True)
_ths = np.unique(SCAN['theta'])
_stay = np.array([SCAN['stayed'][SCAN['theta'] == t].sum() for t in _ths])
_n = np.array([(SCAN['theta'] == t).sum() for t in _ths])
TH_TRANS = float(_ths[_stay < _n].min())
TH_COLL = float(min(t for i, t in enumerate(_ths) if not _stay[i:].any()))
TH_MAX = 0.30
print(f'bands of panel (c): escapes begin at theta = {TH_TRANS:.2f}; all trajectories escape from theta = {TH_COLL:.2f}')

# ----------------------------------------------------------------------------------------------- encodings
REG = ['fixed point', 'limit cycle', 'chaos']
REG_MK = {'fixed point': 'o', 'limit cycle': 's', 'chaos': '^'}
# regime colours: the Okabe-Ito colours that no species uses in this four-species figure (species i -> C[i-1])
REG_COL = {'fixed point': C[6], 'limit cycle': C[5], 'chaos': C[4]}
REG_MS = {'o': 4.3, 's': 3.9, '^': 4.7}    # marker sizes (pt) that look equally large
SERIES_W = 1.0                             # species lines (1.2 pt in Fig. S2; thinner here: dense oscillations)
KEY_FS, LABEL_FS, NOTE_FS = 7.0, 7.0, 6.5   # keys and direct labels 7 pt, notes 6.5 pt
EQ_GREY, EQ_W = '0.25', 1.2                # the line of equilibria: the boldest line of B (attractors: thin lines)
HOPF = dict(color='0.45', lw=0.6, ls=(0, (1, 1.6)))   # as the change lines of Fig. 2


def fmt(x, nd=2):
    return f'{x:+.{nd}f}'.replace('-', '\u2212')


def mk(ax, x, y, reg, ms_scale=1.0, **kw):
    """Regime symbol: shape and fill colour of the regime, thin black edge (A labels, B, inset of B)."""
    m = REG_MK[reg]
    return ax.plot(x, y, ls='none', marker=m, ms=REG_MS[m] * ms_scale, mfc=REG_COL[reg], mec='k', mew=0.5, **kw)


# ----------------------------------------------------------------------------------------------- layout (inches)
W, H = TWO_COL, 4.56
LEFT_X = 0.47                              # left edge of the axes of (a) and (c), as in Figs. S2 and S4
A_W = 2.60                                 # A: room on the right for the direct species labels
C_W = W - 0.12 - LEFT_X                # (c) spans the bottom row
RIGHT_X = 3.97                             # left edge of the axes of B
R_W = W - 0.12 - RIGHT_X                   # 0.12 in right margin: room for the last tick label
BOT, BOT_H = 0.40, 1.33                    # bottom row (C)
TOP_Y, TOP_H = BOT + BOT_H + 0.64, 1.95    # top row (A, B)
A_GAP = 0.07                               # gap between the three rows of A
LETTER_DX, TITLE_DX, HEAD_DY = -0.44, 0.24, 0.10   # as in Figs. S2 and S4

fig = plt.figure(figsize=(W, H))


def add_axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def head(x_axes, y_axes_top, letter, title):
    """Panel label "(a)" in regular weight + short sentence-case title on one baseline."""
    y = (y_axes_top + HEAD_DY) / H
    fig.text((x_axes + LETTER_DX) / W, y, f'({letter.lower()})', fontsize=8.5, fontweight='normal', ha='left',
             va='baseline')                                   # "(a)" in regular weight
    fig.text((x_axes + LETTER_DX + TITLE_DX) / W, y, title, fontsize=7.5, ha='left', va='baseline')


def screen_angle(ax, p0, p1):
    q0, q1 = ax.transData.transform(p0), ax.transData.transform(p1)
    return np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0]))


def offset_points(ax, xy, dx_pt, dy_pt):
    disp = ax.transData.transform(xy)
    return ax.transData.inverted().transform(disp + np.array([dx_pt, dy_pt]) * fig.dpi / 72.0)


# ---------------- A: three kinds of dynamics (shared time axis) ----------------
ROW_H = (TOP_H - 2 * A_GAP) / 3
T_WIN = 300.0
axA = []
for row, i in enumerate(SHOW):
    ax = add_axes(LEFT_X, TOP_Y + TOP_H - (row + 1) * ROW_H - row * A_GAP, A_W, ROW_H)
    axA.append(ax)
    tail, tt = d[f'tail_{i}'], d[f'ttail_{i}']
    t = tt - tt[0]
    sel = t <= T_WIN + 1e-9
    ax.axhline(xavg[i][2], color=C[2], lw=0.8, ls=(0, (3, 2)), zorder=1)       # time average of species 3
    for sp in range(4):
        ax.plot(t[sel], tail[sp, sel], color=C[sp], lw=SERIES_W, zorder=2)
    ax.set_xlim(0, T_WIN); ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.4, 0.8]); ax.set_xticks([0, 100, 200, 300])
    if row < 2:                            # shared time axis: ticks and labels only on the bottom row
        ax.tick_params(axis='x', labelbottom=False, length=0)
    reg = regime[i]
    label = f'{reg}, \u03b8 = {fmt(thetas[i])}' + (f', \u03bb$_{{\\max}}$ = {lyap[i]:.3f}' if reg == 'chaos' else '')
    mk(ax, [0.030], [0.865], reg, transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.052, 0.865, label, transform=ax.transAxes, fontsize=NOTE_FS, color='0.2', ha='left',
            va='center_baseline')
axA[2].set_xlabel('time')
axA[1].set_ylabel(r'abundance $x_i$')
# species, labelled once, at the ends of the flat lines of the fixed point (sp. 1 and 4 nearly coincide: nudged)
yt = axA[0].get_yaxis_transform()
xe = xavg[SHOW[0]]
for sp, dy in zip(range(4), [-0.075, 0.03, -0.03, 0.075]):
    axA[0].text(1.025, xe[sp] + dy, f'sp. {sp + 1}', transform=yt, color=C[sp], fontsize=LABEL_FS, ha='left',
                va='center')
yt = axA[2].get_yaxis_transform()
axA[2].text(1.025, xavg[SHOW[2]][2], r'$\langle x_3\rangle$', transform=yt, color=C[2], fontsize=LABEL_FS, ha='left',
            va='center')
head(LEFT_X, TOP_Y + TOP_H, 'A', 'Three kinds of dynamics')

# ---------------- B: attractors in the (x2, x4) plane, their time averages and the line of equilibria ----------------
p, q = 1, 3                                 # x2 (horizontal), x4 (vertical)
axB = add_axes(RIGHT_X, TOP_Y, R_W, TOP_H)
BX, BY = (0.0, 0.90), (0.0, 0.75)
axB.set_xlim(*BX); axB.set_ylim(*BY)
# the whole feasible line: x* = x0 - s C_.k with all abundances positive
cvec = Cinv[:, k]
s_lo = max(x0[j] / cvec[j] for j in range(4) if cvec[j] < 0)
s_hi = min(x0[j] / cvec[j] for j in range(4) if cvec[j] > 0)
Ls = np.array([x0 - s * cvec for s in np.linspace(s_lo, s_hi, 400)])
axB.plot(Ls[:, p], Ls[:, q], color=EQ_GREY, lw=EQ_W, zorder=4, solid_capstyle='butt')
tail = d['tail_6']
axB.plot(tail[p, :8000], tail[q, :8000], color=REG_COL['chaos'], lw=0.5, alpha=0.7, zorder=1)   # 400 time units
tail = d['tail_3']
axB.plot(tail[p], tail[q], color=REG_COL['limit cycle'], lw=0.8, alpha=0.9, zorder=2)
for i in SHOW:
    mk(axB, [xavg[i][p]], [xavg[i][q]], regime[i], ms_scale=1.15, zorder=6)
axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); axB.set_yticks([0, 0.2, 0.4, 0.6])
axB.set_xlabel(r'abundance $x_2$'); axB.set_ylabel(r'abundance $x_4$')
# label of the line, parallel to it on its upper side, where the chaotic attractor stays within 7 pt of the line
xl0, xl1 = 0.13, 0.35
pa, pb = (np.interp(xl0, Ls[::-1, p], Ls[::-1, q])), (np.interp(xl1, Ls[::-1, p], Ls[::-1, q]))
pa, pb = np.array([xl0, pa]), np.array([xl1, pb])
ang = screen_angle(axB, pa, pb)
pos = offset_points(axB, 0.5 * (pa + pb), -8.5 * np.sin(np.radians(ang)), 8.5 * np.cos(np.radians(ang)))
axB.text(*pos, r'line of equilibria $\mathbf{x}^*(\theta)$', rotation=ang, rotation_mode='anchor', ha='center',
         va='bottom', fontsize=LABEL_FS, color=EQ_GREY)
# key of the regimes (lower left, free of data): attractor (line) and time average (symbol); the theta values of the
# three attractors are those of A, and the same symbols mark all nine averages in the inset
hB = [(Line2D([], [], color=REG_COL[reg], lw=1.2 if reg != 'fixed point' else 0, alpha=0.9),
       Line2D([], [], ls='none', marker=REG_MK[reg], ms=REG_MS[REG_MK[reg]], mfc=REG_COL[reg], mec='k', mew=0.5))
      for reg in REG]
axB.legend(hB, REG, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower left', fontsize=KEY_FS,
           handlelength=1.5, handletextpad=0.5, labelspacing=0.25, borderaxespad=0.15)
# zoom inset in the empty upper right, same scale factor on both axes (the line keeps its slope)
IB = (0.585, 0.415, 0.885, 0.685)          # inset box in data coordinates of B: x0, y0, x1, y1
axin = axB.inset_axes([IB[0], IB[1], IB[2] - IB[0], IB[3] - IB[1]], transform=axB.transData)
bx0, bx1 = xavg[:, p].min(), xavg[:, p].max()
by0, by1 = xavg[:, q].min(), xavg[:, q].max()
mag = min((IB[2] - IB[0]) / ((bx1 - bx0) * 1.30), (IB[3] - IB[1]) / ((by1 - by0) * 1.30))
cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
zx, zy = 0.5 * (IB[2] - IB[0]) / mag, 0.5 * (IB[3] - IB[1]) / mag
axin.set_xlim(cx - zx, cx + zx); axin.set_ylim(cy - zy, cy + zy)
axin.plot(Ls[:, p], Ls[:, q], color=EQ_GREY, lw=EQ_W, zorder=1)
for i in range(nT):
    mk(axin, [xavg[i][p]], [xavg[i][q]], regime[i], zorder=5)
axin.set_xticks([]); axin.set_yticks([])
for s in axin.spines.values():
    s.set_visible(True); s.set_color('0.6'); s.set_linewidth(0.6)
# theta at the two ends of the row of averages: the last value right of its symbol (above the falling line), the
# first one left of its symbol (below the line)
for i_end, off, ha, va in [(0, (-5.0, -2.0), 'right', 'top'), (nT - 1, (5.0, 1.5), 'left', 'bottom')]:
    axin.annotate(f'\u03b8 = {fmt(thetas[i_end])}', (xavg[i_end][p], xavg[i_end][q]), xytext=off,
                  textcoords='offset points', ha=ha, va=va, fontsize=NOTE_FS, color='0.2')
axB.text(0.5 * (IB[0] + IB[2]), IB[3] + 0.008, 'time averages, all nine \u03b8', ha='center', va='bottom',
         fontsize=NOTE_FS, color='0.2')
# zoom rectangle and two connectors, light, behind the attractors
ZOOM_GREY = '0.62'
axB.add_patch(Rectangle((cx - zx, cy - zy), 2 * zx, 2 * zy, fill=False, ec=ZOOM_GREY, lw=0.5, zorder=5))
for (xa, ya), (xb, yb) in [((cx - zx, cy + zy), (0, 1)), ((cx + zx, cy - zy), (1, 0))]:
    axB.add_artist(ConnectionPatch((xa, ya), (xb, yb), coordsA=axB.transData, coordsB=axin.transAxes,
                                   color=ZOOM_GREY, lw=0.5, zorder=0.5))
head(RIGHT_X, TOP_Y + TOP_H, 'B', 'Different attractors, one line')

# ---------------- C: time averages against theta ----------------
axC = add_axes(LEFT_X, BOT, C_W, BOT_H)
plt.rcParams['hatch.linewidth'] = 0.5
HATCH, HATCH_GREY, SHADE = '//////', '0.68', '0.90'
axC.axvspan(TH_TRANS, TH_COLL, facecolor='none', edgecolor=HATCH_GREY, hatch=HATCH, lw=0, zorder=0)
axC.axvspan(TH_COLL, TH_MAX, color=SHADE, lw=0, zorder=0)
axC.axvline(TH_HOPF, zorder=1, **HOPF)
ths_line = np.linspace(-0.5, TH_MAX, 300)
Lt = np.array([line(t) for t in ths_line])
for sp in range(4):
    axC.plot(ths_line, Lt[:, sp], color=C[sp], lw=SERIES_W, zorder=2)
for sp in [3, 2, 1, 0]:                    # species 1 last: its averages sit on top where sp. 1 and 4 coincide
    for i in range(nT):
        m = REG_MK[regime[i]]
        axC.plot([thetas[i]], [xavg[i][sp]], ls='none', marker=m, ms=REG_MS[m] * 0.92, mfc=C[sp], mec='k', mew=0.5,
                 zorder=5)
# after the collapse: the equilibrium of species 3 and 4; species 1 and 2 both at zero (alternating dashes)
for sp in [2, 3]:
    axC.plot([TH_TRANS, TH_MAX], [X34[sp]] * 2, color=C[sp], lw=1.6, zorder=3, solid_capstyle='butt')
axC.plot([TH_TRANS, TH_MAX], [0, 0], color=C[0], lw=1.6, zorder=3, solid_capstyle='butt', clip_on=False)
axC.plot([TH_TRANS, TH_MAX], [0, 0], color=C[1], lw=1.6, zorder=3.1, ls=(0, (2.2, 2.2)), dash_capstyle='butt',
         clip_on=False)
axC.set_xlim(-0.5, TH_MAX); axC.set_ylim(0, 0.9)
axC.set_xticks([-0.4, -0.2, 0, 0.2]); axC.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
axC.set_xlabel(r'$\theta$ (change of $a_{13}$)'); axC.set_ylabel(r'time average $\langle x_i\rangle$')
axC.text(TH_HOPF - 0.012, 0.865, 'Hopf', ha='right', va='center_baseline', fontsize=NOTE_FS, color='0.3')
axC.text(TH_TRANS - 0.012, 0.035, 'sp. 1, 2 lost', ha='right', va='center_baseline', fontsize=NOTE_FS, color='0.3')
# key: the three kinds of attractor (marker shape) and the two kinds of collapse (background)
hC = [Line2D([], [], ls='none', marker=REG_MK[reg], ms=REG_MS[REG_MK[reg]], mfc='white', mec='k', mew=0.6)
      for reg in REG]
hC += [Patch(facecolor='white', edgecolor=HATCH_GREY, hatch=HATCH, lw=0), Patch(facecolor=SHADE, edgecolor='none')]
axC.legend(hC, REG + ['transient chaos', 'collapse'], ncol=2, loc='upper left', bbox_to_anchor=(TH_HOPF + 0.02, 0.9),
           bbox_transform=axC.transData, fontsize=KEY_FS, handlelength=1.0, handleheight=0.9, handletextpad=0.4,
           labelspacing=0.2, columnspacing=1.2, borderaxespad=0.0, borderpad=0.1)
# species, labelled where the lines are apart (between the markers at theta = -0.30 and -0.20)
for sp, dy, va in [(1, 0.025, 'bottom'), (3, 0.022, 'bottom'), (0, -0.022, 'top'), (2, -0.022, 'top')]:
    axC.text(-0.25, line(-0.25)[sp] + dy, f'sp. {sp + 1}', color=C[sp], fontsize=LABEL_FS, ha='center', va=va)
head(LEFT_X, BOT + BOT_H, 'C', 'Averages across the bifurcations')

# ----------------------------------------------------------------------------------------------- output
if __name__ == '__main__':
    out_pdf = os.path.join(OUT, 'figS3_chaos.pdf')
    fig.savefig(out_pdf)
    if shutil.which('pdftocairo'):
        subprocess.run(['pdftocairo', '-png', '-r', '300', '-singlefile', out_pdf, out_pdf[:-4]], check=True)
    else:
        fig.savefig(out_pdf[:-4] + '.png', dpi=300)
    print('saved', os.path.relpath(out_pdf, ROOT), 'and', os.path.relpath(out_pdf[:-4] + '.png', ROOT))
    print('regimes:', ', '.join(f'{fmt(t)}: {g}' for t, g in zip(thetas, regime)))
    print(f'Hopf at theta = {TH_HOPF:.4f}; zoom magnification in B: x{mag:.2f}; '
          f'max |<x> - x*(theta)| = {max(np.abs(xavg[i] - line(thetas[i])).max() for i in range(nT)):.1e}')
