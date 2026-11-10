"""Fig. S4 -- several modified species and other models: one dimension per modified equation; consumer-resource and
replicator models (two columns, 7.0 in).

Run:  python3 -B figure_scripts/figS4_scope.py      (from any directory; needs pdflatex and pdftocairo)
                                                     add --debug to also outline the slot of panel (a)
  1. the script recomputes the small dimension-counting data set (seven-species community, make_community(7, seed=11,
     sigma=0.9), 300 random modifications of the equations of 1, 2, 3 species, as in code/exp_use_and_scope.py) and
     reads data/exp_use.npz (consumer-resource and replicator data);
  2. matplotlib draws the data panels (b)-(d) into figure_scripts/build_figS4/figS4_panels.pdf and leaves the slot of
     panel (a) empty (only its label and title are drawn here, so that all labels and titles share one typeface);
  3. it writes figS4_layout.tex (figure size and position and size of the slot of panel a);
  4. pdflatex compiles figS4_sketch.tex (panel a: TikZ sketch) and then figS4_assemble.tex (places the sketch into
     the slot of the matplotlib PDF), both in figure_scripts/build_figS4/;
  5. the result is figures/figS4_scope.pdf (7.0 in wide, one file) and a 300-dpi PNG.

Layout (7.0 x 3.40 in, 2 x 2): top row (a) (sketch) | (b) (singular values), bottom row (c) (consumer-resource) |
(d) (replicator).

Only the presentation is done here. For drawing only: (i) the four spectra of (b) are offset horizontally by up to
+-0.21 of an index step so that coinciding markers stay visible; (ii) the stars of (a), (c) and (d) mark the
unperturbed state (theta = 0): for (c) the equilibrium of the stored model (rr, AA), for (d) the rest point of the
payoff matrix regenerated exactly as in exp_use_and_scope.py (asserted to reproduce the stored rest points); both are
asserted to lie on the stored lines.
"""
import os
import sys
import shutil
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))                      # the folder that holds code/, data/, figures/
sys.path.insert(0, os.path.join(ROOT, 'code'))                         # glv_tools.py, ml_core.py
from figstyle import plt, C, TWO_COL  # noqa: E402
from ml_core import numerical_rank, glv_saturated_equilibrium  # noqa: E402
from glv_tools import make_community  # noqa: E402

DATA = os.path.join(ROOT, 'data')                                       # data, read only
OUT = os.path.join(ROOT, 'figures')                                     # the figures
os.makedirs(OUT, exist_ok=True)
BUILD = os.path.join(HERE, 'build_figS4')
os.makedirs(BUILD, exist_ok=True)

# ----------------------------------------------------------------------------------------------- data
u = np.load(os.path.join(DATA, 'exp_use.npz'))

# recompute the dimension-counting data (cheap) so the panel is self-contained -- as in code/exp_use_and_scope.py
r5, A5 = make_community(7, seed=11, sigma=0.9)
rng = np.random.default_rng(5)
spectra, ranks, nsamp = {}, {}, {}
for K in [[0], [0, 3], [0, 3, 5]]:
    pts = []
    for t in range(300):
        rt, At = r5.copy(), A5.copy()
        for kk in K:
            rt[kk] += rng.uniform(-0.2, 0.2); At[kk] += rng.uniform(-0.25, 0.25, size=7)
        xs, alive = glv_saturated_equilibrium(rt, At)
        if xs is not None and alive.all():
            pts.append(xs)
    P = np.array(pts).T
    rk, sv = numerical_rank(np.vstack([P, np.ones(P.shape[1])]), rtol=1e-10)
    spectra[len(K)], ranks[len(K)], nsamp[len(K)] = sv / sv[0], rk, P.shape[1]
Pcr, P2cr, Xr = u['Pcr'], u['P2cr'], u['Xr']
rk_cr2, sv_cr2 = numerical_rank(np.vstack([P2cr, np.ones(P2cr.shape[1])]), rtol=1e-10)   # rate shared by 2 eqs
sv_cr2 = sv_cr2 / sv_cr2[0]
rk_cr1, sv_cr1 = numerical_rank(np.vstack([Pcr, np.ones(Pcr.shape[1])]), rtol=1e-10)    # mortality of consumer 1
rk_rep, sv_rep = numerical_rank(Xr, rtol=1e-10)                                          # one payoff entry

# unperturbed states (theta = 0), for the stars of C and D (presentation only; see the docstring)
rr, AA = u['rr'], u['AA']
Y0 = np.linalg.solve(AA, -rr)
_rng = np.random.default_rng(1)
while True:
    PAY = _rng.uniform(0, 1, size=(4, 4))
    if np.all(np.linalg.solve(PAY, np.ones(4)) > 0):
        break
_X = []
for th in np.linspace(-0.6, 0.6, 61):
    Pt = PAY.copy(); Pt[0, 1] += th; xe = np.linalg.solve(Pt, np.ones(4))
    if np.all(xe > 0):
        _X.append(xe / xe.sum())
_X = np.array(_X).T
assert _X.shape == Xr.shape and np.max(np.abs(_X - Xr)) < 1e-12, 'payoff matrix not reproduced'
XR0 = np.linalg.solve(PAY, np.ones(4)); XR0 = XR0 / XR0.sum()


def line_residual(P, p):
    """Distance of point p from the affine line fitted through the columns of P (relative to its spread)."""
    m = P.mean(axis=1)
    U, s, _ = np.linalg.svd(P - m[:, None], full_matrices=False)
    d = (p - m) - U[:, 0] * (U[:, 0] @ (p - m))
    return np.linalg.norm(d) / s[0]


M_RES = 4                                   # y = (R_1..R_4, N_1..N_3)
assert line_residual(Pcr, Y0) < 1e-12 and Pcr[M_RES].min() < Y0[M_RES] < Pcr[M_RES].max()
assert line_residual(Xr, XR0) < 1e-12 and Xr[0].min() < XR0[0] < Xr[0].max()
# direction of the controls along the abscissae: increasing mortality m_1 / payoff a_12 lowers N_1 / x_1
assert Pcr[M_RES, 0] > Pcr[M_RES, -1] and Xr[0, 0] > Xr[0, -1]

# ----------------------------------------------------------------------------------------------- layout (inches)
LEFT_X, L_W = 0.47, 2.90                   # (a) (slot), (c) -- the column grid of Figs. S2 and S3
RIGHT_X = 3.97                             # B, D
W = TWO_COL
R_W = W - 0.12 - RIGHT_X
BOT, ROW_H = 0.40, 1.07                    # bottom row (C, D)
GAP = 0.60
TOP_Y, TOP_H = BOT + ROW_H + GAP, 1.07     # top row (A, B)
H = TOP_Y + TOP_H + 0.26                   # 3.40 in
LETTER_DX, TITLE_DX, HEAD_DY = -0.44, 0.24, 0.10   # as in Figs. S2 and S3
# slot of panel A (filled by the TikZ sketch): from the figure's left edge to the y-label zone of B, and from the
# header zone of the bottom row up to the top of B's axes
SLOT = dict(x=0.03, y=BOT + ROW_H + 0.30, w=RIGHT_X - 0.57 - 0.03, h=TOP_Y + TOP_H - (BOT + ROW_H + 0.30))
LINE_W = 1.2                               # model lines
LABEL_FS, NOTE_FS, KEY_FS = 7.5, 6.5, 7.0  # line-end labels, notes, curve labels
NOTE_GREY = '0.15'
STAR = dict(marker='*', s=52, color='k', edgecolor='white', linewidth=0.5, zorder=6)   # unperturbed (Fig. 1)

fig = plt.figure(figsize=(W, H))


def add_axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def head(x_axes, y_axes_top, letter, title):
    """Panel label "(a)" in regular weight + short sentence-case title on one baseline."""
    y = (y_axes_top + HEAD_DY) / H
    fig.text((x_axes + LETTER_DX) / W, y, f'({letter.lower()})', fontsize=8.5, fontweight='normal', ha='left',
             va='baseline')                                   # "(a)" in regular weight
    fig.text((x_axes + LETTER_DX + TITLE_DX) / W, y, title, fontsize=7.5, ha='left', va='baseline')


def direction_note(ax, x_tail, x_head, y, text):
    """Grey arrow along the abscissa (the direction in which the control moves the state) with a note above it."""
    ax.annotate('', xy=(x_head, y), xytext=(x_tail, y), zorder=5,
                arrowprops=dict(arrowstyle='-|>,head_length=0.5,head_width=0.22', color=NOTE_GREY, lw=0.7,
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    ax.annotate(text, (0.5 * (x_tail + x_head), y), xytext=(0, 2.0), textcoords='offset points', ha='center',
                va='bottom', fontsize=NOTE_FS, color=NOTE_GREY)


def end_labels(ax, x_end, ys, names, colors, min_gap_pt=8.0):
    """Direct labels at the line ends; labels closer than min_gap_pt on the page are pushed apart symmetrically."""
    ys = np.asarray(ys, float)
    y_pt = ax.transData.transform(np.column_stack([np.full(len(ys), x_end), ys]))[:, 1] * 72.0 / fig.dpi
    order = np.argsort(y_pt)
    pos = y_pt[order].copy()
    for _ in range(200):                  # relax overlapping neighbours apart, keeping each cluster centred
        moved = False
        for a in range(len(pos) - 1):
            d = pos[a + 1] - pos[a]
            if d < min_gap_pt - 1e-6:
                pos[a] -= (min_gap_pt - d) / 2; pos[a + 1] += (min_gap_pt - d) / 2; moved = True
        if not moved:
            break
    nudge = np.empty_like(pos); nudge[order] = pos - y_pt[order]
    for y, name, col, dy in zip(ys, names, colors, nudge):
        ax.annotate(name, (x_end, y), xytext=(3.0, dy), textcoords='offset points', ha='left', va='center',
                    fontsize=LABEL_FS, color=col)


# ---------------- A: slot for the TikZ sketch (letter and title only) ----------------
head(LEFT_X, TOP_Y + TOP_H, 'A', 'One dimension per modified equation')

# ---------------- B: singular values of the augmented equilibria ----------------
axB = add_axes(RIGHT_X, TOP_Y, R_W, TOP_H)
CASES = [   # spectrum, rank, colour, marker, horizontal offset, side and height (log10) of the rank label
    dict(sv=spectra[1], rk=ranks[1], c=C[0], m='o', dx=-0.21, side='left', ylab=-8.0),
    dict(sv=sv_cr2, rk=rk_cr2, c=C[3], m='s', dx=-0.07, side='left', ylab=-11.0),
    dict(sv=spectra[2], rk=ranks[2], c=C[1], m='o', dx=0.07, side='right', ylab=-9.5),
    dict(sv=spectra[3], rk=ranks[3], c=C[2], m='o', dx=0.21, side='right', ylab=-7.0),
]
idx = np.arange(1, 9)
for q, cs in enumerate(CASES):            # all lines first, then all markers, so that no marker hides under a line
    axB.plot(idx + cs['dx'], cs['sv'], color=cs['c'], lw=0.9, zorder=2 + 0.1 * q)
for q, cs in enumerate(CASES):
    axB.plot(idx + cs['dx'], cs['sv'], ls='none', marker=cs['m'], ms=3.9 if cs['m'] == 'o' else 3.6, mfc=cs['c'],
             mec='white', mew=0.5, zorder=4 + 0.1 * q)
FLOOR = 1e-16
axB.axhline(FLOOR, color='0.55', lw=0.6, ls=(0, (3, 2)), zorder=1)
axB.annotate('numerical precision', (8.55, FLOOR), xytext=(0, 1.5), textcoords='offset points', ha='right',
             va='bottom', fontsize=NOTE_FS, color='0.35')
axB.set_yscale('log')
axB.set_ylim(1e-18, 8)
axB.set_xlim(0.45, 8.6)
axB.set_xticks(idx)
axB.set_yticks([1e0, 1e-5, 1e-10, 1e-15])
axB.minorticks_off()
axB.set_xlabel('singular value index')
axB.set_ylabel('singular value')
for cs in CASES:   # the rank, next to the drop after the last significant singular value
    r = cs['rk']
    p0 = axB.transData.transform((r + cs['dx'], cs['sv'][r - 1]))
    p1 = axB.transData.transform((r + 1 + cs['dx'], cs['sv'][r]))
    yd = axB.transData.transform((1, 10 ** cs['ylab']))[1]
    xd = p0[0] + (yd - p0[1]) / (p1[1] - p0[1]) * (p1[0] - p0[0])
    off = 3.0 * fig.dpi / 72.0
    xt, yt = axB.transData.inverted().transform((xd - off if cs['side'] == 'left' else xd + off, yd))
    axB.text(xt, yt, f'rank {r}', color=cs['c'], fontsize=KEY_FS, ha='right' if cs['side'] == 'left' else 'left',
             va='center')
axB.text(8.55, 0.2, 'rank of $(\\mathbf{x}^{\\ast}, 1)$\n= dimension + 1', fontsize=NOTE_FS, color='0.35',
         ha='right', va='top', linespacing=1.3)        # links the ranks of B to the dimensions of A
head(RIGHT_X, TOP_Y + TOP_H, 'B', 'Dimension counting')

# ---------------- C: consumer-resource model, mortality of consumer 1 varied ----------------
axC = add_axes(LEFT_X, BOT, L_W, ROW_H)
N1 = Pcr[M_RES]
CR = [0, 1, 2, 3, 5, 6]                                     # R_1..R_4, N_2, N_3
CR_COL = [C[0], C[1], C[2], C[3], '0.35', '0.35']           # resources in the species colours, consumers grey
CR_NAME = ['$R_1$', '$R_2$', '$R_3$', '$R_4$', '$N_2$', '$N_3$']
for j, col in zip(CR, CR_COL):
    axC.plot(N1, Pcr[j], color=col, lw=LINE_W, zorder=2, solid_capstyle='butt')
    axC.scatter([Y0[M_RES]], [Y0[j]], **STAR)
axC.set_xlim(0.69, 1.10); axC.set_ylim(-0.05, 1.12)
axC.set_xticks([0.7, 0.8, 0.9, 1.0]); axC.set_yticks([0, 0.5, 1.0])
axC.set_xlabel(r'abundance $N_1$ (consumer 1)'); axC.set_ylabel('abundance')
ie = int(np.argmax(N1))
end_labels(axC, N1[ie], Pcr[CR, ie], CR_NAME, CR_COL)
direction_note(axC, 1.035, 0.905, 1.0, r'increasing $m_1$')
axC.scatter([0.712], [1.035], **STAR)                     # key for the stars of A, C and D
axC.annotate('unperturbed', (0.712, 1.035), xytext=(5.0, 0), textcoords='offset points', ha='left',
             va='center_baseline', fontsize=NOTE_FS, color=NOTE_GREY)
head(LEFT_X, BOT + ROW_H, 'C', 'Consumer–resource model')

# ---------------- D: replicator equation, one payoff entry varied ----------------
axD = add_axes(RIGHT_X, BOT, R_W, ROW_H)
X1 = Xr[0]
for i in range(1, 4):
    axD.plot(X1, Xr[i], color=C[i], lw=LINE_W, zorder=2, solid_capstyle='butt')
    axD.scatter([XR0[0]], [XR0[i]], **STAR)
axD.set_xlim(0.643, 0.762); axD.set_ylim(-0.005, 0.25)
axD.set_xticks([0.65, 0.70, 0.75]); axD.set_yticks([0, 0.1, 0.2])
axD.set_xlabel(r'frequency $x_1$ (strategy 1)'); axD.set_ylabel(r'frequency $x_i$')
ie = int(np.argmax(X1))
end_labels(axD, X1[ie], Xr[1:, ie], [f'$x_{i + 1}$' for i in range(1, 4)], C[1:4])
direction_note(axD, 0.7485, 0.7155, 0.222, r'increasing $a_{12}$')
head(RIGHT_X, BOT + ROW_H, 'D', 'Replicator equation')
fig.align_ylabels([axB, axD])               # one y-label column on the right


# ----------------------------------------------------------------------------------------------- TikZ inputs
def write_layout():
    cm = 2.54
    with open(os.path.join(HERE, 'figS4_layout.tex'), 'w') as fh:
        fh.write('% GENERATED by figS4_scope.py -- do not edit by hand. Lengths in inches unless noted.\n'
                 f'\\def\\figW{{{W:.4f}}}\\def\\figH{{{H:.4f}}}\n'
                 f'\\def\\slotX{{{SLOT["x"]:.4f}}}\\def\\slotY{{{SLOT["y"]:.4f}}}\n'
                 f'\\def\\slotW{{{SLOT["w"] * cm:.4f}}}\\def\\slotH{{{SLOT["h"] * cm:.4f}}}% slot size in cm\n')


def pdflatex(tex):
    cmd = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={BUILD}', tex]
    res = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stdout.write(res.stdout[-3000:])
        raise SystemExit(f'pdflatex failed on {tex} (log in {BUILD})')


if __name__ == '__main__':
    fig.savefig(os.path.join(BUILD, 'figS4_panels.pdf'))
    if '--debug' in sys.argv:             # optional: draw the outline of the panel-A slot
        fig.patches.append(plt.Rectangle((SLOT['x'] / W, SLOT['y'] / H), SLOT['w'] / W, SLOT['h'] / H,
                                         transform=fig.transFigure, fill=False, ec='r', lw=0.3))
        fig.savefig(os.path.join(BUILD, 'figS4_panels_debug.png'), dpi=300)
    write_layout()
    pdflatex('figS4_sketch.tex')
    pdflatex('figS4_assemble.tex')
    out_pdf = os.path.join(OUT, 'figS4_scope.pdf')
    shutil.copyfile(os.path.join(BUILD, 'figS4_assemble.pdf'), out_pdf)
    subprocess.run(['pdftocairo', '-png', '-r', '300', '-singlefile', out_pdf, out_pdf[:-4]], check=True)
    print('saved', os.path.relpath(out_pdf, ROOT), 'and', os.path.relpath(out_pdf[:-4] + '.png', ROOT))
    print('B: ranks of (x, 1) for 1, 2, 3 modified equations:', [ranks[q] for q in (1, 2, 3)],
          '(draws that keep all seven species:', [nsamp[q] for q in (1, 2, 3)], 'of 300); rate shared by two equations:', rk_cr2)
    for q in (1, 2, 3):
        print(f'   {q} equation(s): sv =', np.array2string(spectra[q], precision=2))
    print('   shared rate:    sv =', np.array2string(sv_cr2, precision=2))
    print(f'C: consumer-resource, mortality of consumer 1: rank (y, 1) = {rk_cr1}, sv =',
          np.array2string(sv_cr1 / sv_cr1[0], precision=2))
    print(f'D: replicator, one payoff entry: rank = {rk_rep}, sv =', np.array2string(sv_rep / sv_rep[0], precision=2))
