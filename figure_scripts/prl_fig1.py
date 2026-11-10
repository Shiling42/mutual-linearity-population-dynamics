"""Fig. 1 -- any change to the equation of one species moves the community along one line (one column, 3.375 in).

Run:  python3 -B figure_scripts/prl_fig1.py      (from any directory; needs pdflatex and pdftocairo)
Data: data/exp_collinear.npz (written by code/exp_collinear.py).
  1. matplotlib draws panels (b) and (c) into figure_scripts/build_fig1/fig1_panels.pdf and leaves the slot of panel
     (a) empty: (b) the abundances against the change theta of A_31, (c) the abundances against x_1* for all four
     modifications of species 3 (markers thinned for legibility, |s| <= 1.3), with the predicted lines; a legend row
     above (c) identifies the four modifications with the glyphs used in (a);
  2. the script writes the TikZ inputs prl_fig1_layout.tex and prl_fig1_edges.tex (the links of the network, from
     the matrix A in the npz), compiles prl_fig1_sketch.tex (panel a: network on top, mechanism below) and places it
     with prl_assemble.tex. In the network every interaction A_ij is drawn as a link from species j that ends in a
     dot at the species i it acts on; the line width grows with |A_ij|, and the weakest are drawn with W_MIN;
  3. the result is figures/fig1.pdf (3.375 in wide) and a 400-dpi PNG for checking.
Sign convention of the paper: B = -A^{-1}, b_k = k-th column of B, x* = x0* + s b_k (s as stored in the npz).
"""
import os
import shutil
import subprocess
import numpy as np
from matplotlib.lines import Line2D
from prl_style import plt, C, DATA, HERE, ROOT, ONE_COL, pdflatex, OUT

W_MIN = 0.28                                # pt, smallest line width of a link, so that the weakest remain visible
BUILD = os.path.join(HERE, 'build_fig1')
os.makedirs(BUILD, exist_ok=True)

d = np.load(os.path.join(DATA, 'exp_collinear.npz'), allow_pickle=True)
r, A, x0, k, l = d['r'], d['A'], d['x0'], int(d['k']), int(d['l'])     # k = species 3 (modified), l = species 1
n = len(r)
bk = -np.linalg.inv(A)[:, k]
X = [d[f'X_{i}'] for i in range(4)]          # equilibria for: press r_3, self-regulation A_33, interaction A_31, mixed
TH = [d[f'th_{i}'] for i in range(4)]
S = [d[f's_{i}'] for i in range(4)]          # effective press (x* = x0* + s b_3)
assert max(np.max(np.abs(X[q] - x0[:, None] - np.outer(bk, S[q]))) for q in range(4)) < 1e-12
keep = [np.abs(S[q]) <= 1.3 for q in range(4)]
Xk = [X[q][:, keep[q]] for q in range(4)]
T_LABEL = [r'press $r_3$', r'$A_{33}$', r'$A_{31}$', 'mixed']
T_MK = ['o', 's', '^', 'D']
T_SIZE = [8, 12, 11, 9]

plt.rcParams.update({'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.labelsize': 7})
W, H = ONE_COL, 2.78
AX_B = [1.86, 1.71, 1.45, 0.88]              # left, bottom, width, height (inches)
AX_C = [1.86, 0.31, 1.45, 0.90]
SLOT = dict(x=0.0, y=0.0, w=1.44, h=H - 0.02)
fig = plt.figure(figsize=(W, H))


def label(x_in, y_in, letter):
    fig.text(x_in / W, y_in / H, f'({letter})', fontsize=8, fontweight='normal', ha='left', va='baseline')


label(0.0, H - 0.13, 'a')

# ---------------- (b) abundances are nonlinear (linear-fractional) functions of the control
ax = fig.add_axes([AX_B[0] / W, AX_B[1] / H, AX_B[2] / W, AX_B[3] / H])
for sp in range(n):
    ax.plot(TH[2], X[2][sp], color=C[sp], lw=1.0, zorder=3 if sp == k else 2, label=f'$x_{sp + 1}$')
ax.set_xlim(-0.68, 1.04); ax.set_ylim(0, 3.45)
ax.set_xticks([-0.5, 0, 0.5, 1]); ax.set_xticklabels(['−0.5', '0.0', '0.5', '1.0'])
ax.set_yticks([0, 1, 2, 3])
ax.set_xlabel(r'$\theta$ (change of $A_{31}$)', labelpad=1)
ax.set_ylabel(r'$x_i^\ast$', labelpad=2)
ax.legend(loc='upper left', ncol=2, handlelength=0.9, handletextpad=0.3, columnspacing=0.6, labelspacing=0.1,
          borderaxespad=0.1, fontsize=6.3)
label(AX_B[0] - 0.36, AX_B[1] + AX_B[3] + 0.04, 'b')

# ---------------- (c) ... but affine functions of each other, the same lines for every modification
ax = fig.add_axes([AX_C[0] / W, AX_C[1] / H, AX_C[2] / W, AX_C[3] / H])
XR, YR, DMIN = 1.04, 2.30, 0.10


def thin(xx, yy, sp, phase):
    """Markers at least DMIN apart along the line, phase-shifted so that the four modifications interleave."""
    t = np.hypot((xx - x0[0]) / XR, (yy - x0[sp]) / YR) * np.sign(xx - x0[0])
    _, first = np.unique(np.floor((t + phase * DMIN) / DMIN), return_index=True)
    return first


sline = np.linspace(-0.86, 1.32, 2)
for sp in [3, 2, 4, 1]:
    ax.plot(x0[0] + sline * bk[0], x0[sp] + sline * bk[sp], color=C[sp], lw=0.8, zorder=1)
    for q in [1, 2, 0, 3]:
        idx = thin(Xk[q][0], Xk[q][sp], sp, q / 4)
        ax.scatter(Xk[q][0][idx], Xk[q][sp][idx], s=T_SIZE[q] * 0.8, marker=T_MK[q],
                   facecolor=C[sp] if q != 3 else 'white', edgecolor='k' if q != 3 else C[sp],
                   linewidth=0.3 if q != 3 else 0.6, zorder=3)
    ax.scatter([x0[0]], [x0[sp]], marker='*', s=30, color='k', edgecolor='white', linewidth=0.3, zorder=6)
ax.set_xlim(1.28, 2.36); ax.set_ylim(-0.05, 2.3)
ax.set_xticks([1.4, 1.8, 2.2]); ax.set_yticks([0, 1, 2])
ax.set_xlabel(r'$x_1^\ast$', labelpad=1); ax.set_ylabel(r'$x_i^\ast$', labelpad=2)
for sp, (tx, ty) in {3: (2.24, 2.17), 2: (2.24, 1.86), 4: (2.24, 1.27), 1: (2.24, 0.70)}.items():
    ax.text(tx, ty, f'$x_{sp + 1}$', color=C[sp], fontsize=7, ha='left', va='center')
# legend row above (c): the four modifications, with the glyphs of panel (a)
handles = [Line2D([], [], ls='none', marker=T_MK[q], ms=3.4 if T_MK[q] != 'o' else 3.6,
                  mfc='0.2' if q != 3 else 'white', mec='0.2', mew=0.0 if q != 3 else 0.7, label=T_LABEL[q])
           for q in range(4)]
fig.legend(handles=handles, loc='center', bbox_to_anchor=((AX_C[0] + AX_C[2] / 2) / W, (AX_C[1] + AX_C[3] + 0.105) / H),
           ncol=4, columnspacing=0.7, handletextpad=0.05, handlelength=0.9, fontsize=6.3, frameon=False,
           borderaxespad=0.0)                # the stars (unperturbed equilibrium) are explained in the caption
label(AX_C[0] - 0.36, AX_C[1] + AX_C[3] + 0.06, 'c')


def write_tikz_inputs():
    lines = ['% GENERATED by prl_fig1.py from data/exp_collinear.npz -- do not edit by hand.']
    for i in range(n):
        for j in range(n):
            if i != j and not (i == k and j == l):          # A_kl is drawn in black by the sketch itself
                lines.append(f'\\draw[interaction, acts, line width={max(1.8 * abs(A[i, j]), W_MIN):.3f}pt] (n{j + 1}) to (n{i + 1});'
                             f'  % A_{i + 1}{j + 1} = {A[i, j]:+.3f}')
    with open(os.path.join(HERE, 'prl_fig1_edges.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    cm = 2.54
    with open(os.path.join(HERE, 'prl_fig1_layout.tex'), 'w') as fh:
        fh.write('% GENERATED by prl_fig1.py\n'
                 f'\\def\\slotW{{{SLOT["w"] * cm:.4f}}}\\def\\slotH{{{SLOT["h"] * cm:.4f}}}% cm\n')
    with open(os.path.join(HERE, 'prl_assemble_layout.tex'), 'w') as fh:
        fh.write('% GENERATED by prl_fig1.py\n'
                 f'\\def\\figW{{{W:.4f}}}\\def\\figH{{{H:.4f}}}\\def\\slotX{{{SLOT["x"]:.4f}}}'
                 f'\\def\\slotY{{{SLOT["y"]:.4f}}}\n'
                 '\\def\\panels{build_fig1/fig1_panels.pdf}\\def\\sketch{build_fig1/prl_fig1_sketch.pdf}\n')


if __name__ == '__main__':
    fig.savefig(os.path.join(BUILD, 'fig1_panels.pdf'))
    write_tikz_inputs()
    pdflatex('prl_fig1_sketch.tex', BUILD)
    pdflatex('prl_assemble.tex', BUILD)
    out = os.path.join(OUT, 'fig1.pdf')
    shutil.copyfile(os.path.join(BUILD, 'prl_assemble.pdf'), out)
    subprocess.run(['pdftocairo', '-png', '-r', '400', '-singlefile', out, out[:-4]], check=True)
    print('saved', os.path.relpath(out, ROOT))
