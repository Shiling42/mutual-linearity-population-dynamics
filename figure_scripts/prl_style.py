"""Shared settings of the figure scripts prl_fig1.py, prl_fig2.py, prl_fig3.py and prl_figS1.py: folders, sizes for
Physical Review Letters (one column 3.375 in, two columns 7.0 in), panel labels "(a)" in regular weight, and helpers
that compile the TikZ parts of a figure.

Builds on figstyle.py (colours C, species i -> C[i-1] in every figure) and changes only the text sizes.
"""
import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))               # the folder that holds code/, data/, figures/
sys.path.insert(0, os.path.join(ROOT, 'code'))                  # glv_tools.py, ml_core.py
from figstyle import plt, C  # noqa: E402,F401

DATA = os.path.join(ROOT, 'data')                               # written by code/exp_*.py
OUT = os.path.join(ROOT, 'figures')                             # the figures
os.makedirs(OUT, exist_ok=True)

ONE_COL = 3.375
TWO_COL = 7.0

plt.rcParams.update({
    'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0,
    'legend.fontsize': 7.0, 'axes.labelpad': 2.0, 'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
})


def panel_label(fig, x_in, y_in, letter, W, H):
    """Panel label "(a)" with its baseline at (x_in, y_in) inches from the lower left of the figure."""
    fig.text(x_in / W, y_in / H, f'({letter})', fontsize=8.5, fontweight='normal', ha='left', va='baseline')


def pdflatex(tex, build, cwd=HERE):
    cmd = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={build}', tex]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stdout.write(res.stdout[-3000:])
        raise SystemExit(f'pdflatex failed on {tex} (log in {build})')


def finish(build_pdf, name):
    """Copy the assembled PDF to figures/<name>.pdf and render a 300-dpi PNG next to it (for checking)."""
    out_pdf = os.path.join(OUT, name + '.pdf')
    shutil.copyfile(build_pdf, out_pdf)
    subprocess.run(['pdftocairo', '-png', '-r', '300', '-singlefile', out_pdf, out_pdf[:-4]], check=True)
    print('saved', os.path.relpath(out_pdf, ROOT))
    return out_pdf
