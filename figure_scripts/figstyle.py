"""Base style of all figures: Matplotlib settings and the colours. Species i has the colour C[i-1] in every figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito palette (distinguishable with colour-vision deficiencies)
C = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']

plt.rcParams.update({
    'font.size': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'lines.linewidth': 1.1, 'axes.linewidth': 0.6, 'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'], 'mathtext.fontset': 'dejavusans',
    'axes.titlesize': 7.5, 'legend.frameon': False, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'pdf.fonttype': 42,
})

TWO_COL = 7.0    # inches: width of a two-column figure
