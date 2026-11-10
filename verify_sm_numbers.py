"""
verify_sm_numbers.py -- every number quoted in the Supplemental Material (SM) of "Mutual Linearity in Nonlinear
Population Dynamics" and in the figure captions of the Letter and the SM, printed next to the value found from the
code and data of this repository, with a PASS or FAIL mark.

Run:  python3 -B verify_sm_numbers.py      (a few seconds; needs the files in data/, written by code/exp_*.py and
                                            code/crisis_scan_extended.py)
Exit status: 0 if every check passes, 1 otherwise. The summary at the end lists every FAIL, and every check that
passes by a tolerance although the digits printed in the text differ.

The column "in the text" holds the number as the SM prints it (3e-4 stands for 3 x 10^-4). It is typed in below, so
it must be updated when the text changes. The rule in brackets says how the two values are compared:
  exact      counts, labels and parameters: equal.
  rounds     the value found, rounded to the digits printed in the text, equals the value in the text.
  bound      the text gives a bound ("to", "below", "better than", "at most"): the value found does not exceed it.
  holds      a statement without a number: the script says what it evaluated and the threshold used.
  noise      the text quotes a value at the level of rounding errors (1e-16 to 1e-15). Such values depend on the
             machine and on the library versions. PASS if the value found is less than ten times the value quoted.
  run        a value from a chaotic trajectory. The SM says that these digits depend on the run (caption of
             Table S1), and they also depend on the machine. PASS within the tolerance named in the rule. Where
             the text gives a bare power of ten, the value found must lie between half of it and ten times it.

Cheap quantities (algebra, the mutualistic example, the Hopf point, the sweeps of Fig. S1(c)) are computed here with
the functions of code/glv_tools.py and code/ml_core.py; everything else is read from data/.
"""
import os
import sys
import warnings
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))
from glv_tools import (make_community, interior, sweep, segments, kink_prediction, boundary_state,  # noqa: E402
                       repeating_maxima, VANO_R, VANO_A)
from ml_core import numerical_rank, glv_saturated_equilibrium  # noqa: E402
with warnings.catch_warnings():             # the module switches warnings off for its own run; undone here
    from crisis_scan_extended import T_END as SCAN_T_END  # noqa: E402

DATA = os.path.join(HERE, 'data')
ROWS = []                                   # (section, ok, what, text, found, rule, digits_differ)
SECTION = ['']


# ---------------------------------------------------------------------------------------------- comparison rules
def sci(x, sig=1):
    """1.99e-3 -> '2e-3', the way the text writes 2 x 10^-3."""
    m, e = f'{x:.{sig - 1}e}'.split('e')
    return f'{m}e{int(e)}'


def as_text(text, found):
    """The value found, written with the precision of the number in the text."""
    if 'e' in text:
        mant = text.split('e')[0]
        return sci(found, len(mant.replace('-', '').replace('.', '')))
    dec = len(text.split('.')[1]) if '.' in text else 0
    return f'{found:.{dec}f}'


def same_digits(text, found):
    return float(as_text(text, found)) == float(text)


def exponent(x):
    """Power of ten of x in scientific notation (3.2e-8 -> -8)."""
    return int(np.floor(np.log10(max(abs(x), 1e-300))))


def section(title):
    SECTION[0] = title


def add(ok, what, text, found, rule, differ=False):
    ROWS.append((SECTION[0], bool(ok), what, str(text), str(found), rule, bool(differ)))


def exact(what, text, found):
    add(found == text, what, text, found, 'exact')


def rounds(what, text, found):
    add(same_digits(text, found), what, text, f'{as_text(text, found)}  ({found:.5g})', 'rounds')


def bound(what, text, found, word='below'):
    add(found <= float(text), what, f'{word} {text}', f'{found:.1e}', 'bound')


def holds(what, ok, found, how):
    add(ok, what, '(statement)', found, f'holds: {how}')


def noise(what, text, found, power=False):
    """Rounding-error level. power=True: the text gives a bare power of ten, so only the exponent is compared."""
    differ = (exponent(found) != exponent(float(text))) if power else not same_digits(text, found)
    add(found < 10 * float(text), what, text, f'{found:.1e}', 'noise', differ)


def run(what, text, found, ok, tolerance, power=False):
    """Chaotic trajectory: PASS within the tolerance; flagged if the printed digits differ."""
    differ = (exponent(found) != exponent(float(text))) if power else not same_digits(text, found)
    add(ok, what, text, f'{as_text(text, found)}  ({found:.4g})', f'run: {tolerance}', differ)


# ---------------------------------------------------------------------------------------------- data
NEEDED = ['exp_collinear.npz', 'exp_stepwise_example.npz', 'exp_stepwise_stats.npz', 'exp_divergence_stats.npz',
          'exp_use.npz', 'exp_chaos.npz', 'crisis_scan_extended.csv']
missing = [f for f in NEEDED if not os.path.exists(os.path.join(DATA, f))]
if missing:
    sys.exit('missing in data/: ' + ', '.join(missing) + '\nrun the experiments first (python3 -B run_all.py)')
col = np.load(os.path.join(DATA, 'exp_collinear.npz'))
ex = np.load(os.path.join(DATA, 'exp_stepwise_example.npz'))
st = np.load(os.path.join(DATA, 'exp_stepwise_stats.npz'))
dv = np.load(os.path.join(DATA, 'exp_divergence_stats.npz'))
use = np.load(os.path.join(DATA, 'exp_use.npz'))
ch = np.load(os.path.join(DATA, 'exp_chaos.npz'))
scan = np.genfromtxt(os.path.join(DATA, 'crisis_scan_extended.csv'), delimiter=',', names=True)

# ---- community of Fig. 1 (species 3 modified)
r1, A1, x01 = col['r'], col['A'], col['x0']
k1, l1 = int(col['k']), int(col['l'])
B1 = -np.linalg.inv(A1)                       # the paper's press-response matrix
b3 = B1[:, k1]
XS = [col[f'X_{i}'] for i in range(4)]        # press r_3, self-regulation A_33, interaction A_31, mixed
TS = [col[f'th_{i}'] for i in range(4)]
SS = [col[f's_{i}'] for i in range(4)]        # effective press from the displacement along the line
SID = [col[f'sid_{i}'] for i in range(4)]     # effective press from its definition, Eq. (3)

# ---- community of Fig. 2 (species 2 modified through A_23)
r2, A2 = ex['r'], ex['A']
k2, l2 = int(ex['k']), int(ex['l'])
X2, TH2 = ex['X'], ex['thetas']
SUPP2 = [tuple(int(v) for v in s) for s in ex['SUPP']]
segs2 = segments(SUPP2)
n2 = len(r2)
E2 = np.eye(n2)

# ---- Fig. S1(c): modifications of species 2 against the press path (same definitions as figure_scripts/prl_figS1.py;
#      the mixed modification is that of Sec. S2: r_2 + 0.3 sin 2 theta, A_25 + theta^2, A_26 - 0.7 theta)
MODS = {
    'press r_2': (lambda th: (r2 + th * E2[k2], A2), np.linspace(-1.0, 1.95, 60)),
    'A_22': (lambda th: (r2, A2 + th * np.outer(E2[k2], E2[k2])), np.linspace(-0.9, 3.0, 120)),
    'A_23': (lambda th: (r2, A2 + th * np.outer(E2[k2], E2[l2])), np.linspace(-0.6, 3.0, 361)),
    'A_21': (lambda th: (r2, A2 + th * np.outer(E2[k2], E2[0])), np.linspace(-3.0, 3.0, 120)),
    'mixed': (lambda th: (r2 + 0.3 * np.sin(2 * th) * E2[k2],
                          A2 + np.outer(E2[k2], th ** 2 * E2[4] - 0.7 * th * E2[5])), np.linspace(-2.0, 2.0, 120)),
}
PATH = {}
for name, (mod, ths) in MODS.items():
    mult = []
    X, SUPP = sweep(r2, A2, mod, ths, multiplicity=mult)
    kept = np.where(~np.isnan(X[0]))[0]
    worst = 0.0
    for q in kept:                              # x*(theta) against the press path at p = s(theta) = h(x*; theta)
        rt, At = mod(ths[q])
        s = (rt[k2] - r2[k2]) + (At[k2] - A2[k2]) @ X[:, q] if X[k2, q] > 0 else -10.0   # species 2 absent: no effect
        xp, _ = glv_saturated_equilibrium(r2 + s * E2[k2], A2)
        worst = max(worst, np.max(np.abs(xp - X[:, q])))
    PATH[name] = dict(worst=worst, comps=len({SUPP[q] for q in kept}), mult=np.array(mult))

# ---- the community of Vano et al.
a0, rho = VANO_A, VANO_R
thetas = ch['thetas']
nT = len(thetas)
xeq = np.array([ch[f'xeq_{i}'] for i in range(nT)])
xavg = np.array([ch[f'xavg_{i}'] for i in range(nT)])
xmin = np.array([ch[f'xmin_{i}'] for i in range(nT)])
lyap = np.array([float(ch[f'lyap_{i}']) for i in range(nT)])
dev = np.max(np.abs(xavg - xeq), axis=1)
regime = [str(g) for g in ch['regime']]
chaotic = [g == 'chaos' for g in regime]


# ============================================================================================== Sec. S1, S2
section('Sec. S1  Notation')
Cm = np.diag(x01) @ A1                          # community matrix of the Fig. 1 community
holds('the k-th column of -C^-1 equals b_k / x_k*  (community of Fig. 1, k = 3)',
      np.max(np.abs(-np.linalg.inv(Cm)[:, k1] - b3 / x01[k1])) < 1e-12,
      f'largest difference = {np.max(np.abs(-np.linalg.inv(Cm)[:, k1] - b3 / x01[k1])):.1e}', 'to 1e-12')
holds('a press r_k -> r_k + p moves the equilibrium to x0* + p b_k  (same community, p = 0.2)',
      np.max(np.abs(interior(r1 + 0.2 * np.eye(len(r1))[k1], A1) - (x01 + 0.2 * b3))) < 1e-12,
      f'largest difference = {np.max(np.abs(interior(r1 + 0.2 * np.eye(len(r1))[k1], A1) - (x01 + 0.2 * b3))):.1e}',
      'to 1e-12')

section('Sec. S2  Numerical methods')
for label, (n_, seed_, gam_), stored in [('Fig. 1', (5, 3, 0.9), (r1, A1)), ('Fig. 2', (6, 28, 1.1), (r2, A2)),
                                          ('Fig. S4(b)', (7, 11, 0.9), (use['dim_r'], use['dim_A']))]:
    rr, AA = make_community(n_, seed=seed_, sigma=gam_)
    xx = interior(rr, AA)
    same = np.array_equal(rr, stored[0]) and np.array_equal(AA, stored[1])
    add(same, f'{label}: n = {n_}, gamma = {gam_}, seed {seed_} gives the stored r and A', f'seed {seed_}',
        'identical' if same else 'different', 'exact')
    holds(f'{label} community: A_ii = -1, r_i in [0.5, 1.5], x0_i > 0.05, locally stable',
          np.all(np.diag(AA) == -1) and rr.min() >= 0.5 and rr.max() <= 1.5 and xx.min() > 0.05
          and np.max(np.linalg.eigvals(np.diag(xx) @ AA).real) < 0,
          f'r in [{rr.min():.2f}, {rr.max():.2f}], min x0 = {xx.min():.3f}, '
          f'max Re eig = {np.max(np.linalg.eigvals(np.diag(xx) @ AA).real):.3f}', 'acceptance rule of the text')
s_mixed = 0.4 * np.sin(2 * TS[3]) + TS[3] ** 2 * XS[3][0] - 0.5 * TS[3] * XS[3][4]
holds('mixed modification of Fig. 1: r_3 + 0.4 sin 2theta, A_31 + theta^2, A_35 - 0.5 theta',
      np.max(np.abs(s_mixed - SS[3])) < 1e-12, f'max |s - h(x*; theta)| = {np.max(np.abs(s_mixed - SS[3])):.1e}',
      'stored effective press equals this h(x*; theta) to 1e-12')

all_mult = np.concatenate([col[f'mult_{i}'] for i in range(4)] + [ex['mult'], st['mult']]
                          + [use[f'dim_mult_{q}'] for q in (1, 2, 3)] + [use['cr_mult']]
                          + [PATH[nm]['mult'] for nm in MODS])
add(int(np.sum(all_mult > 1)) == 0, 'saturated equilibrium "unique in all sweeps reported": grid points with more than one',
    0, f'{int(np.sum(all_mult > 1))} of {len(all_mult)} points', 'exact')
exact('direct integration: number of settings', 600, int(dv['checks']))
exact('   settings at which the composition agrees', 599, int(dv['agree']))
exc = dv['exceptions']
add(len(exc) == 1 and exc[0, 5] == 1 and exc[0, 10] == 1, '   the exception is "one grid step from an extinction"',
    '1 grid step', f'{int(exc[0, 5])} grid step, extinction of the missing species: {bool(exc[0, 10])}'
    if len(exc) == 1 else f'{len(exc)} exceptions', 'exact')
if len(exc) == 1:
    rounds('   equilibrium abundance of the species that had not recovered', '3e-4', exc[0, 8])

bound('Fig. 1: |s from Eq. (3) - s from the displacement|, four modifications', '1e-14',
      max(np.max(np.abs(SS[i] - SID[i])) for i in range(4)), 'better than')
exact('   number of equilibria with full coexistence', 304, int(sum(X.shape[1] for X in XS)))
bound('   distance of these equilibria from the line x0* + s b_3 (Euclidean; s fitted by least squares)', '1e-14',
      float(col['dist'].max()))
allX = np.hstack(XS)
rk, sv = numerical_rank(np.vstack([allX, np.ones(allX.shape[1])]), rtol=1e-12)
exact('   numerical rank of the points (x*, 1)', 2, rk)
rounds('   second singular value / first', '0.29', sv[1] / sv[0])
noise('   third singular value / first', '2e-16', sv[2] / sv[0])

# ============================================================================================== Sec. S3
section('Sec. S3  Effective press, functional responses, press path (Fig. S1)')
holds('Fig. S1(a): s = theta for a press', np.max(np.abs(SS[0] - TS[0])) < 1e-12,
      f'max |s - theta| = {np.max(np.abs(SS[0] - TS[0])):.1e}', 'to 1e-12')
for i, (lab, l_) in [(2, ('A_31', l1)), (1, ('A_33', k1))]:
    pred = TS[i] * x01[l_] / (1 - TS[i] * B1[l_, k1])
    rel = np.max(np.abs(SS[i] - pred) / np.maximum(1.0, np.abs(pred)))
    holds(f'Fig. S1(a): s is the ratio of affine functions of Eq. (6) for a change of {lab}', rel < 1e-12,
          f'max relative deviation = {rel:.1e}', 'to 1e-12')
ds = np.diff(SS[3])
holds('Fig. S1(a): s is nonmonotonic for the mixed modification', (ds > 0).any() and (ds < 0).any(),
      f'{int((ds > 0).sum())} increasing and {int((ds < 0).sum())} decreasing steps', 'both signs occur')
exact('type II: x_j / (1 + h x_j) with h', 0.25, float(col['h']))
f_sat = col['xn0'].copy(); f_sat[:2] = f_sat[:2] / (1 + float(col['h']) * f_sat[:2])      # saturation for j = 1, 2
holds('type II: the stored unperturbed equilibrium solves the model with the saturation for j = 1, 2',
      np.max(np.abs(r1 + A1 @ f_sat)) < 1e-9, f'largest residual = {np.max(np.abs(r1 + A1 @ f_sat)):.1e}', 'to 1e-9')
bound('type II: abundances under the other modifications against those under the press', '4e-6',
      float(np.nanmax(col['t2_dev'])), 'to')
rounds('type II: smallest deviation of a curve from a straight line (%)', '1', 100 * float(col['t2_bend'].min()))
rounds('type II: largest deviation of a curve from a straight line (%)', '7', 100 * float(col['t2_bend'].max()))
four = ['A_23', 'A_22', 'A_21', 'mixed']
comps = [PATH[nm]['comps'] for nm in four]
add(set(comps) <= {5, 6}, 'Fig. S1(c): compositions visited by the changes of A_23, A_22, A_21 and the mixed change',
    'five or six each', ', '.join(str(c) for c in comps), 'exact')
bound('Fig. S1(c): these four modifications against the press path at p = s(theta)', '6e-14',
      max(PATH[nm]['worst'] for nm in four), 'to')

# ============================================================================================== Sec. S4
section('Sec. S4  Composition changes in random communities')
exact('number of random sweeps', 120, int(dv['n_sweeps']))
exact('   communities with n = 5; with n = 8', (60, 60), (int((st['sizes'] == 5).sum()), int((st['sizes'] == 8).sum())))
exact('   gamma', 1.0, float(st['gamma']))
exact('   range of theta; number of steps', (-3.0, 3.0, 121),
      (float(st['thetas'][0]), float(st['thetas'][-1]), len(st['thetas'])))
exact('   grid points of all sweeps', 120 * 121, len(st['mult']))
exact('sweeps with a saturated equilibrium at every step', 93, int(dv['n_complete']))
exact('   the same count in exp_stepwise.py', 93, len(st['nkinks']))
nk = st['nkinks']
rounds('changes of composition per sweep: mean', '1.8', float(nk.mean()))
exact('   maximum', 5, int(nk.max()))
rounds('   sweeps with at least one (%)', '92', 100 * float(np.mean(nk >= 1)))
rounds('   sweeps with at least two (%)', '58', 100 * float(np.mean(nk >= 2)))
sl = st['single_loss_len']
exact('kinks: losses of a single species with k, l present, compared with Eq. (7)', 67, len(st['cosines']))
# The text requires the branch after the kink to be "at least three grid steps long" (four grid points). The
# experiment requires three grid points (two steps). Set KINK_BRANCH to 'points' when the text says so.
KINK_BRANCH = 'steps'
add(int((sl >= (4 if KINK_BRANCH == 'steps' else 3)).sum()) == 67,
    f'   ... "the following branch was at least three grid {KINK_BRANCH} long"', 67,
    f'{int((sl >= 3).sum())} with at least three grid points (two steps); {int((sl >= 4).sum())} with at least three steps',
    'exact')
noise('   1 - |cos| between Eq. (7) and the observed direction, largest', '3e-16', max(1 - float(st['cosines'].min()), 1e-17))
for q in range(1, len(segs2)):                 # the two extinctions of the example of Fig. 2
    (p0, p1), (c0, c1) = segs2[q - 1], segs2[q]
    Sb = [i for i in range(n2) if SUPP2[p0][i]]; Sa = [i for i in range(n2) if SUPP2[c0][i]]
    lost = sorted(set(Sb) - set(Sa))
    if len(lost) == 1 and not set(Sa) - set(Sb):
        d_pred = kink_prediction(A2, k2, Sb, lost[0]); d_obs = X2[:, c1 - 1] - X2[:, c0]
        cos = abs(d_obs @ d_pred) / np.linalg.norm(d_obs) / np.linalg.norm(d_pred)
        holds(f'Fig. 2(b): direction after the loss of species {lost[0] + 1} is that of Eq. (7)', 1 - cos < 1e-12,
              f'1 - |cos| = {1 - cos:.1e}', 'to 1e-12')
noise('Fig. 2 example: third singular value / first on each branch, largest', '1e-16', float(ex['sv3'].max()), power=True)

Am = -np.eye(3) + 0.3 * (np.ones((3, 3)) - np.eye(3)); rm = np.ones(3)      # mutualistic example
Bm = -np.linalg.inv(Am)
holds('mutualistic example (n = 3, c = 0.3, r = 1): all entries of b_1 are positive', np.all(Bm[:, 0] > 0),
      f'b_1 = {np.round(Bm[:, 0], 3)}', 'all > 0')
th_inf = 1.0 / Bm[1, 0]                       # A_12 is changed: k = 1, l = 2, pole at 1 / B_21
rounds('   theta_infinity = 1 / B_21', '1.73', th_inf)
tg = np.linspace(0, 0.999 * th_inf, 400)
Xm = np.array([glv_saturated_equilibrium(rm, Am + t * np.outer(np.eye(3)[0], np.eye(3)[1]))[0] for t in tg])
x_beyond, _ = glv_saturated_equilibrium(rm, Am + 1.005 * th_inf * np.outer(np.eye(3)[0], np.eye(3)[1]))
holds('   all three abundances rise up to the pole, no extinction, no saturated equilibrium beyond it',
      Xm.dtype != object and np.all(Xm > 0) and np.all(np.diff(Xm, axis=0) > 0) and x_beyond is None,
      f'abundances at 0.999 theta_infinity = {np.round(Xm[-1], 0)}', 'grid of 400 values of theta')
exact('sweeps without a saturated equilibrium over the whole range', 27, int(dv['n_div']))
exact('   of these, with k and l present on the last branch', 27, int(dv['sign_total']))
exact('   of these, last branch satisfies the sign criterion', 26, int(dv['sign_ok']))
exact('   of these, pole of that branch within the next grid step', 26, int(dv['pole_in_step']))
res = dv['resolved']
add(len(res) == 1 and res[0, 10] == 1 and res[0, 11] == 1 and res[0, 12] == 1 and res[0, 13] == 1,
    '   the remaining one: "an extinction and the pole lay within one grid step"', 1,
    (f'species {int(res[0, 6]) + 1} lost at theta = {res[0, 7]:.4f}, pole at {res[0, 9]:.4f}, grid step '
     f'({res[0, 4]:.2f}, {res[0, 5]:.2f})') if len(res) == 1 else f'{len(res)} sweeps', 'exact')
exact('complete sweeps: branches (k, l present) that end inside the range by a change of composition', 85,
      int(dv['br_complete']))

# ============================================================================================== Sec. S5
section('Sec. S5  Uses of the lines (Fig. S2)')
s0 = int(use['s0'])
holds('Fig. S2(a),(d): the branch of the two measurements is the one without species 4',
      SUPP2[s0] == (1, 1, 1, 0, 1, 1), f'species present: {SUPP2[s0]}', 'composition of the branch')
slopes_sweep = use['slopes_int']                       # dx_i/dx_1 for i = 2..6 from the sweep of A_23
slopes_press = use['dp'][1:] / use['dp'][0]            # the same ratios from an actual press on species 2
for sp, text in [(2, '1.5146'), (3, '-0.5697'), (5, '-1.2208'), (6, '1.4607')]:
    rounds(f'slope of x_{sp} against x_1 from the sweep of A_23', text, float(slopes_sweep[sp - 2]))
holds('   the slopes from the sweep and from the press "agree to all digits"',
      np.max(np.abs(slopes_sweep - slopes_press)) < 1e-12,
      f'largest difference = {np.max(np.abs(slopes_sweep - slopes_press)):.1e}', 'to 1e-12')
s1 = int(use['s1'])
holds('Fig. S2(a): abundances on that branch predicted from x_1 and two measurements',
      np.max(np.abs(use['pred'] - X2[:, s0:s1])) < 1e-12,
      f'largest error = {np.max(np.abs(use["pred"] - X2[:, s0:s1])):.1e}', 'to 1e-12')
ang = [np.degrees(np.arctan2(X2[2, c1 - 1] - X2[2, c0], X2[0, c1 - 1] - X2[0, c0])) for (c0, c1) in segs2[1:]]
holds('Fig. S2(b): the relation between x_1 and x_3 kinks at each change of composition',
      np.min(np.abs(np.diff(ang))) > 1.0 and np.allclose(X2[:, segs2[0][0]:segs2[0][1]], X2[:, [0]]),
      'directions of the branches in the (x_1, x_3) plane: point, ' + ', '.join(f'{v:.0f}' for v in ang) + ' degrees',
      'consecutive directions differ by more than 1 degree')
a_, b_ = segs2[1]                                      # Fig. S2(c): the branch on which species 6 is absent
S = [j for j in range(n2) if SUPP2[a_][j]]
th0 = TH2[b_ - 1]; At = A2.copy(); At[k2, l2] += th0
Bc = np.linalg.inv(At[np.ix_(S, S)]); xS = X2[S, b_ - 1]; ik, il = S.index(k2), S.index(l2)
pole = th0 - 1.0 / Bc[il, ik]
x_sp2 = lambda t: xS[ik] - (t - th0) * xS[il] / (1 + (t - th0) * Bc[il, ik]) * Bc[ik, ik]
th_ext = brentq(x_sp2, pole + 1e-6, th0, xtol=1e-14)
holds('Fig. S2(c): moving left, species 2 reaches zero before the pole', SUPP2[a_] == (1, 1, 1, 1, 1, 0) and pole < th_ext,
      f'species 2 reaches zero at theta = {th_ext:.3f}, pole at theta = {pole:.3f}', 'branch without species 6')

# ============================================================================================== Sec. S6
section('Sec. S6  Cycles and chaos (Fig. 3, Fig. S3, Table S1)')
same = (tuple(float(v) for v in rho) == (1.0, 0.72, 1.53, 1.27)
        and tuple(tuple(float(v) for v in row) for row in a0)
        == ((1.0, 1.09, 1.52, 0.0), (0.0, 1.0, 0.44, 1.36), (2.33, 0.0, 1.0, 0.47), (1.21, 0.51, 0.35, 1.0)))
add(same, 'model of Vano et al.: rho and the rows of a', 'as in Sec. S6', 'identical' if same else 'different', 'exact')
exact('Fig. 3, Fig. S3: number of values of theta', 9, nT)
exact('integration: method (DOP853 is an eighth-order Runge-Kutta method)', 'DOP853', str(ch['method']))
exact('   relative tolerance', 1e-9, float(ch['rtol']))
exact('   transient; averaging window (time units)', (3000, 40000), (int(ch['t_transient']), int(ch['t_avg'])))
exact('   last stored sample is the end of that window', 43000, int(round(float(ch['ttail_0'][-1]) + float(ch['dt_sample']))))
exact('Lyapunov exponents: time span; step', (3000.0, 0.01), (float(ch['lyap_time']), float(ch['lyap_dt'])))
remax_of = lambda th: np.max(np.linalg.eigvals(-np.diag(rho * np.linalg.solve(a0 + th * np.outer(np.eye(4)[0], np.eye(4)[2]), np.ones(4)))
                                                   @ (a0 + th * np.outer(np.eye(4)[0], np.eye(4)[2]))).real)
rounds('Hopf bifurcation of the interior equilibrium, theta_H', '-0.374', brentq(remax_of, -0.45, -0.30, xtol=1e-12))
C_ = np.linalg.inv(a0); xv0 = C_ @ np.ones(4)
B_v = -np.linalg.inv(-np.diag(rho) @ a0)       # B of the paper for the Vano community (A = -diag(rho) a)
line = lambda th: xv0 + ((-th) * xv0[2] / (1 - (-th) * B_v[2, 0])) * B_v[:, 0]     # Eqs. (4), (6) with theta -> -theta
holds('Fig. S3 caption: the interior equilibria are given by Eqs. (4) and (6) with theta -> -theta',
      max(np.max(np.abs(xeq[i] - line(thetas[i]))) for i in range(nT)) < 1e-12,
      f'max deviation = {max(np.max(np.abs(xeq[i] - line(thetas[i]))) for i in range(nT)):.1e}', 'to 1e-12')

TABLE_S1 = [   # theta, regime, lambda_max, min x, max_i |<x_i> - x_i*|, as printed in Table S1
    (-0.45, 'fixed point', '-0.009', '0.125', '3e-13'),
    (-0.38, 'fixed point', '-0.001', '0.126', '3e-7'),
    (-0.30, 'limit cycle', '0.000', '0.053', '6e-6'),
    (-0.20, 'limit cycle', '0.000', '0.028', '1e-6'),
    (-0.10, 'limit cycle', '-0.001', '0.012', '5e-5'),
    (0.00, 'chaos', '0.020', '0.002', '2e-5'),
    (0.08, 'chaos', '0.023', '1e-4', '1e-5'),
    (0.15, 'weak chaos', '0.004', '3e-8', '3e-5'),
    (0.20, 'chaos', '0.015', '1e-8', '6e-5'),
]
for i, (th, reg, lam, mnx, dvt) in enumerate(TABLE_S1):
    tag = f'Table S1, theta = {th:+.2f}:'
    add(abs(thetas[i] - th) < 1e-12 and reg.replace('weak ', '') == regime[i], f'{tag} regime', reg, regime[i], 'exact')
    minx = float(ch['minx'][i])
    if chaotic[i]:
        run(f'{tag} lambda_max', lam, lyap[i], abs(lyap[i] - float(lam)) <= 2e-3, 'within 0.002')
        run(f'{tag} min x', mnx, minx, 0.5 <= minx / float(mnx) <= 2, 'within a factor of 2')
        run(f'{tag} max |<x> - x*|', dvt, dev[i], 0.2 <= dev[i] / float(dvt) <= 5 and dev[i] <= 2e-4,
            'within a factor of 5, at most 2e-4')
    else:
        rounds(f'{tag} lambda_max', lam, lyap[i])
        rounds(f'{tag} min x', mnx, minx)
        if float(dvt) < 1e-10:                 # stable fixed point: the deviation is the integration error
            noise(f'{tag} max |<x> - x*|', dvt, dev[i])
        else:
            rounds(f'{tag} max |<x> - x*|', dvt, dev[i])
rep = {round(float(thetas[i]), 2): bool(repeating_maxima(ch[f'tail_{i}'][1])) for i in range(nT) if regime[i] != 'fixed point'}
holds('Table S1 caption: the successive maxima of x_2 repeat on the cycles but not at theta = 0.15',
      all(rep[t] for t in (-0.3, -0.2, -0.1)) and not rep[0.15],
      'repeat at theta = ' + ', '.join(f'{t:+.2f}' for t in sorted(rep) if rep[t]) + '; do not repeat at '
      + ', '.join(f'{t:+.2f}' for t in sorted(rep) if not rep[t]), 'maxima over the last 800 time units')
add(dev.max() <= 2e-4, 'deviations of the time averages from the line', 'at most about 1e-4', f'{dev.max():.1e}',
    'bound (about: up to 2e-4)')
rounds('1/T for the averaging window T = 4e4', '2.5e-5', 1 / float(ch['t_avg']))

x34, g0 = boundary_state(rho, a0, [2, 3])
rate = lambda th, i: boundary_state(rho, a0 + th * np.outer(np.eye(4)[0], np.eye(4)[2]), [2, 3])[1][i]
rounds('equilibrium of species 3 and 4: x_3', '0.634', x34[2])
rounds('   x_4', '0.778', x34[3])
rounds('   invaded by species 1 for theta below', '0.056', g0[0] / (rho[0] * x34[2]))
rounds('   invasion growth rate of species 1 at theta = 0', '0.036', rate(0.0, 0))
rounds('   invasion growth rate of species 1 at theta = 0.1', '-0.028', rate(0.1, 0))
holds('   never invaded by species 2', rate(0.0, 1) < 0 and rate(0.0, 1) == rate(0.3, 1),
      f'invasion growth rate of species 2 = {rate(0.0, 1):.3f}, independent of theta', 'negative')
run('smallest abundance of species 3 on the attractor at theta = 0', '2e-3', xmin[5][2],
    0.5 <= xmin[5][2] / 2e-3 <= 2, 'within a factor of 2')
run('   at theta = 0.08', '1e-4', xmin[6][2], 0.5e-4 <= xmin[6][2] < 1e-3, 'power of ten', power=True)
run('   at theta = 0.15', '1e-8', xmin[7][2], 0.5e-8 <= xmin[7][2] < 1e-7, 'power of ten', power=True)
run('   at theta = 0.2', '1e-8', xmin[8][2], 0.5e-8 <= xmin[8][2] < 1e-7, 'power of ten', power=True)

ths_scan = np.unique(scan['theta'])
stay = {round(float(t), 2): int(scan['stayed'][scan['theta'] == t].sum()) for t in ths_scan}
n_ic = {round(float(t), 2): int((scan['theta'] == t).sum()) for t in ths_scan}
grid = [round(0.15 + 0.01 * j, 2) for j in range(16)]
add(tuple(grid) == tuple(sorted(stay)), 'crisis scan: values of theta', '0.15 to 0.30, step 0.01',
    f'{min(stay):.2f} to {max(stay):.2f}, {len(stay)} values', 'exact')
exact('   trajectories at each theta', 12, int(min(n_ic.values())) if len(set(n_ic.values())) == 1 else -1)
exact('   duration of each trajectory (time units)', 4e4, float(SCAN_T_END))
exact('   trajectories that stay at theta = 0.15, 0.16, 0.17', (12, 12, 12), tuple(stay[t] for t in (0.15, 0.16, 0.17)))
exact('   escapes begin at theta', 0.18, min(t for t in grid if stay[t] < n_ic[t]))
esc = scan['t_collapse'][scan['stayed'] == 0]
add(esc.min() >= 100 and esc.max() <= 3000, '   chaotic transient before the escape (time units)', '1e2 to 3e3',
    f'{esc.min():.0f} to {esc.max():.0f}', 'run: inside the range of the text')
exact('   trajectories that stay at theta = 0.18 and 0.22', (0, 0), (stay[0.18], stay[0.22]))
mid = [stay[t] for t in (0.19, 0.20, 0.21, 0.23, 0.24)]
add(min(mid) == 3 and max(mid) == 9, '   trajectories that stay at theta = 0.19-0.21 and 0.23-0.24', 'three to nine',
    ', '.join(str(v) for v in mid), 'exact')
exact('   trajectories that stay at theta >= 0.25', 0, sum(stay[t] for t in grid if t >= 0.25))
i20 = int(np.argmin(np.abs(thetas - 0.2)))
holds('the trajectory of Table S1 at theta = 0.2 stays for the whole run',
      min(xmin[i20][0], xmin[i20][1]) > 1e-9,
      f'smallest x_1, x_2 in the averaging window = {min(xmin[i20][0], xmin[i20][1]):.1e}',
      'x_1, x_2 stay above the escape threshold 1e-9 of the scan')
exact('Fig. S3(c): hatched band starts where escapes begin (first theta with an escape)', 0.18,
      min(t for t in grid if stay[t] < n_ic[t]))
exact('Fig. S3(c): shaded band starts where every trajectory escapes', 0.25,
      min(t for j, t in enumerate(grid) if not any(stay[u] for u in grid[j:])))
holds('Fig. 3(a), Fig. S3(a),(b): the attractors drawn at theta = -0.45, -0.20, 0.08 are a fixed point, a limit cycle, chaos',
      [regime[i] for i in (0, 3, 6)] == ['fixed point', 'limit cycle', 'chaos'],
      ', '.join(regime[i] for i in (0, 3, 6)), 'regimes at these values')

# ============================================================================================== Sec. S7
section('Sec. S7  Several modified species and other models (Fig. S4)')
exact('species in the community of Fig. S4(b)', 7, len(use['dim_r']))
exact('random modifications drawn for each |K|', 300, int(use['dim_draws']))
exact('draws that keep all seven species, |K| = 1, 2, 3', (299, 160, 173), tuple(int(v) for v in use['dim_kept']))
exact('numerical ranks of the points (x*, 1)', (2, 3, 4), tuple(int(v) for v in use['dim_rank']))
for q in (1, 2, 3):
    noise(f'   |K| = {q}: next singular value / first', '1e-16', float(use[f'dim_sv_{q}'][q + 1]), power=True)
rk_u, sv_u = numerical_rank(np.vstack([use['P2cr'], np.ones(use['P2cr'].shape[1])]), rtol=1e-10)
exact('consumption rate u_11 shared by two equations: rank', 3, rk_u)
rounds('   second singular value / first', '0.07', sv_u[1] / sv_u[0])
rounds('   third singular value / first', '4e-4', sv_u[2] / sv_u[0])
noise('   fourth singular value / first', '3e-16', sv_u[3] / sv_u[0])
rk_m, sv_m = numerical_rank(np.vstack([use['Pcr'], np.ones(use['Pcr'].shape[1])]), rtol=1e-10)
rr_, AA_ = use['rr'], use['AA']                       # y = (R_1..R_4, N_1..N_3); consumer 1 is index 4
exact('consumer-resource model: resources (logistic); consumers', (4, 3),
      (int(np.sum(np.diag(AA_) < 0)), int(np.sum(np.diag(AA_) == 0))))
G1 = rr_[:, None] + AA_ @ use['Pcr']                  # growth rates of the unmodified model at the stored equilibria
holds('   mortality m_1 varied: only the equation of consumer 1 is modified',
      np.max(np.abs(np.delete(G1, 4, axis=0))) < 1e-12 and np.ptp(G1[4]) > 0.1,
      f'other equations hold to {np.max(np.abs(np.delete(G1, 4, axis=0))):.1e}', 'to 1e-12')
exact('   rank of the points (R, N, 1)', 2, rk_m)
G2 = rr_[:, None] + AA_ @ use['P2cr']
holds('   u_11 varied: only the equations of resource 1 and consumer 1 are modified',
      np.max(np.abs(np.delete(G2, [0, 4], axis=0))) < 1e-12 and np.ptp(G2[0]) > 0.01 and np.ptp(G2[4]) > 0.01,
      f'other equations hold to {np.max(np.abs(np.delete(G2, [0, 4], axis=0))):.1e}', 'to 1e-12')
rk_r, sv_r = numerical_rank(use['Xr'], rtol=1e-10)
exact('replicator equation: strategies', 4, use['Xr'].shape[0])
exact('   payoff entry a_12 varied: rank of the rest points', 2, rk_r)
PX = use['Pay'] @ use['Xr']                           # payoffs with the unmodified matrix at the stored rest points
holds('   only the rest-point condition of strategy 1 changes: strategies 2 to 4 keep equal payoffs',
      np.max(np.abs(PX[1:] - PX[1])) < 1e-12 and np.max(np.abs(PX[0] - PX[1])) > 0.01,
      f'payoffs of strategies 2 to 4 equal to {np.max(np.abs(PX[1:] - PX[1])):.1e}', 'to 1e-12')

# ============================================================================================== captions of the Letter
section('Captions of Figs. 1 and 2 of the Letter')
exact('Fig. 1: species in the community; modified species; modifications', (5, 3, 4), (len(r1), k1 + 1, len(XS)))
exact('Fig. 1(b): the changed coefficient is A_31', 'A_31', f'A_{k1 + 1}{l1 + 1}')
exact('Fig. 2: species in the community; changed coefficient', (6, 'A_23'), (n2, f'A_{k2 + 1}{l2 + 1}'))
exact('Fig. 2(a): changes of composition along the sweep', 4, len(segs2) - 1)
lost_n = gained_n = 0
for q in range(1, len(segs2)):
    before, after = SUPP2[segs2[q - 1][0]], SUPP2[segs2[q][0]]
    lost_n += sum(1 for i in range(n2) if before[i] and not after[i])
    gained_n += sum(1 for i in range(n2) if after[i] and not before[i])
exact('Fig. 2(b): extinctions; invasions', (2, 2), (lost_n, gained_n))
holds('Fig. 2(a): at the left end species 2 is absent and the change has no effect',
      SUPP2[0][k2] == 0 and np.max(np.abs(X2[:, segs2[0][0]:segs2[0][1]] - X2[:, [0]])) == 0,
      f'species present: {SUPP2[0]}; the equilibrium does not move', 'first branch of the sweep')
holds('Fig. 2(b): a press on r_2 over the same range of effective press traces the same path',
      PATH['A_23']['worst'] < 1e-12, f'max |x*(theta) - x*_press(s(theta))| = {PATH["A_23"]["worst"]:.1e}', 'to 1e-12')

# ============================================================================================== output
if __name__ == '__main__':
    current = None
    for sec, ok, what, text, found, rule, differ in ROWS:
        if sec != current:
            print(f'\n=== {sec} ===')
            print(f'{"":4s}  {"quantity":<96s}  {"in the text":<24s}  found')
            current = sec
        print(f'{"PASS" if ok else "FAIL"}  {what:<96s}  {text:<24s}  {found}   [{rule}]')
    fails = [row for row in ROWS if not row[1]]
    differ = [row for row in ROWS if row[1] and row[6]]
    print(f'\n=== Summary ===\n{len(ROWS)} checks: {len(ROWS) - len(fails)} PASS, {len(fails)} FAIL')
    if fails:
        print('\nFAIL (the text and the code or data disagree):')
        for sec, ok, what, text, found, rule, _ in fails:
            print(f'  - {what.strip()}: text {text}; found {found}')
    if differ:
        print('\nPASS by a tolerance, but the digits printed in the text differ from this run (rounding-error level, '
              'or chaotic trajectories):')
        for sec, ok, what, text, found, rule, _ in differ:
            print(f'  - {what.strip()}: text {text}; found {found}   [{rule}]')
    sys.exit(1 if fails else 0)
