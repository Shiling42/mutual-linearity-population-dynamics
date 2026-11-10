# Code and data for "Mutual Linearity in Nonlinear Population Dynamics"

Shiling Liang

This repository holds the Python code that generates all data and all figures of the Letter and of its Supplemental
Material (SM), the data and figures it produced, and a script that checks every number quoted in the SM.

## Contents

| Folder or file | Content |
|---|---|
| `code/` | The six experiments (`exp_*.py`, `crisis_scan_extended.py`) and the two modules they share (`glv_tools.py`, `ml_core.py`). They need NumPy and SciPy, and they write `data/`. |
| `figure_scripts/` | One script per figure, two style files (`figstyle.py`, `prl_style.py`) and the TikZ parts of Figs. 1, 2 and S4. They read `data/` and write `figures/`. |
| `data/` | Output of the experiments: six `.npz` files and one `.csv` file. |
| `figures/` | The seven figures as PDF, each with a PNG for checking. |
| `verify_sm_numbers.py` | Prints every number quoted in the SM and in the figure captions next to the value found from the code and the data, with a PASS or FAIL mark. |
| `run_all.py` | Runs all experiments, all figure scripts and the check, in order. |
| `reference_logs/` | Printed output and run times of a complete run of this version. That run reproduced `data/` bit for bit and `figures/` pixel for pixel. |
| `requirements.txt`, `LICENSE` | Package versions; license. |

## Requirements

- Python 3 with NumPy, SciPy and Matplotlib. The data and figures stored here were made with Python 3.12.7, NumPy 1.26.4,
  SciPy 1.13.1 and Matplotlib 3.9.2 (`pip install -r requirements.txt`) on macOS 27 (Apple M3, 8 cores).
- `pdflatex` for Figs. 1, 2 and S4, which have a panel drawn in TikZ. Packages: `standalone`, `tikz`, `graphicx`,
  `fontenc`, `helvet`, `sfmath`. TeX Live 2022 was used.
- `pdftocairo` (Poppler; version 25.07 was used) to render the PNG copies of the figures. The scripts of Figs. S2
  and S3 work without it.

The experiments need neither `pdflatex` nor `pdftocairo`.

## Rebuilding everything

```
python3 -B run_all.py
```

This takes about four minutes on the machine named above. It runs the six experiments, the seven figure scripts and
`verify_sm_numbers.py`, and writes the output of each step to `logs/`. Its exit status is 0 if every step ran and
every check passed. `python3 -B run_all.py figures` skips the experiments and uses the files already in `data/`.
All scripts find their folders from their own location, so they can be started from any directory. The option `-B`
keeps Python from writing `__pycache__` folders; it can be left out.

## The experiments

| Script | Computes | Used for | Run time |
|---|---|---|---|
| `code/exp_collinear.py` | Five-species community; species 3 modified in four ways; effective press and distance from the line; the variant with saturating (type II) effects | Fig. 1, Fig. S1(a,b); Secs. S2 and S3 | 1 s |
| `code/exp_stepwise.py` | Six-species community under a change of A_23; 120 random sweeps: changes of composition and the direction after each kink | Fig. 2, Fig. S1(c), Fig. S2; Sec. S4 | 25 s |
| `code/exp_divergence_stats.py` | The same 120 sweeps: comparison with direct integration at 600 settings; sweeps that end in a divergence; how branches end | Secs. S2 and S4 | 25 s |
| `code/exp_use_and_scope.py` | Uses of the lines on the community of Fig. 2; several modified species; consumer-resource model; replicator equation | Figs. S2 and S4; Secs. S5 and S7 | 2 s |
| `code/exp_chaos.py` | Community of Vano et al. (2006) at nine values of theta: time averages, Lyapunov exponents, regimes | Fig. 3, Fig. S3, Table S1; Sec. S6 | 80 s |
| `code/crisis_scan_extended.py` | The two-species state and its invasion growth rates; twelve trajectories at each of 16 values of theta | Fig. S3(c); Sec. S6 | 80 s with 4 processes |

Run `exp_stepwise.py` before `exp_use_and_scope.py`, which reads its output. The other experiments are independent.
`crisis_scan_extended.py` takes the number of processes as an optional argument (default 4).

## One command per figure

| Figure | Command | Reads from `data/` |
|---|---|---|
| Fig. 1 | `python3 -B figure_scripts/prl_fig1.py` | `exp_collinear.npz` |
| Fig. 2 | `python3 -B figure_scripts/prl_fig2.py` | `exp_stepwise_example.npz` |
| Fig. 3 | `python3 -B figure_scripts/prl_fig3.py` | `exp_chaos.npz` |
| Fig. S1 | `python3 -B figure_scripts/prl_figS1.py` | `exp_collinear.npz`, `exp_stepwise_example.npz` |
| Fig. S2 | `python3 -B figure_scripts/figS2_uses.py` | `exp_stepwise_example.npz`, `exp_use.npz` |
| Fig. S3 | `python3 -B figure_scripts/figS3_chaos.py` | `exp_chaos.npz`, `crisis_scan_extended.csv` |
| Fig. S4 | `python3 -B figure_scripts/figS4_scope.py` | `exp_use.npz` (panels c and d; the script recomputes the draws of panel b) |

Each command takes one to three seconds and writes `figures/fig1.pdf`, `fig2.pdf`, `fig3.pdf`, `figS1.pdf`,
`figS2_uses.pdf`, `figS3_chaos.pdf` or `figS4_scope.pdf`, with a PNG of the same name. The scripts of Figs. 1, 2 and
S4 write intermediate files into `figure_scripts/` (`build_*/`, `*_layout.tex`, `prl_fig1_edges.tex`). These can be
deleted.

## Checking the numbers of the Supplemental Material

```
python3 -B verify_sm_numbers.py
```

The script prints one line per number: the quantity, the value in the text, the value found, the rule of comparison
and PASS or FAIL. It ends with a summary that lists every FAIL, and every number that passes by a tolerance although
its printed digits differ. The rules are explained at the top of the script. Its output for the data stored here is
`reference_logs/verify_sm_numbers.log`.

Where the numbers come from:

| Section of the SM | Statement | Generated by |
|---|---|---|
| S2 | Comparison with direct integration at 600 settings | `exp_divergence_stats.py` |
| S2 | Checks of the line (304 equilibria, singular values) | `exp_collinear.py` |
| S2 | Uniqueness of the saturated equilibrium | `exp_collinear.py`, `exp_stepwise.py` and `exp_use_and_scope.py` store the number of saturated equilibria at each point; `verify_sm_numbers.py` adds the sweeps of Fig. S1(c) and sums them |
| S3 | Type II responses | `exp_collinear.py` |
| S3 | Agreement with the press path in Fig. S1(c) | `figure_scripts/prl_figS1.py` (printed), `verify_sm_numbers.py` |
| S4 | Sweeps and kinks | `exp_stepwise.py` |
| S4 | Mutualistic example | `verify_sm_numbers.py` |
| S4 | Sweeps that end in a divergence; branches that end by a change of composition | `exp_divergence_stats.py` |
| S5 | Slopes on the branch without species 4 | `exp_use_and_scope.py` |
| S6 | Table S1 | `exp_chaos.py` |
| S6 | Hopf bifurcation | `verify_sm_numbers.py`; also printed by the scripts of Figs. 3 and S3 |
| S6 | Two-species state, invasion growth rates, crisis scan | `crisis_scan_extended.py` |
| S7 | Ranks and singular values | `exp_use_and_scope.py` |

## What is reproducible

- **Equilibria and statistics** (`exp_collinear.py`, `exp_stepwise.py`, `exp_divergence_stats.py`,
  `exp_use_and_scope.py`). All random numbers come from seeded generators. On another machine the results agree to
  rounding: a run in a second environment, which is not included here, gave abundances that differ by at most
  2e-13. Quantities at the level of rounding errors (singular values near 1e-16, 1 - |cos| near 1e-16) change in
  their digits.
- **Chaotic trajectories** (`exp_chaos.py` at theta >= 0, and `crisis_scan_extended.py`). The initial conditions are
  fixed, and repeated runs on the same machine with the same package versions give identical files. On another
  machine, or with other versions, rounding differences grow along a chaotic trajectory, so the trajectory is a
  different piece of the same attractor. Compared with the run in the second environment, the time averages on the
  chaotic attractors differ by up to 2e-5 and the Lyapunov exponents by up to 8e-4. The largest deviation of a time
  average from the line is 6.5e-5 here and was 5.5e-5 there. The regimes are the same. For the fixed points and
  limit cycles (theta <= -0.1) the time averages agree to 1e-13.
- **Near the end of the chaotic attractor** (theta >= 0.18) a trajectory can escape to the state with two species
  after a long transient. Which trajectories of the scan escape, and when, can therefore differ between machines. The
  same holds for the trajectory of Table S1 at theta = 0.2, which stays for the whole run in the data stored here.

## Conventions

- **Sign.** The paper writes B = -A^{-1} for the press-response matrix and b_k for its k-th column, so that
  x* = x0* + s b_k. Several scripts compute with A^{-1} itself, whose column k is -b_k: `exp_collinear.py`,
  `exp_use_and_scope.py`, `kink_prediction` in `glv_tools.py`, and `figS2_uses.py`. Each script says which of the
  two it uses where it takes the inverse. In the data, the arrays `d` (`exp_collinear.npz`) and `dk` (`exp_use.npz`)
  are column k of A^{-1}. The effective press `s_i` is the paper's s.
- **Names.** Species are numbered from 0 in the code, so species 3 of the paper has index 2. The argument `sigma` of
  `make_community` is gamma of the paper. `SUPP` is the composition: 1 for a present species, 0 for an absent one.
- **Model of Vano et al.** It is written as dx_i/dt = rho_i x_i (1 - sum_j a_ij x_j), and the control is
  a_13 -> a_13 + theta. In the notation of the paper A = -diag(rho) a, so theta of the control is minus the change
  of A_13. The scripts for this model use C = a^{-1}; its first column is b_1, because rho_1 = 1.

## Data files

| File | Main arrays |
|---|---|
| `exp_collinear.npz` | `r`, `A`, `x0`: community of Fig. 1. For the modifications i = 0 (press on r_3), 1 (A_33), 2 (A_31), 3 (mixed): `th_i`, `X_i` (equilibria with all species present), `s_i` (effective press from the displacement), `sid_i` (effective press from Eq. (3)). `C_i`: equilibria of the type II variant. |
| `exp_stepwise_example.npz` | `r`, `A`: community of Fig. 2. `thetas`, `X`, `SUPP`: the sweep of A_23. |
| `exp_stepwise_stats.npz` | `nkinks`: changes of composition in each of the 93 complete sweeps. `cosines`: \|cos\| between Eq. (7) and the observed direction at 67 kinks. `mult`: number of saturated equilibria at each of the 14 520 grid points. `sizes`, `gamma`, `thetas`: number of species of each community, gamma, and the grid of theta. |
| `exp_divergence_stats.npz` | Counts of sweeps and branches; `exceptions`: the setting at which integration and enumeration differ; `resolved`: the sweep with an extinction and the pole inside one grid step. The columns are described at the end of the script. |
| `exp_use.npz` | Data of Fig. S2 (`P1`, `P2`, `pred`, `slopes_int`, `dp`) and of Fig. S4(c,d) (`rr`, `AA`, `Pcr`, `P2cr`: consumer-resource model; `Pay`, `Xr`: replicator equation). `dim_r`, `dim_A`, `dim_kept`, `dim_rank`, `dim_sv_*`: several modified species, as in Fig. S4(b); `verify_sm_numbers.py` reads them. |
| `exp_chaos.npz` | `thetas`; for each of the nine values i: `xeq_i` (interior equilibrium), `xavg_i` (time average), `xmin_i`, `lyap_i`, `remax_i`, `tail_i` and `ttail_i` (the last 800 time units); `regime`, `minx`; the parameters of the integration (`method`, `t_transient`, `t_avg`, `rtol`, `atol`, `lyap_time`, `lyap_dt`). |
| `crisis_scan_extended.csv` | One line per trajectory: `theta`, index of the initial condition, `stayed` (1 or 0), time of the escape (-1 if it stayed). |

## License

MIT License, see `LICENSE`. It covers the code, the data and the figures of this repository.
