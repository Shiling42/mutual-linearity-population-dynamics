"""
run_all.py -- rebuilds everything in this repository, in order: the six experiments (they write data/), the seven figures
(figures/), and the check of the numbers quoted in the Supplemental Material (verify_sm_numbers.py).

Run:  python3 -B run_all.py [workers]      all steps, about four minutes; workers = processes of the crisis scan
                                           (default 4)
      python3 -B run_all.py figures        only the figures and the check, from the files already in data/

The output of every step goes to logs/<step>.log, and the run times to logs/run_times.txt. The folder
reference_logs/ holds the same files from the run that produced the data and figures stored here.
Exit status: 0 if every step ran without error and every check of verify_sm_numbers.py passed.
"""
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, 'logs')
EXPERIMENTS = ['exp_collinear', 'exp_stepwise', 'exp_divergence_stats', 'exp_use_and_scope', 'exp_chaos',
               'crisis_scan_extended']          # exp_use_and_scope reads the output of exp_stepwise
FIGURES = ['prl_fig1', 'prl_fig2', 'prl_fig3', 'prl_figS1', 'figS2_uses', 'figS3_chaos', 'figS4_scope']


def step(name, script, args=()):
    t0 = time.time()
    with open(os.path.join(LOGS, name + '.log'), 'w') as fh:
        rc = subprocess.run([sys.executable, '-B', os.path.join(HERE, script), *args], stdout=fh,
                            stderr=subprocess.STDOUT, cwd=HERE).returncode
    sec = time.time() - t0
    print(f'{name:24s} {sec:7.1f} s   exit status {rc}', flush=True)
    return name, sec, rc


if __name__ == '__main__':
    only_figures = 'figures' in sys.argv[1:]
    workers = [a for a in sys.argv[1:] if a.isdigit()][:1]
    os.makedirs(LOGS, exist_ok=True)
    import numpy, scipy, matplotlib
    head = (f'Python {platform.python_version()}, NumPy {numpy.__version__}, SciPy {scipy.__version__}, '
            f'Matplotlib {matplotlib.__version__}; {platform.platform()}; {os.cpu_count()} processors')
    print(head)
    done = []
    if not only_figures:
        for name in EXPERIMENTS:
            done.append(step(name, os.path.join('code', name + '.py'), workers if name == 'crisis_scan_extended' else ()))
    for name in FIGURES:
        done.append(step(name, os.path.join('figure_scripts', name + '.py')))
    crashed = [name for name, sec, rc in done if rc != 0]
    check = step('verify_sm_numbers', 'verify_sm_numbers.py')
    with open(os.path.join(LOGS, 'verify_sm_numbers.log')) as fh:
        summary = [line.strip() for line in fh if ' checks: ' in line]
    print('verify_sm_numbers.py:', summary[0] if summary else 'no summary (see logs/verify_sm_numbers.log)')
    with open(os.path.join(LOGS, 'run_times.txt'), 'w') as fh:
        fh.write(head + '\n' + time.strftime('%Y-%m-%d') + '\n\n')
        for name, sec, rc in done + [check]:
            fh.write(f'{name:24s} {sec:7.1f} s   exit status {rc}\n')
        fh.write(f'{"total":24s} {sum(sec for _, sec, _ in done + [check]):7.1f} s\n')
        fh.write('\nverify_sm_numbers.py: ' + (summary[0] if summary else 'no summary') + '\n')
    if crashed:
        print('steps that ended with an error:', ', '.join(crashed))
    if check[2] != 0:
        print('verify_sm_numbers.py reports a mismatch or could not run: see logs/verify_sm_numbers.log')
    sys.exit(1 if crashed or check[2] != 0 else 0)
