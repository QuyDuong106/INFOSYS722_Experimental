"""Run ordered notebook/script stages, stopping at the first failure."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'eda': 'notebooks/gallstone_eda.ipynb',
    'prepare': 'notebooks/transformation_feature_engineering.ipynb',
    'holdout': 'notebooks/holdout_method.ipynb',
    'lopo': 'optuna_lopo.py',
    'tuning': 'notebooks/tuning_model.ipynb',
    'errors': 'notebooks/error_analysis.ipynb',
    'patterns': 'scripts/iteration_pattern_analysis.py',
    'figures': 'build_modeling_figures.py',
    'check': None,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-stage', choices=STAGES, default='eda')
    parser.add_argument('--to-stage', choices=STAGES, default='check')
    parser.add_argument('--kernel', default='gallstone-iteration3')
    parser.add_argument('--list', action='store_true', help='Print stages without executing')
    parser.add_argument('--trials', type=int, default=20, help='Optuna trials per outer fold/model')
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', nargs='+', choices=('logistic', 'random_forest', 'xgboost', 'lightgbm', 'stacking', 'dummy_prior'))
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, help='Directory for nested Leave One Patient Out evidence')
    parser.add_argument('--max-folds', type=int, help='Partial Leave One Patient Out smoke run')
    parser.add_argument('--refit-final', action='store_true')
    parser.add_argument('--fixed-lopo', action='store_true', help='Run the historical fixed-parameter notebook instead')
    args = parser.parse_args()
    names = list(STAGES)
    start, end = names.index(args.from_stage), names.index(args.to_stage)
    if start > end:
        parser.error('--from-stage must precede --to-stage')
    if args.max_folds is not None and (start != names.index('lopo') or end != start):
        parser.error('--max-folds requires --from-stage lopo --to-stage lopo')
    if args.fixed_lopo and (args.max_folds is not None or args.refit_final):
        parser.error('--max-folds and --refit-final apply only to Optuna, not --fixed-lopo')
    if args.list:
        for name in names[start:end + 1]:
            print(f'{name:10} {"notebooks/leave_one_out_method.ipynb" if name == "lopo" and args.fixed_lopo else STAGES[name] or "unittest discovery"}')
        return
    # Capture the environment for every invocation, including resumed runs.
    logs = ROOT / 'results' / 'runs'
    logs.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (logs / f'{stamp}-environment.txt').write_text(
        sys.version + '\n' + subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], text=True))
    for name in names[start:end + 1]:
        path = 'notebooks/leave_one_out_method.ipynb' if name == 'lopo' and args.fixed_lopo else STAGES[name]
        print(f'Running {name}: {path or "tests"}', flush=True)
        if name == 'lopo' and not args.fixed_lopo:
            command = [sys.executable, path, '--trials', str(args.trials),
                       '--inner-folds', str(args.inner_folds), '--seed', str(args.seed),
                       '--top-k', str(args.top_k)]
            if args.models:
                command += ['--models', *args.models]
            if args.output_dir:
                command += ['--output-dir', str(args.output_dir.resolve())]
            if args.max_folds is not None:
                command += ['--max-folds', str(args.max_folds)]
            if args.refit_final:
                command += ['--refit-final']
        elif path is None:
            command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
        elif path.endswith('.ipynb'):
            command = [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook',
                       '--execute', '--inplace', f'--ExecutePreprocessor.kernel_name={args.kernel}',
                       '--ExecutePreprocessor.timeout=-1', path]
        else:
            command = [sys.executable, path]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
