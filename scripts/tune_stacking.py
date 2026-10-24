"""Rerun Stacking with matched validation and complete saved experiment evidence.

The reference individual-model experiment fixes the comparison protocol. Equal
trial budgets are explicitly distinguished from equal optimisation difficulty.
Existing reported results are never overwritten by this entry point.
"""
import argparse
import ast
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from feature_selection_evidence import file_sha256, source_fingerprint
from optuna_lopo import run_optuna_lopo, suggest_parameters, selection_parameters
from utils import workflow_data, frame_signature

BASES = ('xgboost', 'lightgbm', 'ridge')
FAMILIES = ('logistic', 'random_forest', 'xgboost', 'lightgbm', 'stacking')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class SearchSpace:
    """Record the actual suggest calls without maintaining duplicate ranges."""
    def __init__(self):
        self.parameters = {}

    def suggest_float(self, name, low, high, **kwargs):
        self.parameters[name] = dict(type='float', low=low, high=high, **kwargs)
        return low

    def suggest_int(self, name, low, high, **kwargs):
        self.parameters[name] = dict(type='int', low=low, high=high, **kwargs)
        return low

    def suggest_categorical(self, name, choices):
        self.parameters[name] = dict(type='categorical', choices=list(choices))
        return choices[0]


def search_spaces(k_values):
    spaces = {}
    for name in FAMILIES:
        trial = SearchSpace()
        suggest_parameters(trial, name)
        selection_parameters(trial, name, k_values)
        spaces[name] = trial.parameters
    return spaces


def check_reference(args, X, y):
    reference = args.reference_dir.resolve()
    config = json.loads((reference / 'run_config.json').read_text())
    progress = json.loads((reference / 'progress.json').read_text())
    if not progress['complete'] or progress['evaluated_folds'] != len(X):
        raise ValueError('Reference evaluation must be complete on this cohort')
    if set(config['models']) != set(FAMILIES) - {'stacking'}:
        raise ValueError('Reference must contain the four individual-model configurations')
    partial = args.max_folds is not None and args.max_folds < len(X)
    required = dict(signature=frame_signature(X, y), rows=len(X), seed=args.seed,
                    inner_folds=3, threshold=.5, selection='rfe',
                    k_values=sorted(set(args.k_values)), objective='mean inner-fold ROC-AUC',
                    continuous_transform='PowerTransformer(method=yeo-johnson, standardize=True)',
                    sampler_seed_rule='seed + outer fold; independent of model list/order')
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'Reference mismatch for {key}; keep the comparison protocol fixed')
    if args.trials != config['trials'] and not partial:
        raise ValueError('A complete comparison must match the reference trial budget; use a new matched experiment for budget sensitivity')
    if config.get('source_fingerprint') != source_fingerprint(ROOT):
        raise ValueError('Reference training-source fingerprint differs; audit or rerun all affected configurations')
    if any(version(package) != saved for package, saved in config['packages'].items()):
        raise ValueError('Package versions differ from the reference experiment')
    predictions = pd.read_csv(reference / 'predictions.csv')
    for model in config['models']:
        frame = predictions.loc[predictions.model.eq(model)].set_index('source_row').sort_index()
        if not frame.index.equals(X.index):
            raise ValueError('Reference predictions have incomplete or duplicate source rows')
        np.testing.assert_array_equal(frame.actual_present, y)
        np.testing.assert_array_equal(frame.predicted_present, frame.probability_present.ge(.5))
    if args.output_dir.resolve() == reference:
        raise ValueError('Use a new Stacking output directory, not the reference directory')
    return config, partial


def make_manifest(args, X, y, reference, partial):
    sources = dict(source_fingerprint(ROOT))
    sources['scripts/tune_stacking.py'] = file_sha256(Path(__file__))
    return dict(
        manifest_version=1, model='stacking', bases=list(BASES),
        cohort_signature=frame_signature(X, y), rows=len(X), source_features=list(X.columns),
        reference_directory=str(args.reference_dir.resolve()),
        reference_config_sha256=file_sha256(args.reference_dir / 'run_config.json'),
        reference_predictions_sha256=file_sha256(args.reference_dir / 'predictions.csv'),
        raw_data_sha256=file_sha256(ROOT / 'data/raw/dataset_gallstone'),
        source_sha256=sources, packages=reference['packages'],
        python_version=platform.python_version(),
        protocol=dict(seed=args.seed, trials=args.trials, inner_folds=3,
                      k_values=sorted(set(args.k_values)), threshold=.5,
                      selection='CART RFE before transformation and one-hot encoding',
                      objective='mean inner-fold ROC-AUC', sampler='Optuna TPESampler; versioned library defaults',
                      sampler_seed='(seed + zero-based outer fold) modulo 2**32',
                      stacking_cv='StratifiedKFold(3, shuffle=True, random_state=seed)',
                      meta_classifier='StandardScaler + LogisticRegression(C=1, max_iter=3000)',
                      passthrough=False, save_fold_models=True),
        search_spaces=search_spaces(sorted(set(args.k_values))),
        budget_matches_reference=args.trials == reference['trials'],
        comparison_scope='Same validation design and candidate-trial budget; not equal compute, search dimension, or proven convergence',
        interpretation='Internal descriptive comparison; no independently validated winner or statistical superiority claim',
    )


def split_records(rows, X, y, seed):
    rows = pd.Index(rows)
    return [dict(fold=i, training_source_rows=rows[a].tolist(),
                 validation_source_rows=rows[b].tolist())
            for i, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed)
                                      .split(X.loc[rows], y.loc[rows]))]


def parse(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def export_validation(out, X, y, seed):
    """Export completed checkpoints only; retain trial and cross-fit boundaries."""
    trials_long, inner_long, selected = [], [], []
    for file in sorted((out / 'folds').glob('fold_*_stacking.json')):
        cp = json.loads(file.read_text())
        train = cp['training_source_rows']
        assert cp['source_row'] not in train and set(train) == set(X.index) - {cp['source_row']}
        expected = split_records(train, X, y, seed)
        saved = cp['validation']['inner_splits']
        assert [[s['training_source_rows'], s['validation_source_rows']] for s in saved] == [
            [s['training_source_rows'], s['validation_source_rows']] for s in expected]
        crossfits = dict(
            provenance='Deterministic split maps derived from saved training IDs and declared Stacking splitter; common to all three bases and trials',
            source_row=cp['source_row'], outer_fold=cp['fold'], bases=list(BASES),
            outer_training_base_crossfits=split_records(train, X, y, seed),
            inner_training_base_crossfits=[dict(inner_fold=s['inner_fold'],
                crossfits=split_records(s['training_source_rows'], X, y, seed)) for s in saved])
        write_json(out / 'crossfit_partitions' / f"fold_{cp['fold']:04d}.json", crossfits)
        table = pd.read_csv(file.with_name(file.stem + '_trials.csv'))
        winner = table.loc[table.number.eq(cp['validation']['best_trial'])].iloc[0]
        np.testing.assert_allclose(winner.value, table.value.max(), atol=1e-12)
        np.testing.assert_allclose(winner.value, cp['inner_cv_auc'], atol=1e-12)
        selected.append(dict(outer_fold=cp['fold'], source_row=cp['source_row'],
            best_trial=cp['validation']['best_trial'], selected_k=cp['validation']['selected_k'],
            inner_cv_auc=cp['inner_cv_auc'], fit_predict_seconds=cp['fit_predict_seconds']))
        for trial in table.itertuples(index=False):
            common = dict(outer_fold=cp['fold'], source_row=cp['source_row'], trial=trial.number,
                          selected=trial.number == cp['validation']['best_trial'])
            params = parse(trial.user_attrs_pipeline_params)
            trials_long.append(dict(**common, state=trial.state, objective=trial.value,
                selected_k=trial.params_feature_count, duration_seconds=pd.Timedelta(trial.duration).total_seconds(),
                parameters=json.dumps(params, sort_keys=True)))
            folds = parse(trial.user_attrs_validation_metrics)
            names = parse(trial.user_attrs_inner_feature_selection)
            for i, metrics in enumerate(folds):
                assert set(names[i]['selected_features']) == set(BASES)
                assert all(len(v) == trial.params_feature_count for v in names[i]['selected_features'].values())
                inner_long.append(dict(**common, inner_fold=i, **metrics,
                    selected_features=json.dumps(names[i]['selected_features'], sort_keys=True)))
    directory = out / 'validation'
    directory.mkdir(exist_ok=True)
    pd.DataFrame(trials_long).to_csv(directory / 'trial_summary.csv', index=False)
    pd.DataFrame(inner_long).to_csv(directory / 'all_trial_inner_metrics.csv', index=False)
    pd.DataFrame(selected).to_csv(directory / 'selected_trial_summary.csv', index=False)
    write_json(directory / 'optimization_summary.json', dict(completed_outer_folds=len(selected),
        total_outer_folds=len(X), complete=len(selected) == len(X),
        completed_trials=len(trials_long), recorded_inner_evaluations=len(inner_long),
        parameters_searched=len(search_spaces([10, 20, 30, 38])['stacking']),
        total_recorded_trial_seconds=sum(r['duration_seconds'] for r in trials_long),
        total_recorded_fit_predict_seconds=sum(r['fit_predict_seconds'] for r in selected),
        metric_scope='Inner metrics describe optimisation; use pooled outer predictions for reported performance',
        timing_scope='Observed run time, not a controlled equal-compute benchmark'))


def run(args):
    X, y = workflow_data()
    if args.trials < 1 or (args.max_folds is not None and args.max_folds < 1):
        raise ValueError('Positive trials and max-folds are required')
    reference, partial = check_reference(args, X, y)
    manifest = make_manifest(args, X, y, reference, partial)
    out = args.output_dir.resolve()
    path = out / 'experiment_manifest.json'
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Experiment provenance changed; use a new output directory')
        for name, digest in manifest['source_sha256'].items():
            snapshot = out / 'provenance/source' / name
            if not snapshot.is_file() or file_sha256(snapshot) != digest:
                raise ValueError('Source snapshot is missing or altered')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Use a new empty directory; existing experiment evidence is protected')
        out.mkdir(parents=True, exist_ok=True)
        for name in manifest['source_sha256']:
            target = out / 'provenance/source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (out / 'provenance/pip-freeze.txt').write_text(freeze)
        write_json(path, manifest)
    event = dict(started_utc=timestamp(), command=sys.argv, partial_request=partial,
                 max_folds=args.max_folds, platform=platform.platform(), cpu_count=os.cpu_count(),
                 python=sys.version, executable=sys.executable,
                 thread_environment={key: os.environ.get(key) for key in (
                     'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    event_path = out / 'provenance/invocations' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    event['status'] = 'running'
    write_json(event_path, event)
    try:
        run_optuna_lopo(models=['stacking'], seed=args.seed, trials=args.trials,
            inner_folds=3, top_k=min(reference['top_k'], X.shape[1]),
            k_values=args.k_values, output_dir=out, max_folds=args.max_folds,
            save_fold_models=True)
        event['status'] = 'completed_invocation'
    except BaseException as error:
        event['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        event['error_type'] = type(error).__name__
        raise
    finally:
        event['ended_utc'] = timestamp()
        write_json(event_path, event)
        export_validation(out, X, y, args.seed)
    print(f'Stacking evidence: {out}')
    print('Validation exports describe tuning; final performance comes from outer predictions only.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=ROOT / 'results/dynamic_individual_v4')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/stacking_v4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 20, 30, 38])
    parser.add_argument('--max-folds', type=int, help='Incomplete smoke run only; never report as full evaluation')
    parser.add_argument('--dry-run', action='store_true', help='Check matching protocol and print planned provenance without fitting')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
