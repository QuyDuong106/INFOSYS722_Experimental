"""Unweighted hard majority voting from matching complete saved LOPO predictions.

ROC-AUC uses the fraction of positive votes (not averaged probabilities).
An even-voter tie follows the explicit --tie-break rule, negative by default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from cv_reporting import classification_metrics


def combine_predictions(runs, *, tie_break='negative'):
    if tie_break not in {'negative', 'positive'}:
        raise ValueError('Choose negative or positive tie breaking')
    if len(runs) < 2:
        raise ValueError('At least two model prediction directories are required')
    frames, provenance, reference = [], [], None
    names = set()
    for folder in map(Path, runs):
        config = json.loads((folder / 'run_config.json').read_text())
        progress = json.loads((folder / 'progress.json').read_text())
        frame = pd.read_csv(folder / 'predictions.csv')
        if not progress.get('complete') or progress.get('evaluated_folds') != config['rows']:
            raise ValueError(f'Incomplete LOPO run: {folder}')
        if len(frame) != config['rows'] or frame.source_row.duplicated().any():
            raise ValueError(f'Expected exactly one held-out prediction per patient: {folder}')
        if frame.model.nunique() != 1 or frame.seed.nunique() != 1:
            raise ValueError('Each input must contain one model and one seed')
        name = str(frame.model.iloc[0])
        if name in names:
            raise ValueError(f'Duplicate voting model: {name}')
        names.add(name)
        if int(frame.seed.iloc[0]) != config['seed'] or config.get('threshold') != .5:
            raise ValueError('Prediction seed/configuration or threshold mismatch')
        if frame.evaluation.nunique() != 1 or frame.evaluation.iloc[0] != 'lopo_optuna':
            raise ValueError('Use nested Optuna LOPO predictions, not training/holdout predictions')
        frame = frame.sort_values('source_row').reset_index(drop=True)
        prob = frame.probability_present.to_numpy()
        if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
            raise ValueError('Invalid prediction probability')
        if set(frame.actual_present.unique()) != {0, 1}:
            raise ValueError('Both binary outcome classes are required')
        if not np.array_equal(frame.predicted_present, (prob >= .5).astype(int)):
            raise ValueError('Stored decisions do not match the common threshold')
        key = (config['signature'], config['seed'], config['rows'], frame.evaluation.iloc[0])
        if reference is None:
            reference = key
            labels = frame[['source_row', 'actual_present']].copy()
        elif reference != key or not frame[['source_row', 'actual_present']].equals(labels):
            raise ValueError('Cohort signature, seed, patient IDs or labels differ')
        frames.append(frame.predicted_present.to_numpy(dtype=int))
        provenance.append(dict(model=name, directory=str(folder.resolve()),
            search_version=config.get('search_version'),
            predictions_sha256=hashlib.sha256((folder / 'predictions.csv').read_bytes()).hexdigest()))
    votes = np.array(frames).T
    count = votes.sum(axis=1)
    fraction = count / len(frames)
    decisions = (count * 2 >= len(frames) if tie_break == 'positive' else count * 2 > len(frames)).astype(int)
    result = labels.copy()
    result['seed'] = reference[1]
    result['positive_votes'] = count
    result['voters'] = len(frames)
    result['vote_fraction'] = fraction
    result['predicted_present'] = decisions
    for info, votes_for_model in zip(provenance, frames):
        result[f"vote_{info['model']}"] = votes_for_model
    metrics = classification_metrics(result.actual_present, fraction, decisions)
    tn, fp, fn, tp = confusion_matrix(result.actual_present, decisions, labels=[0, 1]).ravel()
    metrics.update(model='majority_voting', seed=reference[1], rows=len(result),
                   f1=float(f1_score(result.actual_present, decisions, zero_division=0)),
                   TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    metadata = dict(signature=reference[0], seed=reference[1], inputs=provenance,
        method='unweighted hard voting', tie_break=tie_break,
        auc_score='fraction of positive votes; not a calibrated probability',
        interpretation='Pooled matching held-out predictions; no weights or subset fitted. '
                       'Retrospective ensemble assessment, not nested ensemble selection.')
    return result, pd.DataFrame([metrics]), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('results/majority_voting'))
    parser.add_argument('--tie-break', choices=('negative', 'positive'), default='negative')
    args = parser.parse_args()
    predictions, metrics, metadata = combine_predictions(args.runs, tie_break=args.tie_break)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Choose an empty output directory to preserve existing ensemble evidence')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / 'predictions.csv', index=False)
    metrics.to_csv(args.output_dir / 'metrics.csv', index=False)
    (args.output_dir / 'voting_config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
