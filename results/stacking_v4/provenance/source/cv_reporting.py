"""Inner-validation metrics only: never substitute outer single-patient scores."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, confusion_matrix)

METRICS = ('accuracy', 'roc_auc', 'precision', 'recall', 'specificity', 'sensitivity')


def classification_metrics(y, probability, prediction=None):
    y = np.asarray(y)
    probability = np.asarray(probability)
    prediction = probability >= .5 if prediction is None else np.asarray(prediction)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y, prediction)),
                roc_auc=float(roc_auc_score(y, probability)),
                precision=float(precision_score(y, prediction, zero_division=0)),
                recall=float(recall_score(y, prediction, zero_division=0)),
                specificity=float(tn / (tn + fp)),
                sensitivity=float(recall_score(y, prediction, zero_division=0)))


def summarize_validation(validation, *, inner_folds):
    folds = validation['folds']
    if len(folds) != inner_folds:
        raise ValueError('Incomplete inner validation evidence')
    row = dict(best_trial=validation['best_trial'], selected_k=validation.get('selected_k'))
    for metric in METRICS:
        values = np.array([fold[metric] for fold in folds], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Invalid inner validation metric')
        row[metric] = float(values.mean())
        row[metric + '_std'] = float(values.std(ddof=1))
        for index, value in enumerate(values, 1):
            row[f'fold_{index}_{metric}'] = float(value)
    return row


def write_cv_metrics(directory, checkpoints, *, seed, inner_folds):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    models = sorted({e['model'] for e in checkpoints})
    for model in models:
        rows = []
        for e in sorted((e for e in checkpoints if e['model'] == model), key=lambda e: e['fold']):
            rows.append(dict(seed=seed, model=model, outer_fold=e['fold'],
                left_out_source_row=e['source_row'], sampler_seed=e['sampler_seed'],
                inner_folds=inner_folds, score_scope='selected-trial inner validation',
                **summarize_validation(e['validation'], inner_folds=inner_folds)))
        frame = pd.DataFrame(rows)
        if frame.outer_fold.duplicated().any() or frame.left_out_source_row.duplicated().any():
            raise ValueError('Duplicate outer-patient validation evidence')
        target = directory / f'{model}.csv'
        temporary = target.with_suffix('.csv.tmp')
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
