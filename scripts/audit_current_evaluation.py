"""Read-only audit of the five completed paper evaluations; writes audit evidence only."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from feature_selection_evidence import file_sha256, selection_records, source_fingerprint
from utils import workflow_data, frame_signature, load_pickle, make_preparation


def parse(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def main():
    X, y = workflow_data()
    signature = frame_signature(X, y)
    paths = {name: ROOT / 'results' / ('stacking_v3' if name == 'stacking' else 'dynamic_individual_v4')
             for name in ('logistic', 'random_forest', 'xgboost', 'lightgbm', 'stacking')}
    result = {'scope': 'All current paper checkpoints/trials; all 1276 individual saved pipeline predictions; Stacking source and reconstructed evidence, not full retuning',
              'rows': len(X), 'predictors': len(X.columns), 'missing_cells': int(X.isna().sum().sum()),
              'duplicate_predictor_rows': int(X.duplicated().sum()), 'signature': signature, 'models': {}}
    all_configs = []
    for name, source in paths.items():
        cfg = json.loads((source / 'run_config.json').read_text())
        progress = json.loads((source / 'progress.json').read_text())
        assert progress['complete'] and progress['evaluated_folds'] == len(X)
        assert cfg['signature'] == signature and cfg['k_values'] == [10, 20, 30, 38]
        assert cfg['seed'] == 42 and cfg['trials'] == 20 and cfg['inner_folds'] == 3 and cfg['threshold'] == .5
        if name != 'stacking':
            assert cfg['source_fingerprint'] == source_fingerprint(ROOT)
        all_configs.append(cfg)
        pred = pd.read_csv(source / 'predictions.csv')
        pred = pred.loc[pred.model.eq(name)].set_index('source_row').sort_index()
        assert pred.index.equals(X.index) and len(pred) == len(X)
        np.testing.assert_array_equal(pred.actual_present, y)
        np.testing.assert_array_equal(pred.predicted_present, pred.probability_present.ge(.5))
        total_trials = saved_pipelines = recorded_partitions = recorded_selections = 0
        inner_scores = []
        for fold, row in enumerate(X.index):
            cp = json.loads((source / 'folds' / f'fold_{fold:04d}_{name}.json').read_text())
            assert cp['fold'] == fold and cp['source_row'] == row and cp['actual'] == y.loc[row]
            assert cp['sampler_seed'] == 42 + fold
            np.testing.assert_allclose(cp['probability'], pred.loc[row, 'probability_present'], atol=1e-12)
            train = X.drop(index=row)
            expected_splits = [dict(inner_fold=i, training_source_rows=train.index[a].tolist(),
                                    validation_source_rows=train.index[b].tolist())
                               for i, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(train, y.drop(index=row)))]
            if 'training_source_rows' in cp:
                assert cp['training_source_rows'] == train.index.tolist()
                assert cp['validation']['inner_splits'] == expected_splits
                recorded_partitions += 3
            trials = pd.read_csv(source / 'folds' / f'fold_{fold:04d}_{name}_trials.csv')
            assert len(trials) == 20 and trials.state.eq('COMPLETE').all()
            assert trials.number.is_unique and np.isfinite(trials.value).all()
            for trial in trials.itertuples(index=False):
                k = int(trial.params_feature_count)
                assert k in cfg['k_values']
                scores = parse(trial.user_attrs_inner_auc)
                np.testing.assert_allclose(np.mean(scores), trial.value, atol=1e-12)
                params = parse(trial.user_attrs_pipeline_params)
                keys = [f'{base}__select__n_features_to_select' for base in ('xgboost', 'lightgbm', 'ridge')] if name == 'stacking' else ['select__n_features_to_select']
                assert all(params[key] == k for key in keys)
                metrics = parse(trial.user_attrs_validation_metrics)
                np.testing.assert_allclose([v['roc_auc'] for v in metrics], scores, atol=1e-12)
                if hasattr(trial, 'user_attrs_inner_feature_selection'):
                    selections = parse(trial.user_attrs_inner_feature_selection)
                    assert len(selections) == 3
                    for selected in selections:
                        for fields in selected['selected_features'].values():
                            assert len(fields) == k and len(set(fields)) == k and set(fields) <= set(X.columns)
                            recorded_selections += 1
                total_trials += 1
            best = trials.loc[trials.number.eq(cp['validation']['best_trial'])].iloc[0]
            np.testing.assert_allclose(best.value, trials.value.max(), atol=1e-12)
            np.testing.assert_allclose(best.value, cp['inner_cv_auc'], atol=1e-12)
            assert cp['validation']['selected_k'] == best.params_feature_count
            expected_params = parse(best.user_attrs_pipeline_params)
            assert set(expected_params) == set(cp['best_params'])
            for key in expected_params:
                if isinstance(expected_params[key], (int, float)):
                    np.testing.assert_allclose(expected_params[key], cp['best_params'][key], atol=1e-12)
                else:
                    assert expected_params[key] == cp['best_params'][key]
            np.testing.assert_allclose([v['roc_auc'] for v in cp['validation']['folds']], parse(best.user_attrs_inner_auc), atol=1e-12)
            inner_scores.append(float(best.value))
            if name != 'stacking':
                artifact = source / cp['model_artifact']['path']
                assert file_sha256(artifact) == cp['model_artifact']['sha256']
                model = load_pickle(artifact)
                assert list(model.named_steps) == ['impute', 'select', 'transform', 'model']
                transform = model.named_steps['transform']
                power = transform.named_transformers_['continuous'].named_steps['yeo_johnson']
                encoder = transform.named_transformers_['categorical']
                assert power.method == 'yeo-johnson' and power.standardize and encoder.handle_unknown == 'ignore'
                assert selection_records(model, X.columns, name) == cp['feature_selection']
                assert int(model.named_steps['select'].get_support().sum()) == cp['validation']['selected_k']
                with threadpool_limits(limits=1):
                    probability = model.predict_proba(X.loc[[row]])[0, list(model.classes_).index(1)]
                np.testing.assert_allclose(probability, cp['probability'], atol=1e-8, rtol=0)
                saved_pipelines += 1
        prob = pred.probability_present
        decision = prob.ge(.5)
        tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
        calculated = dict(roc_auc=roc_auc_score(y, prob), accuracy=accuracy_score(y, decision),
                          precision=precision_score(y, decision), recall=recall_score(y, decision),
                          specificity=tn/(tn+fp), f1=f1_score(y, decision), fnr=fn/(fn+tp))
        stored = pd.read_csv(source / 'metrics.csv').set_index('model').loc[name]
        for key, value in calculated.items():
            np.testing.assert_allclose(value, stored[key], atol=1e-10, rtol=0)
        result['models'][name] = dict(checkpoints=len(X), completed_trials=total_trials, saved_pipelines_verified=saved_pipelines,
            recorded_inner_partitions_verified=recorded_partitions, trial_inner_selections_verified=recorded_selections,
            metrics=calculated, confusion=dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)),
            mean_best_inner_auc=float(np.mean(inner_scores)), original_source_fingerprint_present='source_fingerprint' in cfg)
        print('Verified:', name, result['models'][name], flush=True)
    reference = all_configs[0]
    for cfg in all_configs[1:]:
        for key in ('signature', 'seed', 'trials', 'inner_folds', 'threshold', 'selection', 'k_values', 'continuous_transform', 'objective', 'packages', 'sampler_seed_rule'):
            assert cfg[key] == reference[key]
    # Reproduce the legacy visualisation incompatibility without changing exports.
    prep = make_preparation(X, 42, top_k=20)
    with threadpool_limits(limits=1): prep.fit(X.iloc[1:], y.iloc[1:])
    numeric = prep.named_steps['transform'].named_transformers_['continuous']
    assert 'standardize' not in numeric.named_steps and numeric.named_steps['yeo_johnson'].standardize
    notebook = json.loads((ROOT / 'notebooks/transformation_feature_engineering.ipynb').read_text())
    result['legacy_visualisation_obsolete_scaler_lookup_confirmed'] = any('numeric.named_steps["standardize"]' in ''.join(c.get('source', [])) for c in notebook['cells'])
    result['passed'] = True
    (ROOT / 'docs/audit/current_evaluation_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print('All current evidence checks passed; legacy workflow issue remains separately documented.', flush=True)


if __name__ == '__main__':
    main()
