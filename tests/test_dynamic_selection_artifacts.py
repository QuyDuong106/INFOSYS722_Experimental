"""Dynamic feature selection persists held-out pipelines and honest fold evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import optuna_lopo as workflow
from feature_selection_evidence import load_fold_pipeline, selection_records, selected_names
from utils import disease_probability


class DynamicSelectionArtifactTests(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=30, n_features=6, n_informative=4,
                                  n_redundant=0, random_state=42)
        self.X = pd.DataFrame(X, index=np.arange(100, 130), columns=list('abcdef'))
        self.y = pd.Series(y, index=self.X.index)

    def test_four_models_persist_selection_models_and_resume_without_retuning(self):
        models = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(workflow, 'workflow_data', return_value=(self.X, self.y)), \
                patch('utils.workflow_data', return_value=(self.X, self.y)):
            kwargs = dict(models=models, trials=2, top_k=4, k_values=[2, 4, 6],
                          output_dir=temp, save_fold_models=True)
            workflow.run_optuna_lopo(max_folds=1, **kwargs)
            with patch.object(workflow, 'tune_training_fold', wraps=workflow.tune_training_fold) as tune:
                workflow.run_optuna_lopo(max_folds=2, **kwargs)
                self.assertEqual(tune.call_count, 4)  # only the second exclusion
            out = Path(temp)
            self.assertFalse((out/'metrics.csv').exists())
            self.assertFalse(json.loads((out/'progress.json').read_text())['complete'])
            for fold, row in enumerate(self.X.index[:2]):
                for name in models:
                    fitted, checkpoint = load_fold_pipeline(out, name, int(row))
                    self.assertNotIn(row, checkpoint['training_source_rows'])
                    self.assertEqual(len(checkpoint['training_source_rows']), 29)
                    records = checkpoint['feature_selection']
                    self.assertEqual(selected_names(selection_records(fitted, self.X.columns, name)),
                                     checkpoint['selected_features'])
                    self.assertEqual(len(records), 6)
                    k = checkpoint['validation']['selected_k']
                    self.assertIn(k, [2, 4, 6])
                    self.assertEqual(sum(r['selected'] for r in records), k)
                    self.assertAlmostEqual(sum(r['cart_importance'] for r in records), 1)
                    self.assertTrue(all(r['cart_importance'] == 0 for r in records if not r['selected']))
                    self.assertAlmostEqual(disease_probability(fitted, self.X.loc[[row]])[0],
                                           checkpoint['probability'])
                    validation = checkpoint['validation']
                    for split, features in zip(validation['inner_splits'], validation['inner_feature_selection']):
                        train, valid = set(split['training_source_rows']), set(split['validation_source_rows'])
                        self.assertTrue(train.isdisjoint(valid))
                        self.assertNotIn(row, train | valid)
                        self.assertEqual(train | valid, set(checkpoint['training_source_rows']))
                        self.assertEqual(len(features['selected_features'][name]), k)
                    trials = pd.read_csv(out/'folds'/f'fold_{fold:04d}_{name}_trials.csv')
                    for _, trial in trials.iterrows():
                        inner = json.loads(trial.user_attrs_inner_feature_selection)
                        self.assertEqual(len(inner), 3)
                        self.assertTrue(all(len(f['selected_features'][name]) == trial.params_feature_count
                                            for f in inner))
            all_features = pd.read_csv(out/'feature_selection_all_folds.csv')
            self.assertEqual(len(all_features), 2*4*6)
            frequency = pd.read_csv(out/'feature_selection_frequency.csv')
            self.assertTrue(frequency.evaluated_folds.eq(2).all())
            np.testing.assert_allclose(frequency.selection_frequency, frequency.selected_folds/2)
            counts = pd.read_csv(out/'feature_count_summary.csv')
            self.assertTrue(counts.groupby('model').folds.sum().eq(2).all())
            config = json.loads((out/'run_config.json').read_text())
            self.assertEqual(config['k_values'], [2, 4, 6])
            self.assertTrue(config['save_fold_models'])
            self.assertIn('utils.py', config['source_fingerprint'])

    def test_resume_and_xai_loader_reject_altered_pipeline(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(workflow, 'workflow_data', return_value=(self.X, self.y)), \
                patch('utils.workflow_data', return_value=(self.X, self.y)):
            kwargs = dict(models=['logistic'], trials=1, top_k=4, k_values=[2, 4],
                          max_folds=1, output_dir=temp, save_fold_models=True)
            workflow.run_optuna_lopo(**kwargs)
            path = Path(temp)/'fold_models/fold_0000_logistic.pkl'
            path.write_bytes(path.read_bytes()+b'altered')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_fold_pipeline(temp, 'logistic', 100)
            with self.assertRaisesRegex(ValueError, 'altered saved fold model'):
                workflow.run_optuna_lopo(**kwargs)

    def test_resume_rejects_changed_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(workflow, 'workflow_data', return_value=(self.X, self.y)):
            kwargs = dict(models=['logistic'], trials=1, top_k=4, k_values=[2, 4],
                          max_folds=1, output_dir=temp)
            workflow.run_optuna_lopo(**kwargs)
            path = Path(temp)/'run_config.json'
            config = json.loads(path.read_text())
            config['source_fingerprint']['utils.py'] = 'different-source'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'different configuration'):
                workflow.run_optuna_lopo(**kwargs)


if __name__ == '__main__':
    unittest.main()
