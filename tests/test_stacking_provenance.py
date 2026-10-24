"""Stacking reruns retain fold evidence and reject unmatched comparisons."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from scripts import tune_stacking
from feature_selection_evidence import load_fold_pipeline
from utils import ROOT, frame_signature


class StackingProvenanceTests(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=30, n_features=6, n_informative=4,
                                  n_redundant=0, random_state=42)
        self.X = pd.DataFrame(X, index=np.arange(100, 130), columns=list('abcdef'))
        self.y = pd.Series(y, index=self.X.index)

    def reference(self, directory):
        reference = directory / 'reference'
        reference.mkdir()
        config = json.loads((ROOT / 'results/dynamic_individual_v4/run_config.json').read_text())
        config.update(signature=frame_signature(self.X, self.y), rows=len(self.X),
                      k_values=[2, 4, 6], top_k=4)
        (reference / 'run_config.json').write_text(json.dumps(config))
        (reference / 'progress.json').write_text(json.dumps(dict(complete=True, evaluated_folds=len(self.X))))
        pd.concat([pd.DataFrame(dict(model=name, source_row=self.X.index,
            actual_present=self.y.to_numpy(), probability_present=.5, predicted_present=1))
            for name in config['models']], ignore_index=True).to_csv(reference / 'predictions.csv', index=False)
        return reference

    def test_smoke_saves_validation_models_partitions_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = argparse.Namespace(reference_dir=self.reference(directory), output_dir=directory / 'stacking',
                seed=42, trials=2, k_values=[2, 4, 6], max_folds=1, dry_run=False)
            with patch('scripts.tune_stacking.workflow_data', return_value=(self.X, self.y)), \
                 patch('optuna_lopo.workflow_data', return_value=(self.X, self.y)), \
                 patch('utils.workflow_data', return_value=(self.X, self.y)):
                tune_stacking.run(args)
                args.max_folds = 2
                with patch('scripts.tune_stacking.run_optuna_lopo', wraps=tune_stacking.run_optuna_lopo) as run:
                    tune_stacking.run(args)
                    self.assertEqual(run.call_count, 1)
                model, cp = load_fold_pipeline(args.output_dir, 'stacking', 100)
                self.assertEqual(list(model.named_estimators_), ['xgboost', 'lightgbm', 'ridge'])
                self.assertNotIn(100, cp['training_source_rows'])
                self.assertEqual(len(cp['feature_selection']), 3 * 6)
                self.assertEqual(len(cp['validation']['inner_splits']), 3)
                manifest = json.loads((args.output_dir / 'experiment_manifest.json').read_text())
                self.assertFalse(manifest['budget_matches_reference'])  # explicit two-trial smoke exception
                self.assertEqual(len(manifest['search_spaces']['stacking']), 14)
                summary = json.loads((args.output_dir / 'validation/optimization_summary.json').read_text())
                self.assertFalse(summary['complete'])
                self.assertEqual(summary['completed_trials'], 4)
                self.assertEqual(summary['recorded_inner_evaluations'], 12)
                self.assertEqual(len(pd.read_csv(args.output_dir / 'CV_Metrics/stacking.csv')), 2)
                self.assertEqual(len(pd.read_csv(args.output_dir / 'validation/all_trial_inner_metrics.csv')), 12)
                self.assertFalse((args.output_dir / 'metrics.csv').exists())
                maps = json.loads((args.output_dir / 'crossfit_partitions/fold_0000.json').read_text())
                self.assertEqual(len(maps['outer_training_base_crossfits']), 3)
                for split in maps['outer_training_base_crossfits']:
                    self.assertNotIn(100, split['training_source_rows'] + split['validation_source_rows'])
                    self.assertTrue(set(split['training_source_rows']).isdisjoint(split['validation_source_rows']))
                self.assertTrue((args.output_dir / 'provenance/source/optuna_lopo.py').exists())
                self.assertEqual(len(list((args.output_dir / 'provenance/invocations').glob('*.json'))), 2)
                snapshot = args.output_dir / 'provenance/source/optuna_lopo.py'
                snapshot.write_text('altered snapshot')
                with self.assertRaisesRegex(ValueError, 'snapshot'):
                    tune_stacking.run(args)

    def test_complete_run_rejects_unmatched_budget_and_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = argparse.Namespace(reference_dir=self.reference(directory), output_dir=directory / 'stacking',
                seed=42, trials=2, k_values=[2, 4, 6], max_folds=None, dry_run=False)
            with self.assertRaisesRegex(ValueError, 'trial budget'):
                tune_stacking.check_reference(args, self.X, self.y)
            args.trials = 20
            args.seed = 7
            with self.assertRaisesRegex(ValueError, 'seed'):
                tune_stacking.check_reference(args, self.X, self.y)
            self.assertFalse(args.output_dir.exists())


if __name__ == '__main__':
    unittest.main()
