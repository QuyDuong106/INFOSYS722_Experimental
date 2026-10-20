"""Exercise real base-learner tuning, validation persistence and voting alignment."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from optuna_lopo import make_tuning_template, tune_training_fold, suggest_parameters
from cv_reporting import METRICS, write_cv_metrics
from majority_voting import combine_predictions


class LowTrial:
    def suggest_float(self, name, low, high, **kwargs):
        return low
    def suggest_int(self, name, low, high, **kwargs):
        return low
    def suggest_categorical(self, name, choices):
        return choices[0]


class StackingVotingTests(unittest.TestCase):
    def test_real_stacking_tunes_bases_and_preserves_meta(self):
        X, y = make_classification(n_samples=48, n_features=6, n_informative=4,
                                    n_redundant=0, random_state=42)
        X, y = pd.DataFrame(X, columns=list('abcdef')), pd.Series(y)
        stack = make_tuning_template('stacking', X, 42, 4)
        self.assertEqual([name for name, _ in stack.estimators], ['xgboost','lightgbm','ridge'])
        self.assertEqual(stack.final_estimator.named_steps['logistic'].C, 1)
        params = suggest_parameters(LowTrial(), 'stacking')
        self.assertTrue(all(not key.startswith('final_estimator') for key in params))
        self.assertIn('ridge__model__alpha', params)
        stack.set_params(**params)  # Check every generated nested estimator parameter.
        fitted, params, score, trials = tune_training_fold(
            'stacking', X, y, trials=2, top_k=4, seed=42, k_values=[2,4])
        self.assertEqual(fitted.final_estimator_.named_steps['logistic'].C, 1)
        selected_k = fitted.lopo_validation_['selected_k']
        self.assertIn(selected_k, [2,4])
        for base in fitted.named_estimators_.values():
            self.assertEqual(int(base.named_steps['select'].get_support().sum()), selected_k)
            numeric = base.named_steps['transform'].named_transformers_['continuous']
            self.assertEqual(list(numeric.named_steps), ['yeo_johnson'])
            self.assertTrue(numeric.named_steps['yeo_johnson'].standardize)
        self.assertEqual(len(fitted.lopo_validation_['folds']), 3)
        self.assertAlmostEqual(score, np.mean([m['roc_auc'] for m in fitted.lopo_validation_['folds']]))
        for m in fitted.lopo_validation_['folds']:
            self.assertEqual(m['recall'], m['sensitivity'])
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = dict(model='stacking', fold=0, source_row=99,
                sampler_seed=42, validation=fitted.lopo_validation_)
            write_cv_metrics(temp, [checkpoint], seed=42, inner_folds=3)
            csv = pd.read_csv(Path(temp)/'stacking.csv')
            self.assertEqual(csv.left_out_source_row.iloc[0], 99)
            for metric in METRICS:
                self.assertIn(metric, csv)
                self.assertIn('fold_3_'+metric, csv)

    def test_hard_voting_ties_and_patient_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            folders=[]
            for name, probabilities in [('a',[.9,.8,.2,.1]),('b',[.1,.8,.9,.1])]:
                folder=Path(temp)/name;folder.mkdir();folders.append(folder)
                (folder/'run_config.json').write_text(json.dumps(dict(
                    signature='same', seed=42, rows=4, threshold=.5)))
                (folder/'progress.json').write_text(json.dumps(dict(complete=True,evaluated_folds=4)))
                pd.DataFrame(dict(source_row=[10,11,12,13],actual_present=[0,1,1,0],
                    model=name,seed=42,evaluation='lopo_optuna',probability_present=probabilities,
                    predicted_present=(np.array(probabilities)>=.5).astype(int))).iloc[::-1].to_csv(folder/'predictions.csv',index=False)
            pred, metrics, config=combine_predictions(folders)
            self.assertEqual(list(pred.predicted_present),[0,1,0,0])
            self.assertAlmostEqual(metrics.accuracy.iloc[0],.75)
            positive,_,_=combine_predictions(folders,tie_break='positive')
            self.assertEqual(list(positive.predicted_present),[1,1,1,0])
            altered=pd.read_csv(folders[1]/'predictions.csv')
            altered.loc[0,'actual_present']=1
            altered.to_csv(folders[1]/'predictions.csv',index=False)
            with self.assertRaisesRegex(ValueError,'labels differ'):
                combine_predictions(folders)


if __name__ == '__main__':
    unittest.main()
