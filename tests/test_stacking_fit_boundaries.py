"""Trace supervised selector fits through tuning and Stacking cross-fitting."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from optuna_lopo import tune_training_fold


class StackingFitBoundaryTests(unittest.TestCase):
    def test_selector_never_sees_its_validation_or_outer_test_record(self):
        values, labels = make_classification(n_samples=72, n_features=6,
            n_informative=4, n_redundant=0, random_state=42)
        X = pd.DataFrame(values, index=np.arange(1000, 1072), columns=list('abcdef'))
        y = pd.Series(labels, index=X.index)
        excluded = X.index[0]
        train, target = X.iloc[1:], y.iloc[1:]
        calls = []
        original = RFE.fit

        def record(selector, frame, target_values, **kwargs):
            calls.append(set(frame.index))
            return original(selector, frame, target_values, **kwargs)

        with patch.object(RFE, 'fit', record):
            tune_training_fold('stacking', train, target, seed=42, trials=1,
                               top_k=4, k_values=[4])
        expected = [set(train.index[a]) for a, _ in
                    StratifiedKFold(3, shuffle=True, random_state=42).split(train, target)]
        expected.append(set(train.index))
        self.assertEqual(len(calls), 48)  # four fits, each: 3 full bases + 9 cross-fits
        for block, allowed in zip([calls[i:i+12] for i in range(0, 48, 12)], expected):
            self.assertTrue(all(rows == allowed for rows in block[:3]))
            self.assertTrue(all(rows < allowed for rows in block[3:]))
            self.assertTrue(all(excluded not in rows for rows in block))
            # For each base learner, every training record is held out once.
            for first in (3, 6, 9):
                validation_sets = [allowed - rows for rows in block[first:first+3]]
                self.assertEqual(set.union(*validation_sets), allowed)
                self.assertEqual(sum(map(len, validation_sets)), len(allowed))


if __name__ == '__main__':
    unittest.main()
