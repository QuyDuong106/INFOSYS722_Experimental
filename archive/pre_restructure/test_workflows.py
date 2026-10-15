"""Regression checks for evaluation coverage and seed aggregation."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_holdout
import run_loocv
from utils import DEFAULT_SEEDS, METRICS, load_data, summarize


class WorkflowTests(unittest.TestCase):
    def test_sample_variance(self):
        rows = [dict(seed=i, model="example", evaluation="test",
                     **{metric: float(i) for metric in METRICS}) for i in range(5)]
        summary = summarize(pd.DataFrame(rows))
        np.testing.assert_allclose(summary["mean"], 2.0)
        np.testing.assert_allclose(summary["variance"], 2.5)
        self.assertTrue(summary.n_seeds.eq(5).all())

    def test_both_workflows(self):
        rng = np.random.default_rng(10)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = pd.DataFrame({"measurement_a": rng.normal(size=20),
                                 "measurement_b": rng.normal(size=20),
                                 "gender": np.tile([0, 1, 1, 0], 5),
                                 "gallstone_status": np.tile([0, 1], 10)})
            path = root / "data.csv"
            data.to_csv(path, index=False)
            for strategy, runner in [("holdout", run_holdout), ("loocv", run_loocv)]:
                output = root / strategy
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.main(["--data", str(path), "--output-dir", str(output)])
                metrics = pd.read_csv(output / "metrics_per_seed.csv")
                summary = pd.read_csv(output / "summary.csv")
                self.assertTrue(summary.n_seeds.eq(5).all())
                self.assertEqual(len(metrics), 50 if strategy == "holdout" else 25)
                for seed in DEFAULT_SEEDS:
                    directory = output / f"seed_{seed}"
                    if strategy == "holdout":
                        membership = pd.read_csv(directory / "split_membership.csv")
                        self.assertEqual(membership.source_row.nunique(), 20)
                        self.assertEqual(membership.split.value_counts().to_dict(),
                                         {"train": 14, "validation": 3, "test": 3})
                    for name in metrics.model.unique():
                        predictions = pd.read_csv(directory / f"{name}_predictions.csv")
                        self.assertEqual(predictions.source_row.nunique(),
                                         6 if strategy == "holdout" else 20)
                        self.assertTrue(predictions.probability_1.between(0, 1).all())
                        for evaluation, group in predictions.groupby("evaluation"):
                            row = metrics.loc[(metrics.seed == seed) & (metrics.model == name)
                                              & (metrics.evaluation == evaluation)].iloc[0]
                            self.assertAlmostEqual(row.roc_auc, roc_auc_score(
                                group.actual, group.probability_1))

    def test_reject_duplicate_patient_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            pd.DataFrame({"record_id": [1, 1], "feature": [2, 3],
                          "gallstone_status": [0, 1]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "patient IDs"):
                load_data(path, "gallstone_status", ["record_id"])


if __name__ == "__main__":
    unittest.main()
