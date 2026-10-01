"""Checks for exported fold isolation, raw-input pickles and clinical labels."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (export_partition, read_transformed, fit_exported_model, load_pickle,
                   dump_pickle, disease_probability, result_row, workflow_data, load_data, ROOT)


class NotebookWorkflowTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.normal(size=(60, 23)), columns=[f"feature_{i}" for i in range(23)])
        self.X["gender"] = np.tile([0, 1], 30)
        self.y = pd.Series(np.tile([0, 1], 30), name="gallstones_present")
        self.train = self.X.index[:40]
        self.test = self.X.index[40:]

    def test_test_rows_cannot_change_training_transformation(self):
        with tempfile.TemporaryDirectory() as temporary:
            for selection in ("rfe", "cart"):
                directory = Path(temporary) / selection
                partitions = {"train": self.train, "test": self.test}
                export_partition(self.X, self.y, self.train, partitions, directory, 42, selection)
                before, _ = read_transformed(directory, "train")
                changed_X = self.X.copy()
                changed_y = self.y.copy()
                changed_X.loc[self.test, "feature_0"] = 1000
                changed_y.loc[self.test] = 1 - changed_y.loc[self.test]
                export_partition(changed_X, changed_y, self.train, partitions, directory, 42, selection)
                after, _ = read_transformed(directory, "train")
                pd.testing.assert_frame_equal(before, after)
                selected = pd.read_csv(directory / "selected_features.csv")
                self.assertEqual(selected.selected.sum(), 20)

    def test_saved_pipeline_matches_exported_matrix_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export_partition(self.X, self.y, self.train, {"train": self.train, "test": self.test}, directory, 42)
            model = fit_exported_model("logistic", 42, directory, "train", self.X, self.y)
            matrix, _ = read_transformed(directory, "test")
            expected = model.named_steps["model"].predict_proba(matrix)[:, 1]
            path = directory / "model.pkl"
            dump_pickle(model, path)
            np.testing.assert_allclose(disease_probability(load_pickle(path), self.X.loc[self.test]), expected)

    def test_disease_positive_mapping_and_fnr(self):
        X, y = workflow_data()
        _, source = load_data(ROOT / "data/raw/dataset_gallstone", "gallstone_status", ["record_id"])
        np.testing.assert_array_equal(y, source.eq(0).astype(int))
        row = result_row(42, "example", "test", [1, 1, 0, 0], np.array([0.1, 0.8, 0.7, 0.2]), 0, "rfe")
        self.assertEqual(row["FN"], 1)
        self.assertEqual(row["TP"], 1)
        self.assertEqual(row["fnr"], 0.5)


if __name__ == "__main__":
    unittest.main()
