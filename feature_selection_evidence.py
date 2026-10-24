"""Training-only feature selection evidence and verified held-out model loading."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

SELECTION_COLUMNS = ["base_model", "feature", "selected", "rfe_rank", "cart_importance"]


def selection_records(fitted, input_features, model_name):
    """Describe the final RFE tree; excluded features receive zero final-tree score.

    Retained RFE ranks are all 1; CART scores are not classifier or SHAP scores.
    For Stacking these are full-training base selectors, not internal cross-fits.
    """
    if isinstance(fitted, DummyClassifier):
        return []
    pipelines = ([(name, fitted.named_estimators_[name]) for name, _ in fitted.estimators]
                 if hasattr(fitted, "named_estimators_") else [(model_name, fitted)])
    records = []
    for base_name, pipeline in pipelines:
        selector = pipeline.named_steps["select"]
        names = (pipeline.named_steps["impute"].get_feature_names_out()
                 if "impute" in pipeline.named_steps else np.asarray(input_features))
        selected = selector.get_support()
        if len(names) != len(selected):
            raise ValueError("Selector columns cannot be mapped to raw features")
        scores = np.zeros(len(names), dtype=float)
        scores[selected] = selector.estimator_.feature_importances_
        ranks = getattr(selector, "ranking_", np.where(selected, 1, 2))
        for feature, keep, rank, score in zip(names, selected, ranks, scores):
            records.append(dict(base_model=base_name, feature=str(feature),
                                selected=bool(keep), rfe_rank=int(rank),
                                cart_importance=float(score)))
    return records


def selected_names(records):
    groups = {}
    for row in records:
        groups.setdefault(row["base_model"], [])
        if row["selected"]:
            groups[row["base_model"]].append(row["feature"])
    return groups


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root):
    """Bind resumes to computational source, including uncommitted edits."""
    names = ("optuna_lopo.py", "utils.py", "modeling.py", "cv_reporting.py",
             "feature_selection_evidence.py")
    return {name: file_sha256(Path(root) / name) for name in names}


def write_fold_selection(out, checkpoint):
    folder = Path(out) / "feature_selection"
    folder.mkdir(exist_ok=True)
    records = checkpoint["feature_selection"]
    table = pd.DataFrame(records, columns=SELECTION_COLUMNS)
    for name, value in reversed(list(dict(model=checkpoint["model"], fold=checkpoint["fold"],
                                        source_row=checkpoint["source_row"]).items())):
        table.insert(0, name, value)
    table.to_csv(folder / f'fold_{checkpoint["fold"]:04d}_{checkpoint["model"]}.csv', index=False)


def write_selection_summaries(out, checkpoints):
    """Aggregate completed exclusions only; partial coverage is explicit."""
    out = Path(out)
    rows, counts = [], []
    for checkpoint in checkpoints:
        identity = dict(model=checkpoint["model"], fold=checkpoint["fold"],
                        source_row=checkpoint["source_row"])
        rows.extend({**identity, **row} for row in checkpoint["feature_selection"])
        k = checkpoint["validation"].get("selected_k")
        if k is not None:
            counts.append({**identity, "selected_k": k})
    table = pd.DataFrame(rows, columns=["model", "fold", "source_row", *SELECTION_COLUMNS])
    table.to_csv(out / "feature_selection_all_folds.csv", index=False)
    columns = ["model", "base_model", "feature", "evaluated_folds", "selected_folds",
               "selection_frequency", "mean_cart_importance", "sd_cart_importance"]
    if len(table):
        summary = table.groupby(["model", "base_model", "feature"], as_index=False).agg(
            evaluated_folds=("fold", "nunique"), selected_folds=("selected", "sum"),
            mean_cart_importance=("cart_importance", "mean"),
            sd_cart_importance=("cart_importance", "std"))
        summary["selection_frequency"] = summary.selected_folds / summary.evaluated_folds
        summary = summary[columns].sort_values(
            ["model", "base_model", "selection_frequency", "mean_cart_importance"],
            ascending=[True, True, False, False])
    else:
        summary = pd.DataFrame(columns=columns)
    summary.to_csv(out / "feature_selection_frequency.csv", index=False)
    if counts:
        counts = pd.DataFrame(counts).groupby(["model", "selected_k"], as_index=False).agg(
            folds=("fold", "nunique"))
        counts["evaluated_folds"] = counts.groupby("model").folds.transform("sum")
        counts["fraction"] = counts.folds / counts.evaluated_folds
    else:
        counts = pd.DataFrame(columns=["model", "selected_k", "folds", "evaluated_folds", "fraction"])
    counts.to_csv(out / "feature_count_summary.csv", index=False)


def load_fold_pipeline(run_dir, model_name, source_row):
    """Load a trusted local held-out pipeline, checking cohort, file and prediction.

    Returns (fitted_pipeline, checkpoint). Background rows for explanations must
    come from checkpoint['training_source_rows'], not the entire dataset.
    Loading pickles executes Python: use only your own trusted run directories.
    """
    from utils import workflow_data, frame_signature, load_pickle, disease_probability
    out = Path(run_dir)
    config = json.loads((out / "run_config.json").read_text())
    X, y = workflow_data()
    if config["signature"] != frame_signature(X, y):
        raise ValueError("Current raw dataset does not match this saved run")
    fold = X.index.get_loc(source_row)
    checkpoint = json.loads((out / "folds" / f"fold_{fold:04d}_{model_name}.json").read_text())
    if checkpoint["source_row"] != source_row or checkpoint["model"] != model_name:
        raise ValueError("Checkpoint identity mismatch")
    expected_training = X.index[X.index != source_row].tolist()
    if checkpoint["training_source_rows"] != expected_training:
        raise ValueError("Checkpoint training rows do not match the held-out record")
    artifact = checkpoint.get("model_artifact")
    if not artifact:
        raise ValueError("This run has no persisted fold model; use --save-fold-models")
    path = (out / artifact["path"]).resolve()
    if not path.is_relative_to(out.resolve()) or file_sha256(path) != artifact["sha256"]:
        raise ValueError("Saved fold pipeline path or checksum mismatch")
    fitted = load_pickle(path)
    np.testing.assert_allclose(disease_probability(fitted, X.loc[[source_row]])[0],
                               checkpoint["probability"], atol=1e-8, rtol=0)
    return fitted, checkpoint
