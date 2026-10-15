"""Shared input validation, experiment arguments, metrics and CSV reporting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from modeling import MODEL_NAMES

ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (42, 7, 21, 84, 123)
METRICS = ("roc_auc", "average_precision", "accuracy", "balanced_accuracy", "precision",
           "recall", "specificity", "f1", "log_loss", "TN", "FP", "FN", "TP")


def parse_args(strategy, argv=None):
    parser = argparse.ArgumentParser(description=f"Five-seed {strategy} model comparison")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset_gallstone")
    parser.add_argument("--target", default="gallstone_status")
    parser.add_argument("--id-columns", nargs="*", default=["record_id"])
    parser.add_argument("--seeds", nargs=5, type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "modeling_results" / strategy)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != 5 or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Provide five distinct seeds between 0 and 2**32 - 1.")
    args.models = list(dict.fromkeys(args.models))
    return args


def load_data(path, target, id_columns):
    path = Path(path)
    data = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    exported_indices = []
    for c in data:
        if str(c).startswith("Unnamed:") or not str(c).strip():
            values = pd.to_numeric(data[c], errors="coerce").to_numpy()
            if any(np.array_equal(values, np.arange(start, len(data) + start)) for start in (0, 1)):
                exported_indices.append(c)
    data = data.drop(columns=exported_indices)
    if target not in data or data[target].isna().any() or set(data[target].unique()) != {0, 1}:
        raise ValueError("Target must exist, contain both 0 and 1, and have no missing values.")
    if target in id_columns:
        raise ValueError("Target cannot also be an ID column.")
    ids = [c for c in id_columns if c in data]
    if ids and (data[ids].isna().any().any() or data.duplicated(ids).any()):
        raise ValueError("Missing or repeated patient IDs require review or grouped evaluation.")
    if data.drop(columns=ids).duplicated().any():
        raise ValueError("Review duplicate records before row-level evaluation.")
    X = data.drop(columns=[target, *ids]).replace([np.inf, -np.inf], np.nan)
    if X.shape[1] == 0:
        raise ValueError("No predictors remain.")
    return X, data[target].astype(int)


def metric_row(seed, model, evaluation, target, probability, seconds):
    probability = np.asarray(probability)
    if probability.shape != (len(target),) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Every evaluated row must have one finite probability in [0, 1].")
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return dict(seed=seed, model=model, evaluation=evaluation, rows=len(target),
                roc_auc=roc_auc_score(target, probability),
                average_precision=average_precision_score(target, probability),
                accuracy=accuracy_score(target, prediction),
                balanced_accuracy=balanced_accuracy_score(target, prediction),
                precision=precision_score(target, prediction, zero_division=0),
                recall=recall_score(target, prediction, zero_division=0),
                specificity=tn / (tn + fp), f1=f1_score(target, prediction, zero_division=0),
                log_loss=log_loss(target, probability, labels=[0, 1]),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), fit_predict_seconds=seconds)


def prediction_rows(seed, model, evaluation, X, y, probability):
    actual = np.asarray(y)
    predicted = (probability >= 0.5).astype(int)
    return pd.DataFrame(dict(seed=seed, model=model, evaluation=evaluation, source_row=X.index,
                             actual=actual, probability_1=probability, prediction=predicted,
                             error_type=np.select([(actual == 0) & (predicted == 1),
                                                   (actual == 1) & (predicted == 0)],
                                                  ["FP", "FN"], default="Correct")))


def summarize(metrics):
    long = metrics.melt(id_vars=["seed", "model", "evaluation"], value_vars=METRICS,
                        var_name="metric", value_name="value")
    return long.groupby(["model", "evaluation", "metric"], sort=False)["value"].agg(
        n_seeds="count", mean="mean", variance="var", std="std").reset_index()


def save_results(args, strategy, rows):
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics_per_seed.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metadata = dict(strategy=strategy, data=str(args.data.resolve()), target=args.target,
                    id_columns=args.id_columns, seeds=args.seeds, models=args.models,
                    threshold=0.5, variance_ddof=1,
                    seed_scope="split, estimator and inner stacking CV randomness")
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved results to {args.output_dir.resolve()}", flush=True)


def continuous_columns(frame):
    """Schema-based columns, including after selection changes the frame width."""
    from modeling import CODED_FIELDS
    return [c for c in frame if c not in CODED_FIELDS and pd.api.types.is_numeric_dtype(frame[c])]


def categorical_columns(frame):
    return [c for c in frame if c not in continuous_columns(frame)]


def make_evaluation_model(name, X, seed, selection="rfe", top_k=20):
    """Select raw predictors with CART, then transform within every training fold.

    The supplied dataset stores all categories as numeric codes. CART selection
    uses these codes; the final classifier receives one-hot encoded categories.
    Twenty means source predictors, not twenty expanded dummy columns.
    """
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import RFE, SelectFromModel
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from modeling import make_model

    if selection not in {"rfe", "cart"}:
        raise ValueError("Selection must be 'rfe' or 'cart'.")
    if not 1 <= top_k <= X.shape[1]:
        raise ValueError("top_k must be between 1 and the number of source predictors.")
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
        raise ValueError("CART selection expects the numeric category codes in the supplied dataset.")
    if name == "stacking":
        stack = make_model(name, X, seed)
        stack.set_params(estimators=[
            (base, make_evaluation_model(base, X, seed, selection, top_k))
            for base in MODEL_NAMES[:-1]
        ])
        return stack

    impute = ColumnTransformer([
        ("continuous", SimpleImputer(strategy="median", keep_empty_features=True), continuous_columns),
        ("categorical", SimpleImputer(strategy="most_frequent", keep_empty_features=True), categorical_columns),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=seed)
    selector = (RFE(tree, n_features_to_select=top_k, step=1) if selection == "rfe"
                else SelectFromModel(tree, threshold=-np.inf, max_features=top_k))
    selector.set_output(transform="pandas")
    transform = ColumnTransformer([
        ("continuous", Pipeline([
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("standardize", StandardScaler()),
        ]), continuous_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ])
    estimator = clone(make_model(name, X, seed).named_steps["model"])
    return Pipeline([("impute", impute), ("select", selector),
                     ("transform", transform), ("model", estimator)])


def selected_feature_table(fitted):
    """Return selected source fields, separately for each stacking base learner."""
    pipelines = ([(name, fitted.named_estimators_[name]) for name, _ in fitted.estimators]
                 if hasattr(fitted, "named_estimators_") else [("model", fitted)])
    rows = []
    for name, pipeline in pipelines:
        selector = pipeline.named_steps["select"]
        names = pipeline.named_steps["impute"].get_feature_names_out()
        ranks = getattr(selector, "ranking_", np.where(selector.get_support(), 1, 2))
        for feature, selected, rank in zip(names, selector.get_support(), ranks):
            rows.append(dict(base_model=name, feature=feature, selected=bool(selected), rank=int(rank)))
    return pd.DataFrame(rows)
