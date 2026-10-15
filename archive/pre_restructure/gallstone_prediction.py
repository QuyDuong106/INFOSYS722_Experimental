"""Gallstone binary prediction with feature selection and Optuna tuning.

Supports two evaluation strategies:
1. ``holdout``: stratified 70% train / 15% validation / 15% test.
2. ``loo``: leave-one-out predictions for every patient. Hyperparameters are
   selected only from an inner stratified CV, and the complete preprocessing +
   feature-selection pipeline is refitted for every left-out patient.

The script accepts one table, or it can merge the two supplied project tables
on record_id. It writes metrics, predictions, selected features, the fitted
holdout model, and the Optuna study table to the output directory.
"""

from __future__ import annotations

import argparse
import json
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


SEED = 42
DEFAULT_DEMOGRAPHIC = Path(__file__).with_name("SPSS_modeler") / "gallstone_demographic_clinical.csv"
DEFAULT_BIOLOGICAL = Path(__file__).with_name("SPSS_modeler") / "gallstone_biological_body_composition.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV or XLSX containing features and target")
    parser.add_argument("--demographic", type=Path, default=DEFAULT_DEMOGRAPHIC)
    parser.add_argument("--biological", type=Path, default=DEFAULT_BIOLOGICAL)
    parser.add_argument("--target", default="gallstone_status")
    parser.add_argument("--id-columns", nargs="*", default=["record_id"])
    parser.add_argument("--strategy", choices=["holdout", "loo", "both"], default="both")
    model_choices = ["logistic", "random_forest", "extra_trees", "svm", "xgboost", "lightgbm"]
    parser.add_argument("--models", nargs="+", choices=model_choices, default=model_choices)
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=None, help="Optional tuning timeout in seconds per model")
    parser.add_argument("--output-dir", type=Path, default=Path("gallstone_results"))
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.data:
        return read_table(args.data)
    left, right = read_table(args.demographic), read_table(args.biological)
    keys = [column for column in args.id_columns if column in left.columns and column in right.columns]
    if not keys:
        raise ValueError("The two input files have no shared ID column; pass --data instead")
    # Target is kept from the left table and checked before dropping its duplicate.
    if args.target in left and args.target in right and not left[args.target].equals(right[args.target]):
        raise ValueError("Target values disagree between demographic and biological files")
    return left.merge(right.drop(columns=[args.target], errors="ignore"), on=keys, validate="one_to_one")


def prepare_xy(data: pd.DataFrame, target: str, id_columns: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if target not in data:
        raise ValueError(f"Target {target!r} not found. Available columns: {list(data.columns)}")
    data = data.dropna(subset=[target]).reset_index(drop=True)
    labels = data[target].drop_duplicates().tolist()
    if len(labels) != 2:
        raise ValueError(f"Binary target required; found {labels}")
    mapping = {value: index for index, value in enumerate(sorted(labels, key=str))}
    y = data[target].map(mapping).astype(int)
    ids = data[[c for c in id_columns if c in data]].copy()
    X = data.drop(columns=[target, *[c for c in id_columns if c in data]])
    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain after removing target and IDs")
    return X, y, ids


def make_pipeline(model_name: str, params: dict[str, Any], n_raw_features: int) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    k = params.pop("select_k", "all")
    # k refers to transformed features. "all" is always safe; integer values are
    # capped by the raw width and are therefore safe for numeric-only datasets.
    selector = SelectKBest(score_func=partial(mutual_info_classif, random_state=SEED), k=k)
    if model_name == "logistic":
        model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED, **params)
    elif model_name == "random_forest":
        model = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED, **params)
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED, **params)
    elif model_name == "svm":
        model = SVC(probability=True, class_weight="balanced", random_state=SEED, **params)
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("XGBoost is required: python3 -m pip install xgboost") from error
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
            **params,
        )
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError("LightGBM is required: python3 -m pip install lightgbm") from error
        model = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
            verbosity=-1,
            **params,
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", preprocess), ("select", selector), ("model", model)])


def suggest_params(trial: optuna.Trial, model_name: str, n_features: int) -> dict[str, Any]:
    candidates = sorted(set(["all", max(1, n_features // 4), max(1, n_features // 2), max(1, 3 * n_features // 4)]), key=str)
    params: dict[str, Any] = {"select_k": trial.suggest_categorical("select_k", candidates)}
    if model_name == "logistic":
        params.update(C=trial.suggest_float("C", 1e-3, 1e3, log=True), solver="liblinear")
    elif model_name in {"random_forest", "extra_trees"}:
        params.update(
            n_estimators=trial.suggest_int("n_estimators", 150, 700, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        )
    elif model_name == "svm":
        params.update(
            C=trial.suggest_float("C", 1e-3, 1e3, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            kernel="rbf",
        )
    elif model_name == "xgboost":
        params.update(
            n_estimators=trial.suggest_int("n_estimators", 100, 700, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 15.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
    elif model_name == "lightgbm":
        params.update(
            n_estimators=trial.suggest_int("n_estimators", 100, 700, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            max_depth=trial.suggest_int("max_depth", 2, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
    return params


def tune(X: pd.DataFrame, y: pd.Series, model_name: str, args: argparse.Namespace) -> tuple[dict[str, Any], optuna.Study]:
    minority = int(y.value_counts().min())
    folds = min(args.inner_folds, minority)
    if folds < 2:
        raise ValueError("At least two examples of each class are needed for tuning")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        pipeline = make_pipeline(model_name, suggest_params(trial, model_name, X.shape[1]), X.shape[1])
        return float(cross_val_score(pipeline, X, y, scoring="roc_auc", cv=cv, n_jobs=1).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)
    return study.best_params, study


def scores(y_true: np.ndarray | pd.Series, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    type_ii_error_rate = false_negative / (false_negative + true_positive)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "false_negatives_type_ii": int(false_negative),
        "type_ii_error_rate": float(type_ii_error_rate),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "true_positives": int(true_positive),
        "confusion_matrix": matrix.tolist(),
    }


def selected_features(fitted: Pipeline) -> list[str]:
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    mask = fitted.named_steps["select"].get_support()
    return names[mask].tolist()


def save_study(study: optuna.Study, output: Path, prefix: str) -> None:
    study.trials_dataframe().to_csv(output / f"{prefix}_optuna_trials.csv", index=False)


def consolidated_results(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten all comparable evaluation metrics into one model-results table."""
    rows: list[dict[str, Any]] = []
    metric_columns = [
        "roc_auc", "average_precision", "accuracy", "balanced_accuracy",
        "precision", "recall_sensitivity", "f1", "false_negatives_type_ii",
        "type_ii_error_rate", "true_negatives", "false_positives", "true_positives",
    ]
    for model_name, model_results in summary["models"].items():
        holdout = model_results.get("holdout_70_15_15")
        if holdout:
            for evaluation_set in ("validation", "test"):
                metrics = holdout[evaluation_set]
                rows.append({
                    "model": model_name,
                    "strategy": "70_15_15",
                    "evaluation_set": evaluation_set,
                    **{column: metrics[column] for column in metric_columns},
                })
        loo = model_results.get("leave_one_out")
        if loo:
            metrics = loo["metrics"]
            rows.append({
                "model": model_name,
                "strategy": "leave_one_out",
                "evaluation_set": "all_out_of_sample_predictions",
                **{column: metrics[column] for column in metric_columns},
            })
    return pd.DataFrame(rows)


def run_holdout(X: pd.DataFrame, y: pd.Series, ids: pd.DataFrame, model_name: str,
                args: argparse.Namespace, output: Path) -> dict[str, Any]:
    # First reserve 30%, then divide it equally: exact 70/15/15 proportions up to rounding.
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, ids, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    params, study = tune(X_train, y_train, model_name, args)
    validation_model = make_pipeline(model_name, params.copy(), X.shape[1]).fit(X_train, y_train)
    val_probability = validation_model.predict_proba(X_val)[:, 1]
    validation_metrics = scores(y_val, val_probability)

    # Test remains untouched until model choice/tuning is finished. Refit using
    # train + validation, which is standard after validation has served its role.
    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])
    final_model = make_pipeline(model_name, params.copy(), X.shape[1]).fit(X_fit, y_fit)
    test_probability = final_model.predict_proba(X_test)[:, 1]
    test_metrics = scores(y_test, test_probability)
    predictions = id_test.copy()
    predictions["actual"] = y_test.to_numpy()
    predictions["probability"] = test_probability
    predictions["prediction"] = (test_probability >= 0.5).astype(int)
    predictions.to_csv(output / f"holdout_{model_name}_predictions.csv", index=False)
    joblib.dump(final_model, output / f"holdout_{model_name}_model.joblib")
    save_study(study, output, f"holdout_{model_name}")
    return {
        "best_params": params,
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "validation": validation_metrics,
        "test": test_metrics,
        "selected_features": selected_features(final_model),
    }


def run_loo(X: pd.DataFrame, y: pd.Series, ids: pd.DataFrame, model_name: str,
            args: argparse.Namespace, output: Path) -> dict[str, Any]:
    # Tune via inner CV, then generate out-of-sample predictions by refitting the
    # full pipeline in each LOO fold. The final LOO metrics never use in-sample predictions.
    params, study = tune(X, y, model_name, args)
    base = make_pipeline(model_name, params.copy(), X.shape[1])
    probability = np.empty(len(y), dtype=float)
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X, y):
        fitted = clone(base).fit(X.iloc[train_index], y.iloc[train_index])
        probability[test_index[0]] = fitted.predict_proba(X.iloc[test_index])[:, 1][0]
    predictions = ids.copy()
    predictions["actual"] = y.to_numpy()
    predictions["probability"] = probability
    predictions["prediction"] = (probability >= 0.5).astype(int)
    predictions.to_csv(output / f"loo_{model_name}_predictions.csv", index=False)
    final_model = clone(base).fit(X, y)
    joblib.dump(final_model, output / f"loo_{model_name}_model.joblib")
    save_study(study, output, f"loo_{model_name}")
    return {"best_params": params, "metrics": scores(y, probability), "selected_features": selected_features(final_model)}


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = load_data(args)
    X, y, ids = prepare_xy(data, args.target, args.id_columns)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "records": len(X), "features": X.shape[1], "class_counts": y.value_counts().sort_index().to_dict(),
        "models": {},
    }
    for model_name in args.models:
        print(f"Running {model_name} ...", flush=True)
        result: dict[str, Any] = {}
        if args.strategy in {"holdout", "both"}:
            result["holdout_70_15_15"] = run_holdout(X, y, ids, model_name, args, args.output_dir)
        if args.strategy in {"loo", "both"}:
            result["leave_one_out"] = run_loo(X, y, ids, model_name, args, args.output_dir)
        summary["models"][model_name] = result
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    consolidated_results(summary).to_csv(args.output_dir / "all_model_results.csv", index=False)
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()
