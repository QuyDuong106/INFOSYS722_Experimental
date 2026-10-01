"""Shared, training-only setup for the two Experimental XAI notebooks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss
from utils import workflow_data, split_indices, make_evaluation_model, categorical_columns, frame_signature

ROOT = Path(__file__).resolve().parent

def setup(model_name="stacking", seed=42):
    X, y = workflow_data()
    split = split_indices(X, y, seed)
    train, test = X.loc[split["development"]], X.loc[split["test"]]
    model = make_evaluation_model(model_name, train, seed, "rfe", min(20, X.shape[1]))
    with threadpool_limits(limits=1):
        model.fit(train, y.loc[train.index])
    # Explain the entire raw-feature pipeline, including its learned selection.
    # Training-only imputation supplies finite inputs for SHAP and LIME.
    numeric = [c for c in X if c not in categorical_columns(X)]
    category = categorical_columns(X)
    medians = SimpleImputer(strategy="median", keep_empty_features=True).fit(train[numeric])
    modes = SimpleImputer(strategy="most_frequent", keep_empty_features=True).fit(train[category])
    def fill(frame):
        result = frame.copy()
        result[numeric] = medians.transform(frame[numeric])
        result[category] = modes.transform(frame[category])
        return result
    train_filled, test_filled = fill(train), fill(test)
    def predict_proba(values):
        frame = pd.DataFrame(np.asarray(values), columns=X.columns)
        with threadpool_limits(limits=1):
            return model.predict_proba(frame)[:, [list(model.classes_).index(0), list(model.classes_).index(1)]]
    probability = predict_proba(test_filled)[:, 1]
    review = test.copy()
    review["actual"] = y.loc[test.index]
    review["probability"] = probability
    review["prediction"] = (probability >= .5).astype(int)
    review["error_type"] = np.select([
        (review.actual == 1) & (review.prediction == 0),
        (review.actual == 0) & (review.prediction == 1),
        (review.actual == 1) & (review.prediction == 1)], ["FN", "FP", "TP"], default="TN")
    review["confidence"] = np.where(review.prediction == 1, probability, 1-probability)
    out = ROOT / "results" / "xai" / f"{model_name}_seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps({
        "model": model_name, "seed": seed, "signature": frame_signature(X,y),
        "training_rows": train.index.tolist(), "test_rows": test.index.tolist(),
        "threshold": .5, "positive_class": "1 = gallstones present; source code 0",
        "evaluation": "Fresh fixed 85/15 development/test split; no test-based model selection",
        "explanation": "Raw features; training-only background and imputation; probability scale"
    }, indent=2))
    return model, train_filled, test_filled, y.loc[test.index], predict_proba, review, out

def explainers(train, predict_proba, seed=42, background_size=25):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    categorical = [train.columns.get_loc(c) for c in categorical_columns(train)]
    # Dataset category codes must be contiguous integer values for LIME's mapper.
    for i in categorical:
        observed = np.unique(train.iloc[:, i])
        if not np.array_equal(observed, np.arange(int(observed.max()) + 1)):
            raise ValueError(f"Recode non-contiguous categories before LIME: {train.columns[i]}")
    background = train.sample(min(background_size, len(train)), random_state=seed)
    kernel = shap.KernelExplainer(lambda values: predict_proba(values)[:, 1], background.to_numpy(), link="identity")
    lime = LimeTabularExplainer(train.to_numpy(), feature_names=list(train.columns),
        categorical_features=categorical, class_names=["absent", "present"],
        mode="classification", discretize_continuous=True, random_state=seed)
    return kernel, lime

def shap_values(kernel, frame, nsamples=256, seed=42):
    np.random.seed(seed)
    values = np.asarray(kernel.shap_values(frame.to_numpy(), nsamples=nsamples, l1_reg=0.0))
    if values.shape != frame.shape:
        raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {frame.shape}")
    return values

def metrics(actual, probability, threshold=.5):
    tn,fp,fn,tp = confusion_matrix(actual, probability >= threshold, labels=[0,1]).ravel()
    return {"threshold":threshold, "TN":tn,"FP":fp,"FN":fn,"TP":tp,
        "recall":tp/(tp+fn) if tp+fn else np.nan,
        "specificity":tn/(tn+fp) if tn+fp else np.nan,
        "precision":tp/(tp+fp) if tp+fp else np.nan,
        "roc_auc":roc_auc_score(actual,probability),
        "average_precision":average_precision_score(actual,probability),
        "brier_score":brier_score_loss(actual,probability)}
