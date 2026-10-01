"""Reproduce Section 8's validation-only revision and fitted-model evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.tree import export_text
from scipy.special import expit
from threadpoolctl import threadpool_limits

from utils import DEFAULT_SEEDS, ROOT, frame_signature, make_evaluation_model, split_indices, workflow_data


def main():
    X, y = workflow_data()
    out = ROOT / "results" / "iteration_patterns"
    out.mkdir(parents=True, exist_ok=True)
    metrics, coefficients, importance, selections, groups = [], [], [], [], []
    for seed in DEFAULT_SEEDS:
        split = split_indices(X, y, seed)
        train, valid = split["train"], split["validation"]
        for k in (20, 10):
            model = make_evaluation_model("logistic", X, seed, "rfe", k)
            model.fit(X.loc[train], y.loc[train])
            prob = model.predict_proba(X.loc[valid])[:, 1]
            assert np.isfinite(prob).all() and np.isfinite(model.named_steps["model"].coef_).all()
            z = np.asarray(model[:-1].transform(X.loc[valid]))
            assert np.isfinite(z).all()
            classifier = model.named_steps["model"]
            independent = expit(np.einsum("ij,j->i", z, classifier.coef_[0]) + classifier.intercept_[0])
            np.testing.assert_allclose(prob, independent, atol=1e-12)
            metrics.append(dict(seed=seed, top_k=k, auc=roc_auc_score(y.loc[valid], prob),
                                sensitivity=recall_score(y.loc[valid], prob >= .5)))
            if k != 20:
                continue
            names = model.named_steps["impute"].get_feature_names_out()[model.named_steps["select"].get_support()]
            selections.extend(dict(seed=seed, feature=f) for f in names)
            transformed = model.named_steps["transform"].get_feature_names_out()
            coefficients.extend(dict(seed=seed, feature=f, coefficient=float(c)) for f, c in
                                zip(transformed, model.named_steps["model"].coef_[0]))
            perm = permutation_importance(model, X.loc[valid], y.loc[valid], scoring="roc_auc",
                                          n_repeats=20, random_state=seed, n_jobs=1)
            importance.extend(dict(seed=seed, feature=f, auc_drop=float(v)) for f, v in
                              zip(X.columns, perm.importances_mean))
            tree = model.named_steps["select"].estimator_
            (out / f"rfe_tree_seed_{seed}.txt").write_text(export_text(tree, feature_names=list(names), decimals=4))
            if seed == 42:
                for group in (0, 1):
                    mask = X.loc[valid, "gender"].eq(group).to_numpy()
                    positive = mask & y.loc[valid].eq(1).to_numpy()
                    groups.append(dict(gender=group, n=int(mask.sum()), positive=int(positive.sum()),
                                       missed=int((positive & (prob < .5)).sum()),
                                       mean_probability=float(prob[mask].mean())))
        prep = make_evaluation_model("logistic", X, seed, "cart", 20)
        prep.fit(X.loc[train], y.loc[train])
        tree = prep.named_steps["select"].estimator_
        names = prep.named_steps["impute"].get_feature_names_out()
        nodes = []
        for node in range(tree.tree_.node_count):
            nodes.append(dict(node=node, feature=str(names[tree.tree_.feature[node]]) if tree.tree_.feature[node] >= 0 else "leaf",
                              threshold=float(tree.tree_.threshold[node]), n=int(tree.tree_.n_node_samples[node]),
                              present_fraction=float(tree.tree_.value[node, 0, 1] / tree.tree_.value[node, 0].sum())))
        (out / f"cart_nodes_seed_{seed}.json").write_text(json.dumps(nodes, indent=2))
    for name, rows in (("cycle_metrics", metrics), ("coefficients", coefficients),
                       ("permutation", importance), ("selections", selections), ("gender_validation", groups)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    summary = pd.DataFrame(metrics).groupby("top_k")[["auc", "sensitivity"]].agg(["mean", "var"])
    summary.to_csv(out / "cycle_summary.csv")
    (out / "protocol.json").write_text(json.dumps(dict(source_signature=frame_signature(X, y),
        seeds=list(DEFAULT_SEEDS), revision="RFE top 20 to top 10; logistic C=1 unchanged",
        evaluation="validation only; no test data accessed; exploratory reused validation",
        permutation_repeats=20, variance_ddof=1), indent=2))
    print(summary.to_string())
    print(pd.DataFrame(importance).groupby("feature").auc_drop.mean().sort_values(ascending=False).to_string())
    print(pd.DataFrame(coefficients).groupby("feature").coefficient.agg(["count", "mean", "min", "max"]).to_string())


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("once", RuntimeWarning)
        main()
    warning_text = sorted({str(w.message) for w in captured})
    (ROOT / "results/iteration_patterns/runtime_warnings.json").write_text(json.dumps(warning_text, indent=2))
    print("Finite-data and independent probability checks passed; runtime warnings recorded:", len(warning_text))
