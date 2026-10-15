"""Run stratified 70/15/15 holdout comparisons for five seeds."""

from time import perf_counter

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from modeling import make_model
from utils import load_data, metric_row, parse_args, prediction_rows, save_results


def main(argv=None):
    args = parse_args("holdout", argv)
    X, y = load_data(args.data, args.target, args.id_columns)
    rows = []
    for seed in args.seeds:
        train, reserve = train_test_split(X.index, test_size=0.30, stratify=y, random_state=seed)
        valid, test = train_test_split(reserve, test_size=0.50, stratify=y.loc[reserve], random_state=seed)
        if min(y.loc[train].value_counts().reindex([0, 1], fill_value=0)) < 3:
            raise ValueError("Training split needs at least three rows per class for stacking CV.")
        directory = args.output_dir / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.DataFrame({"source_row": indices, "split": name})
                   for name, indices in [("train", train), ("validation", valid), ("test", test)]]
                  ).sort_values("source_row").to_csv(directory / "split_membership.csv", index=False)
        development = train.append(valid)
        for name in args.models:
            predictions = []
            for evaluation, fit_indices, eval_indices in [("validation", train, valid),
                                                          ("test", development, test)]:
                start = perf_counter()
                fitted = make_model(name, X, seed).fit(X.loc[fit_indices], y.loc[fit_indices])
                probability = fitted.predict_proba(X.loc[eval_indices])[:, 1]
                rows.append(metric_row(seed, name, evaluation, y.loc[eval_indices], probability,
                                       perf_counter() - start))
                predictions.append(prediction_rows(seed, name, evaluation, X.loc[eval_indices],
                                                   y.loc[eval_indices], probability))
            pd.concat(predictions).to_csv(directory / f"{name}_predictions.csv", index=False)
            joblib.dump(fitted, directory / f"{name}.joblib")
            print(f"Holdout seed {seed}: {name} complete", flush=True)
    save_results(args, "holdout", rows)


if __name__ == "__main__":
    main()
