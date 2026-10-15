"""Run outer LOOCV for five seeds, pooling predictions before scoring."""

from time import perf_counter

import numpy as np
from sklearn.model_selection import LeaveOneOut

from modeling import make_model
from utils import load_data, metric_row, parse_args, prediction_rows, save_results


def main(argv=None):
    args = parse_args("loocv", argv)
    X, y = load_data(args.data, args.target, args.id_columns)
    if y.value_counts().min() < 4:
        raise ValueError("LOOCV needs at least four rows per class for inner three-fold stacking CV.")
    rows = []
    for seed in args.seeds:
        directory = args.output_dir / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        for name in args.models:
            start = perf_counter()
            probability = np.full(len(y), np.nan)
            seen = np.zeros(len(y), dtype=int)
            for fold, (train, test) in enumerate(LeaveOneOut().split(X), start=1):
                fitted = make_model(name, X, seed).fit(X.iloc[train], y.iloc[train])
                probability[test] = fitted.predict_proba(X.iloc[test])[:, 1]
                seen[test] += 1
                if fold == 1 or fold % 40 == 0 or fold == len(y):
                    print(f"LOOCV seed {seed}, {name}: {fold}/{len(y)}", flush=True)
            if not np.all(seen == 1):
                raise RuntimeError("Each row must be held out exactly once.")
            rows.append(metric_row(seed, name, "loocv", y, probability, perf_counter() - start))
            prediction_rows(seed, name, "loocv", X, y, probability).to_csv(
                directory / f"{name}_predictions.csv", index=False)
    save_results(args, "loocv", rows)


if __name__ == "__main__":
    main()
