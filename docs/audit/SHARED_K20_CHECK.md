# Shared k = 20 check

Checked 22 October 2026. No model results or manuscript claims were changed.

## Findings

- The four individual classifiers reported in the paper already use k=20 in all 319 outer exclusions.
- The current Stacking run shares one k across XGBoost, LightGBM and Ridge within each trial, but selects k from 10, 20, 30 and 38. Every saved Stacking checkpoint was checked: its three base selectors agree on k. The selected-k counts are 1, 45, 34 and 239, respectively.
- Consequently, the current Stacking score (AUC 0.8762) is not a fixed-k20 result. The 45 exclusions whose winning trial used 20 features cannot replace a complete fixed-k evaluation. That outcome-dependent subset includes 11 cases and 34 controls and is not the complete comparison cohort.
- An archived k20 Stacking evaluation exists under results/optuna_stacking, but it uses the earlier four-base architecture with only meta-classifier regularisation tuned. It cannot supply the fixed-k20 score for the current XGBoost/LightGBM/Ridge architecture.

## Implementation check

Using outer fold 0's training data, and each of its three training-only validation partitions, the selectors were fitted for logistic regression, random forest, XGBoost, LightGBM and all three Stacking bases. In each partition, all seven pipelines retained exactly 20 identical source fields. This follows from their shared imputation, CART settings, random state and RFE procedure. Different training partitions can legitimately select different fields.

The check did not fit a global feature selector and then reuse its result in validation. Selection stays inside the appropriate training partition. Detailed selected fields and checks are in shared_k20_selector_checks.json.

`--k-values 20` forces the search to use exactly 20 source features. `--top-k 20` alone does not: it only initialises the template before the searched feature count is applied. The shared Stacking controls are:

```text
xgboost__select__n_features_to_select = 20
lightgbm__select__n_features_to_select = 20
ridge__select__n_features_to_select = 20
```

These are source fields before categorical one-hot encoding, not necessarily 20 final encoded columns.

## Complete controlled evaluation

No completed fixed-k20 evaluation of the current Stacking architecture was found. To generate all candidates consistently in a fresh directory:

```sh
.venv/bin/python optuna_lopo.py \
  --models logistic random_forest xgboost lightgbm stacking \
  --seed 42 --trials 20 --inner-folds 3 \
  --k-values 20 \
  --output-dir results/shared_k20
```

This is a full new experiment: 319 exclusions, 20 trials and three inner folds per candidate. It was not launched as part of this check. A stacking-only rerun could be compared with the saved individual runs with the preprocessing/version difference explicitly documented; rerunning all five under the same code avoids that provenance complication.

A shared k resolves the feature-count inequality. It does not equalise compute/search dimensionality or remove earlier cohort exploration and retrospective selection optimism.
