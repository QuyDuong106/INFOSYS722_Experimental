# Dynamic feature-count tuning for individual models

The entry point is `scripts/tune_dynamic_models.py`. It tunes logistic regression, random forest, XGBoost and LightGBM with the same source-feature count candidates as the saved Stacking experiment: **10, 20, 30 and 38**. Stacking is excluded from this entry point and its saved experiment is preserved. The underlying training engine is search version 4 in `optuna_lopo.py`; version 4 adds persistent selection evidence, evaluated fold pipelines and source fingerprinting without changing the candidate classifier parameter ranges.

## Full run

```sh
cd /Users/quyduong106/projects/INFOSYS722_Experimental
.venv/bin/python scripts/tune_dynamic_models.py \
  --models logistic random_forest xgboost lightgbm \
  --k-values 10 20 30 38 \
  --seed 42 --trials 20 --inner-folds 3 \
  --save-fold-models \
  --output-dir results/dynamic_individual_v4
```

The default command evaluates 319 exclusions for four models: 1,276 fitted outer pipelines and 25,520 tuning trials. It can take several hours. Each model independently chooses its best feature count and parameters using training-only validation. The held-out record is excluded from imputation, CART selection, transformation and parameter selection. Different exclusions can select different fields even when the winning count agrees. Count means source fields, before one-hot expansion.

Rerun the **same command** after interruption to resume completed folds. Dataset, code fingerprint, environment, selected models and run options must match. Use a fresh output directory for changed settings. Version-1 and version-3 result directories are not overwritten or migrated. Finish or resume tuning with unchanged computational source; write future explanation analyses in separate scripts.

For a single candidate, pass for example `--models xgboost`. To assess a shared fixed count instead, use `--k-values 20` and another output directory. `--top-k 20` alone only initialises the template.

## Smoke check

```sh
.venv/bin/python scripts/tune_dynamic_models.py \
  --models logistic random_forest xgboost lightgbm \
  --k-values 10 20 30 38 \
  --seed 42 --trials 2 --inner-folds 3 --max-folds 2 \
  --output-dir results/dynamic_individual_v4_smoke
```

This real-data check has been completed for the first two exclusions. All eight saved pipelines were loaded successfully and their probabilities matched the checkpoints. It is explicitly marked incomplete; no full-evaluation metrics are written. Its two-trial configuration cannot be resumed as a twenty-trial run in the same directory. The full twenty-trial run subsequently completed all 319 exclusions and now supplies the individual-model paper results. Stacking evidence is retained separately.

## Saved outputs

```text
results/dynamic_individual_v4/
├── run_config.json                     # dataset, code hashes, versions, k choices
├── progress.json                       # explicit complete/incomplete status
├── folds/
│   ├── fold_0000_logistic.json          # selected k, names, ranks, CART scores,
│   │                                   # training IDs, inner splits, best parameters
│   ├── fold_0000_logistic_trials.csv    # per-trial inner selected names in JSON
│   └── ...
├── feature_selection/
│   ├── fold_0000_logistic.csv           # all source features, selected flag,
│   │                                   # RFE rank and final selector CART score
│   └── ...
├── fold_models/
│   ├── fold_0000_logistic.pkl           # evaluated raw-input fitted pipeline
│   └── ...
├── feature_selection_all_folds.csv      # combined outer selection evidence
├── feature_selection_frequency.csv      # per-model/per-feature retention fractions
├── feature_count_summary.csv            # winning-k counts per model
├── CV_Metrics/                          # selected-trial validation metrics
├── fold_best_parameters.csv
├── predictions.csv                     # full evaluation only
├── metrics.csv                         # full evaluation only
└── summary.csv                         # full evaluation only
```

Each checkpoint includes:

- `selected_features`: retained raw field names for the fitted outer pipeline.
- `feature_selection`: selected flags, RFE ranks and mapped CART scores for every raw field.
- `validation.selected_k`: the winning feature count.
- `validation.inner_splits`: training and validation row IDs for each of the three splits, excluding the outer record.
- `validation.inner_feature_selection`: selected names in each inner training fold of the winning trial.
- `model_artifact`: relative pickle path and SHA-256 checksum when pipeline saving is enabled.

Every trial CSV additionally contains `user_attrs_inner_feature_selection`, a JSON list describing selected names in its three validation-training partitions. Use `json.loads()` to read it. Only the winning outer pipeline is pickled; the procedure does not persist every trial estimator. The per-trial records describe the selector after fitting each inner training partition.

Per-fold files are written as exclusions complete. Combined selection summaries are refreshed on normal completion of a partial or complete invocation. If a process is interrupted, completed checkpoint and per-fold files remain available; resume to rebuild the combined summaries.

## Interpretation for feature comparison

`selection_frequency` is a fraction in [0,1], with `evaluated_folds` explicitly reporting coverage. Partial-run frequencies must not be treated as full-cohort estimates. `rfe_rank=1` identifies retained features, not their ordering by importance. `cart_importance` describes the final selection tree; removed fields receive zero and retained fields may also have zero if the final tree never splits on them. This is not classifier importance or SHAP importance. Mean/SD summaries describe overlapping training fits and are not independent confidence estimates.

The feature frequency summaries are descriptive outputs. Do not use their global top fields to pre-filter the dataset before LOPO. That would introduce information across evaluation boundaries.

## Load an evaluated pipeline for later XAI

Run from the project root in the recorded project environment:

```python
from pathlib import Path
from feature_selection_evidence import load_fold_pipeline
from utils import workflow_data, disease_probability

run = Path('results/dynamic_individual_v4')
model, checkpoint = load_fold_pipeline(run, 'logistic', source_row=0)
X, y = workflow_data()

# Backgrounds come only from the training records for this exact exclusion.
training = X.loc[checkpoint['training_source_rows']]
background = training.sample(25, random_state=42)
test = X.loc[[checkpoint['source_row']]]
probability = disease_probability(model, test)[0]
print(checkpoint['selected_features'])
print(probability, checkpoint['probability'])
```

The loader checks dataset identity, the excluded-record training boundary, artifact checksum and saved probability. It accepts only trusted local pickle artifacts. Supply **all raw predictor columns** to the saved pipeline: the pipeline applies its own feature selection, transformation and encoding. Do not truncate the inputs to the selected-field list. Explanations must use this excluded-record pipeline rather than a model refitted on all records.

`--refit-final` optionally creates additional full-data inference models and their selection tables under `final_models/`. These models are separate from evaluated fold models and must not be used to explain the held-out predictions as if they excluded those records.

## Validation performed

Eleven tests pass, covering training isolation, persistence, dynamic selection for all four individual families, resume without retuning completed folds, artifact tampering and changed-code rejection. The real-data two-fold smoke run also passed saved-pipeline probability checks and produced 152 per-feature frequency records. The full run has now completed. `scripts/update_dynamic_results.py` validates its metrics and feature evidence against checkpoints before generating the updated paper tables and charts.
