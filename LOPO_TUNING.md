# Current version 4: dynamic individual-model tuning and saved features

The current engine saves training-only feature-selection evidence and optional fitted outer pipelines. Use `scripts/tune_dynamic_models.py` for logistic regression, random forest, XGBoost and LightGBM without rerunning Stacking. It saves fold models by default. See [the full commands and output schema](docs/DYNAMIC_FEATURE_TUNING.md).

Current runs require **new result directories**, such as `results/dynamic_individual_v4`. Source fingerprints prevent mixing code versions on resume. The historical version-3 commands and descriptions below document existing evidence; they are not resume commands for version 4. Reading and explaining saved Stacking evidence remains supported.

---

# Historical version 3: joint feature-count search

Search version 3 jointly tunes RFE feature count and classifier/base-learner parameters using only the three inner training-CV folds. The default candidates for the 38-predictor dataset are **10, 20, 30 and 38**; override them with `--k-values`. In stacking, one candidate k is shared across XGBoost, LightGBM and ridge within each trial, and each base learner refits its own selector inside its training folds. The logistic meta learner remains fixed.

Continuous preparation now uses exactly `PowerTransformer(method="yeo-johnson", standardize=True)`, without a separate continuous StandardScaler. The meta learner's existing scaler is retained. The feature-count search happens before evaluation of a trial's validation scores; the outer patient is excluded from every selection decision. Best k is stored in `CV_Metrics/<model>.csv` as `selected_k`, in checkpoint pipeline parameters and in trial histories (`params_feature_count`). Counts refer to source fields before categorical expansion. The dummy has no feature count.

```bash
cd /Users/quyduong106/projects/INFOSYS722_Experimental
.venv/bin/python optuna_lopo.py --trials 20 --inner-folds 3 --seed 42 \
  --k-values 10 20 30 38 --output-dir results/lopo_v3

# Stacking only:
.venv/bin/python optuna_lopo.py --models stacking --trials 20 --inner-folds 3 \
  --seed 42 --k-values 10 20 30 38 --output-dir results/stacking_v3
```

Use a new output directory: versions 1 and 2 are incompatible with this search. The legacy `--top-k` option initializes the pipeline template; it does not override the joint search. For fixed k, pass a single candidate, e.g. `--k-values 20`. The 20-trial budget is now shared across feature counts and model settings; it is not 20 trials per k. Historical saved scores are preserved. The version-3 stacking run has now completed and is reported separately in the manuscript; see the completed evidence section below.

---

# Nested LOPO tuning: search version 3

Historical `results/optuna_*` evidence remains preserved. The manuscript now retains only completed version-3 base-tuned stacking from `results/stacking_v3`; no incomplete version-2 results are reported.

## Stacking and fair validation

New stacking has three base learners: XGBoost, LightGBM and `RidgeClassifier`. The meta learner remains the existing `StandardScaler` plus logistic regression (`C=1`, maximum 3,000 iterations), with no tuned meta parameters or raw-feature passthrough. Stacking uses three-fold stratified out-of-fold base inputs: positive-class probabilities from boosting and decision scores from ridge (`stack_method='auto'`). Ridge decision scores are not labelled as probabilities.

For each of 319 outer exclusions, the other 318 records supply **three stratified Optuna validation folds**, with shuffle seed 42 by default. All imputation, RFE20, Yeo–Johnson, scaling and encoding are refitted inside the training partitions. Stacking additionally constructs its meta-training inputs using its own three-fold splits inside each training partition. Those additional folds do not see the outer test patient or the Optuna validation fold. Each outer best model refits on all 318 records, then predicts the excluded patient.

The tuning objective remains mean inner ROC–AUC, threshold 0.5 and 20 trials per patient/model. Stacking jointly searches its base parameters with the same trial budget and validation partitions as standalone candidates; its larger joint search and extra stacking fits still have different computational costs. This is equal budget and assessment, not a guarantee of equal optimization difficulty. Model seed and inner split seed are shared; the TPE sampler uses `(seed + outer_fold) mod 2**32`, independent of model order or whether a candidate runs alone.

| Candidate/base | Optuna search |
| --- | --- |
| Logistic standalone | C: 0.01–100, log scale |
| Random forest standalone | Trees: 40–160, step 20; depth: 2–8; minimum leaf: 2–10; max features: sqrt / 0.5 / 1.0 |
| XGBoost standalone and stacking base | Trees: 40–160, step 20; learning rate: 0.01–0.15, log; depth: 1–4; row/column sampling: 0.6–1.0; lambda: 0.1–10, log |
| LightGBM standalone and stacking base | Trees: 40–160, step 20; learning rate: 0.01–0.15, log; leaves: 3–15; minimum child samples: 10–40; column sampling: 0.6–1.0; lambda: 0.1–10, log |
| Ridge stacking base | Alpha: 0.01–100, log scale |
| Logistic meta learner | Fixed C=1; not searched |

Ridge is a stacking base, not an additional standalone candidate in this change. Historical fixed pipelines, report reconstruction and copied archives remain unchanged.

## Run without reinstalling the existing local environment

From `INFOSYS722_Experimental`:

```bash
# All five predictive families plus the untuned dummy; complete LOPO.
.venv/bin/python run_workflow.py --from-stage lopo --to-stage lopo \
  --trials 20 --inner-folds 3 --seed 42 --output-dir results/lopo_v2

# Stacking alone; new folder, same split/seed rules.
.venv/bin/python optuna_lopo.py --models stacking --trials 20 \
  --inner-folds 3 --seed 42 --output-dir results/stacking_base_v2
```

In Colab or another configured environment, use `python` in place of `.venv/bin/python`. Existing version-1 checkpoints are deliberately incompatible with version 2. Resume new runs with the identical command and output directory. Do not reuse completed manuscript output directories.

## Per-patient validation CSVs

Each output directory contains **`CV_Metrics/<model>.csv`**, e.g. `results/lopo_v2/CV_Metrics/stacking.csv`. There is one row per completed patient exclusion (319 on a complete run). Each row includes:

- seed, model, zero-based outer-fold index, excluded source-row identifier and sampler seed;
- selected trial number (zero-based), inner-fold count and score scope;
- **accuracy, roc_auc, precision, recall, specificity and sensitivity**, averaged over the selected trial's three validation folds;
- sample standard deviation and each of the three individual fold values for every metric.

Recall and sensitivity are the same positive-class measure and are saved under both requested names. Classification metrics use threshold 0.5 and precision uses zero when no positive prediction exists. Trial CSVs also retain the validation metrics of every tested trial. Selected validation scores are search diagnostics and may be optimistic after selection; final test metrics still come from pooled outer predictions in `metrics.csv`.

Checkpoints retain the validation evidence. CSVs are updated atomically after each completed outer patient and regenerated on resume, without repeating completed fits. A partial run may have partial validation CSVs but never publishes complete outer metrics. Old trial histories recorded AUC only; missing historical accuracy/precision/etc. cannot be recovered without new fitting and are not fabricated.

## Hard majority voting from completed predictions

```bash
.venv/bin/python majority_voting.py \
  --runs results/optuna_logistic results/optuna_random_forest \
         results/optuna_xgboost results/optuna_lightgbm results/optuna_stacking \
  --output-dir results/majority_historical
```

This example combines the five **historical** completed models; it does not evaluate the new stacking design. After version-2 runs finish in separate per-family directories, pass those directories instead. Each input must contain a single model/seed with complete nested LOPO predictions, matching source rows, labels, cohort signature and threshold. Duplicate candidates, partial runs, mixed seeds and mismatched patients are rejected. No classifiers are trained by this script.

Each candidate casts one binary vote. More than half positive votes yields a positive decision; even-voter ties are negative by default (`--tie-break positive` changes this explicitly). ROC–AUC uses the fraction of positive votes, not the binary decision or mean probability. It is a discrete ranking score, not a calibrated probability. Outputs are `predictions.csv`, `metrics.csv` and `voting_config.json`, with input hashes and tie policy. Existing output evidence is not overwritten.

Choose the voter set and tie policy before assessment. Combining already inspected predictions is a retrospective ensemble analysis; it does not evaluate a nested voter-selection procedure. Do not optimize voter subsets or weights using the outer held-out labels.

## Completed version-3 stacking evidence

`results/stacking_v3` contains a complete 319-patient, seed-42 run with twenty trials per outer exclusion (6,380 complete trials). Pooled AUC is 0.8761695102, accuracy 0.8056426332, sensitivity 0.8385093168 and specificity 0.7721518987. Its selected-k frequencies are {10: 1, 20: 45, 30: 34, 38: 239}. The paper retains this as its only stacking configuration; historical meta-only stacking evidence remains on disk for provenance. No new standalone k-search results are claimed. Version-4 execution treats this directory as historical evidence; do not resume it with the current engine. Use the original version-3 code to reproduce that run, and a fresh directory for new settings.

## Stacking with complete provenance

The dedicated current entry point is `scripts/tune_stacking.py`. It fixes the comparison against the completed four-model dynamic experiment and always saves evaluated fold pipelines and validation/feature records. See [the rerun guide](docs/STACKING_PROVENANCE_RERUN.md). Use a new `results/stacking_v4` directory; retain `stacking_v3` until the new evaluation and explanations are verified.
