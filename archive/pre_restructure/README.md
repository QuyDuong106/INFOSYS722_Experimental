# Iteration 3: OpenStack experiments

## Notebook evaluation

Open `evaluation.ipynb` and run all cells from this directory. It uses the
existing `utils.py` and model settings with five stratified 70/15/15 splits.
RFE with CART and direct CART importance selection each retain 20 source
predictors. Continuous features then receive Yeo-Johnson and standardization;
categorical fields receive one-hot encoding. All fitted steps remain inside
the training pipelines, including stacking's inner folds.

Validation chooses the model and selector for each seed. The winner is refitted
on 85% and evaluated once on the held-out test set. The notebook reports
accuracy, ROC-AUC curves, false-negative rate, mean and sample variance, confusion
matrices, and descriptive error analysis. Models and reports are saved under
`output/notebook_evaluation/`.

**Label convention:** the dataset uses 0 = gallstones present and 1 = absent.
This notebook explicitly recodes the modeling target to 1 = present, so FNR
counts missed gallstone cases. The original command-line runners retain their
original code-1 positive-class convention; their sensitivity/error labels are
therefore not directly comparable to the notebook's disease-positive results.

Notebook execution additionally requires `jupyter`, `matplotlib` and `seaborn`.
The serialized pipelines use helpers in local `utils.py`; keep that directory
on the Python import path when loading them.

## Files

- `gallstone_eda.ipynb`: retained EDA notebook.
- `feature_engineering.ipynb`: retained feature-engineering notebook.
- `modeling.py`: logistic regression, random forest, XGBoost, LightGBM and stacking.
- `utils.py`: data validation, metrics, five-seed arguments and summary exports.
- `run_holdout.py`: stratified 70/15/15 evaluation, refitting on 85% before test.
- `run_loocv.py`: outer leave-one-out evaluation with inner three-fold stacking.

The original `stacking.ipynb`, `error_analysis.ipynb`, `gallstone_prediction.py`
and their result directories remain as historical references. Use the two new
entry points for repeated experiments; the older prediction README describes
the separate Optuna workflow.

## Run

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run_holdout.py
python3 run_loocv.py
```

Defaults resolve relative to the scripts, so absolute script paths work from
any directory. Both run five distinct seeds: 42, 7, 21, 84, 123.

```bash
python3 run_holdout.py --seeds 42 7 21 84 123 --models logistic random_forest
python3 run_loocv.py --data dataset_gallstone.xlsx --output-dir modeling_results/loocv_custom
```

The fixed settings are taken from the stacking/error-analysis notebooks.
Imputation, encoding, scaling and 75% ANOVA feature selection are fitted inside
each training fold. The positive class is code 1 and the threshold is 0.5.
Holdout seeds control both data splits and model/inner-CV randomness. LOOCV has
no random outer split; its seeds control estimators and inner stacking folds.
Deterministic models can have zero LOOCV variance. Metrics are computed on the
complete pooled out-of-fold predictions, not on individual one-row folds.

## Results

Each method writes to its own `modeling_results/<method>/` directory:

- `metrics_per_seed.csv`: one row per seed, model and evaluation set.
- `summary.csv`: mean, sample variance (`ddof=1`), sample standard deviation and
  seed count for every metric, grouped by model and evaluation set.
- `run_config.json`: input and experiment settings.
- `seed_<seed>/<model>_predictions.csv`: source row, target, probability,
  prediction and FP/FN classification.
- Holdout additionally saves split membership and fitted development models.

Metrics include ROC-AUC, average precision, accuracy, balanced accuracy,
precision, recall, specificity, F1, log loss and TN/FP/FN/TP counts.
Validation and test results are summarized separately. Repeated splits overlap;
the variance measures variability across runs, not independent uncertainty.
The dataset was previously explored, so these are internal evaluations.
LOOCV runs 319 outer fits per model per seed on the supplied dataset, with
additional internal fits for stacking, and can take substantially longer.
Use a different `--output-dir` to retain multiple experiment configurations.

## Verification

`python3 -m unittest test_workflows -q` runs both entry points with all five
models and five seeds on a small synthetic dataset, checks prediction coverage
and pooled ROC-AUC, tests sample variance, and rejects repeated patient IDs.
