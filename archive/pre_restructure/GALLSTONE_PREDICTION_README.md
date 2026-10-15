# Gallstone prediction

This pipeline compares six binary classifiers: logistic regression, random
forest, extra trees, RBF SVM, XGBoost, and LightGBM. It applies
mutual-information feature selection inside the pipeline and tunes both model
and feature-selection hyperparameters with Optuna.

## Install

```bash
python3 -m pip install pandas numpy scikit-learn optuna joblib openpyxl xgboost lightgbm
```

## Run

From the project root, use the included demographic and biological files:

```bash
python3 INFOSYS722/gallstone_prediction.py --strategy both --trials 30
```

For a different CSV/XLSX (replace the target and identifier names if needed):

```bash
python3 INFOSYS722/gallstone_prediction.py \
  --data path/to/gallstone.csv \
  --target gallstone_status \
  --id-columns record_id \
  --strategy both \
  --trials 50 \
  --output-dir gallstone_results
```

Useful faster runs:

```bash
# 70/15/15 only, one model
python3 INFOSYS722/gallstone_prediction.py --strategy holdout --models logistic --trials 10

# Leave-one-out only (slow, because every model is fitted once per patient)
python3 INFOSYS722/gallstone_prediction.py --strategy loo --models logistic random_forest --trials 20

# Boosted tree models only
python3 INFOSYS722/gallstone_prediction.py --strategy both --models xgboost lightgbm --trials 30
```

The holdout workflow tunes on inner CV of the 70% training portion, reports the
15% validation result, refits on train + validation, and evaluates once on the
untouched 15% test set. The LOO workflow tunes with stratified inner CV and then
refits the entire preprocessing/selection/model pipeline for every held-out
patient. Because tuning once on all observations can mildly bias LOO estimates,
use the untouched holdout test result as the primary performance estimate.

Outputs include `all_model_results.csv`, `metrics.json`, per-patient prediction
CSVs, Optuna trial CSVs, selected feature names, and fitted `.joblib` models.
The consolidated CSV has one row per model/evaluation set and includes accuracy,
precision, sensitivity/recall, F1, false-negative count (Type II errors), Type II
error rate, ROC AUC, average precision, balanced accuracy, and confusion-matrix
counts. Type II error rate is `FN / (FN + TP)`, equivalent to `1 - recall`.
Positive-class predictions use a fixed 0.5 threshold.
