# INFOSYS722 Experimental

Gallstone classification with nested Optuna tuning under **Leave One Patient Out** evaluation, plus SHAP/LIME model understanding and error-analysis notebooks.

## Setup

```bash
git clone https://github.com/QuyDuong106/INFOSYS722_Experimental.git
cd INFOSYS722_Experimental
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name gallstone-iteration3 --display-name "Gallstone Experimental"
```

Windows PowerShell: use `.venv\Scripts\Activate.ps1` instead of `source`.

The public Gallstone source data are included under `data/raw/`. Source label 0
means gallstones present; the workflow recodes this to modeled positive class 1.
Source-row identifiers are metadata, not predictors. Leave One Patient Out
assumes one record per person; repeated patient records require grouped splits.

## Nested Optuna evaluation

```bash
# All five predictive families and the untuned dummy benchmark.
python run_workflow.py --from-stage lopo --to-stage lopo --trials 20

# One family; choose its own output directory.
python run_workflow.py --from-stage lopo --to-stage lopo --models logistic --trials 20 --output-dir results/optuna_logistic
```

Available model names: `logistic`, `random_forest`, `xgboost`, `lightgbm`,
`stacking`, `dummy_prior`. Repeat the one-family command with another name and
output directory to run it separately.

For each of 319 outer folds, exclude one record and tune using only the remaining
318. Optuna maximises mean ROC–AUC across three stratified inner folds. Imputation,
RFE–CART selection of 20 source features, transformations, and model fitting are
repeated within each inner training fold. Refit the best pipeline on the 318
records and predict the excluded record. Pool one held-out prediction per record
to calculate metrics at threshold 0.5. Stacking tunes only its logistic
meta-learner C; its base settings remain fixed. The dummy is not tuned.

```bash
# Short smoke run, not a complete evaluation.
python run_workflow.py --from-stage lopo --to-stage lopo --models logistic --trials 1 --max-folds 1 --output-dir results/optuna_smoke

# Also save separately tuned, full-data inference pipelines.
python run_workflow.py --from-stage lopo --to-stage lopo --trials 20 --refit-final
```

Completed folds are resumed when the same command is rerun. Interrupted studies
restart. Changed data, settings, search implementation or dependency versions
require a new output directory. Trial budgets apply per outer fold/model, so
full nested evaluation, especially stacking, can take substantial time.

Default outputs: `results/lopo_optuna/metrics.csv`, `predictions.csv`,
`fold_best_parameters.csv`, trial CSVs, configuration and fold checkpoints.
Partial runs do not publish complete evaluation metrics. Optional inference
models are stored in `final_models/` beneath the selected output directory and
never generate outer scores. Parameters are chosen separately for each outer
fold; they are best among the tested trials, not proven global optima.

## SHAP and LIME notebooks

```bash
python -m jupyter lab notebooks
```

Select the installed **Gallstone Experimental** kernel and run cells in order:

1. `01_model_understanding_shap_lime.ipynb`: global SHAP, individual waterfall,
   LIME explanation and local surrogate fidelity.
2. `02_error_analysis_shap_lime.ipynb`: held-out errors, calibration plots,
   descriptive threshold sensitivity, subgroup error rates, SHAP/LIME error
   cases and LIME seed stability.

These notebooks use a separate reproducible 85/15 development/test experiment,
with stacking by default. They do **not** explain the fold-specific tuned outer
models. Outputs go to `results/xai/<model>_seed_<seed>/`. Explanations describe
model behaviour, not causal effects. Low surrogate fidelity and correlated
features can weaken local interpretations.

## Validation and manuscript

```bash
python -m unittest discover -s tests -v
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The source manuscript reports the existing **fixed-parameter** Leave One Patient
Out results. Its literature review was condensed to two pages in the current
format. Historical result CSVs are included for reference; they are not Optuna
results. Update manuscript numbers only after a complete tuned run is reviewed.

Validation before upload: six tests passed, an actual one-fold Optuna smoke run
covered all five predictive families and the dummy, checkpoint reuse passed,
and a complete small synthetic nested run exercised scoring and inference
refitting. A complete 319-fold tuned study has not been run.

Other notebooks and workflow stages reproduce earlier holdout, grid-search,
feature-count and plotting experiments. Use the explicit `lopo` stage commands
above for the main nested evaluation. The historical fixed notebook can be run
with `--fixed-lopo` after its preparation exports have been generated. The
legacy error/figure scripts consume the included historical results, not the
new nested Optuna outputs.

## Main files

- `run_workflow.py`: stage runner and tuning options.
- `optuna_lopo.py`: nested tuning, evaluation, checkpoints and optional inference refits.
- `utils.py`, `modeling.py`: shared data and model pipelines.
- `xai_utils.py`: shared SHAP/LIME notebook setup.
- `notebooks/`, `tests/`, `scripts/`: analysis, invariant tests and historical comparison code.
- `main.tex` and section files: revised manuscript; references in `references.bib`.

Virtual environments, generated models, preparation caches and new run outputs
are ignored by Git. Only load trusted model pickle files.
