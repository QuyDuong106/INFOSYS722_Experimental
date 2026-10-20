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

Continuous predictors use `PowerTransformer(method="yeo-johnson", standardize=True)`. Optuna jointly tunes RFE feature count via `--k-values 10 20 30 38`; each candidate selector is fitted within training CV. Stacking shares k across its three base pipelines and keeps its meta learner fixed.

**Updated search version 3:** use a fresh output directory. See [LOPO_TUNING.md](LOPO_TUNING.md) for base-learner search settings, per-patient validation CSVs and majority voting. The manuscript retains historical search-version-1 comparisons and now separately reports the completed version-3 stacking experiment.

```bash
# All five predictive families and the untuned dummy benchmark.
python run_workflow.py --from-stage lopo --to-stage lopo --trials 20 --output-dir results/lopo_v3

# One family; choose its own output directory.
python run_workflow.py --from-stage lopo --to-stage lopo --models logistic --trials 20 --output-dir results/new_optuna_logistic_v2
```

Available model names: `logistic`, `random_forest`, `xgboost`, `lightgbm`,
`stacking`, `dummy_prior`. Repeat the one-family command with another name and
output directory to run it separately.

For each of 319 outer folds, exclude one record and tune using only the remaining
318. Optuna maximises mean ROC–AUC across three stratified inner folds. Imputation,
RFE–CART with Optuna-tuned source-feature count (default 10/20/30/38), transformations, and model fitting are
repeated within each inner training fold. Refit the best pipeline on the 318
records and predict the excluded record. Pool one held-out prediction per record
to calculate metrics at threshold 0.5. New stacking runs jointly tune XGBoost, LightGBM and ridge base learners. The existing standardized logistic meta-model stays fixed at C=1. Ridge uses decision scores; the boosting learners use probabilities. The dummy is not tuned. Three inner folds are required for every candidate.

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
`fold_best_parameters.csv`, trial CSVs, configuration and fold checkpoints. Each run also writes `CV_Metrics/<model>.csv`, containing selected-trial validation metrics for each excluded patient; these are separate from the final held-out metrics.
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

The final manuscript uses **IEEE conference format** and reports both historical fixed-parameter LOPO and completed nested Optuna results. All five tuned families completed 319 exclusions each, with 20 trials and three training CV folds per exclusion. Saved metrics have been independently checked against the pooled held-out probabilities. These are single-seed internal evaluations, not independent replications or external validation.

Open [main.pdf](main.pdf) for the final paper; [main.tex](main.tex) is its master source. Build the matching PDF with:

```bash
latexmk -pdf -outdir=ieee_build -interaction=nonstopmode -halt-on-error main.tex
cp ieee_build/main.pdf main.pdf
```

The published-paper XAI figures explain **fold-specific tuned logistic regression**, generated by `scripts/build_xai_report.py` using `results/optuna_logistic`. The two exploratory XAI notebooks described above remain separate 85/15 experiments and do not generate those paper figures.

Completed evidence is stored in `results/optuna_logistic`, `optuna_random_forest`, `optuna_xgboost`, `optuna_lightgbm` and `optuna_stacking`. Each contains `run_config.json`, `progress.json`, `metrics.csv`, `predictions.csv`, selected parameters and per-fold trial histories. Use a **new output directory** for new experiments; the observed paper outputs are preserved. The package versions recorded in each run configuration describe the original environment; the broad requirements file alone does not guarantee identical retraining.

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
- `cv_reporting.py`: selected-trial, per-patient inner-validation metrics.
- `majority_voting.py`: aligned hard voting from complete saved LOPO predictions.
- `notebooks/`, `tests/`, `scripts/`: analysis, invariant tests and historical comparison code.
- `main.tex` and section files: revised manuscript; references in `references.bib`.

Virtual environments, generated models, preparation caches and new run outputs
are ignored by Git. Only load trusted model pickle files. Completed per-family paper evidence is explicitly included in Git; new transient runs remain ignored.

## Canonical research-paper folder

`INFOSYS722_Experimental` is the single main research-paper repository. On 16 October 2026 the newer IEEE manuscript, XAI figures and observed LOPO evidence from `INFOSYS722/Research_Paper` were consolidated here. Repository-only XAI notebooks and dependencies were retained. The old Research Paper path is a local compatibility link to this folder. The separate `INFOSYS722/Experimental` environment remains available and was not reinstalled or removed.

See [the consolidation manifest](docs/consolidation_manifest.json) for source hashes and copied paths. Historical revisions are under `archive/` and `report_archive/`; they are not alternative final manuscripts. Temporary LaTeX builds and caches are excluded.

## Latest completed stacking result

The version-3 run in [`results/stacking_v3`](results/stacking_v3) completed all 319 outer exclusions, 20 trials and three inner folds per exclusion (seed 42). Verified pooled metrics are AUC **0.8762**, accuracy **0.8056**, precision **0.7895**, sensitivity **0.8385**, specificity **0.7722** and F1 **0.8133**; confusion counts are TN=122, FP=36, FN=26, TP=135. It tunes XGBoost/LightGBM/ridge bases and RFE k with the logistic meta learner fixed. Selected k counts: 10 in 1 fold, 20 in 45, 30 in 34 and 38 in 239.

The updated paper reports this as a separate extension because the historical standalone models did not search k. Historical XAI figures still explain tuned logistic regression. Per-patient validation metrics are in `results/stacking_v3/CV_Metrics/stacking.csv`; these differ from pooled outer-test results. Historical version-1 execution code is preserved in Git at commit `bc917c0`. The incomplete `results/lopo_v3` attempt is not submitted as completed evidence.
