# DESRIST Research Paper and Experimental Workflow

The canonical full research paper is [main.pdf](main.pdf), now **18 pages in Springer LNCS format** for DESRIST. Its source, build instructions, full dependency tree and consolidation details are in [DESRIST_BUILD.md](DESRIST_BUILD.md). Initial review is anonymous; Stacking is reported alongside the individual classifiers.

## Dynamic feature tuning for individual models

Use [`scripts/tune_dynamic_models.py`](scripts/tune_dynamic_models.py) to retune logistic regression, random forest, XGBoost and LightGBM with `k = 10/20/30/38`. The script excludes Stacking and saves selected fields, RFE ranks, CART scores, inner selections and fitted outer-fold pipelines for XAI. [Commands and output schema](docs/DYNAMIC_FEATURE_TUNING.md). Search version 4 uses new output directories and source fingerprints; the completed four-model dynamic evaluation now supplies the paper results, alongside retained Stacking evidence.

## Latest completed dynamic comparison

All four individual models completed 319 exclusions with dynamic feature counts. Logistic regression now leads accuracy (0.8150), sensitivity (0.8820) and F1 (0.8280); Stacking retains the highest AUC (0.8762). The top-ten feature table and charts pool five models × 319 exclusions, with Stacking counted once per exclusion. Rebuild these outputs using `.venv/bin/python scripts/update_dynamic_results.py`.

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

**Updated search version 3:** use a fresh output directory. See [LOPO_TUNING.md](LOPO_TUNING.md) for base-learner search settings, per-patient validation CSVs and majority voting. The manuscript reports standalone comparisons and retains only the completed version-3 base-tuned stacking result.

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

## Stacking provenance rerun

Use `.venv/bin/python scripts/tune_stacking.py --output-dir results/stacking_v4` to rebuild Stacking with the reference experiment's validation design and twenty-trial budget. It saves evaluated fold pipelines, validation results for every trial, feature records, split maps, source snapshots and environment provenance. See [full commands, evidence schema and interpretation](docs/STACKING_PROVENANCE_RERUN.md). The full rerun is pending; current paper results retain completed `stacking_v3` evidence and make no statistical-superiority or independently validated winner claim.

## SHAP and LIME notebooks

```bash
python -m jupyter lab notebooks
```

Select the installed **Gallstone Experimental** kernel and run cells in order:

1. `01_model_understanding_shap_lime.ipynb`: paired global SHAP importance, including logistic regression.
2. `02_error_analysis_shap_lime.ipynb`: local SHAP/LIME explanations, condition weights and surrogate fidelity for shared held-out errors and logistic coefficients.

Both inspect the paper outputs from `figures/xai/comparison`, generated by `scripts/compare_logistic_stacking_xai.py` using `results/dynamic_individual_v4` and `results/stacking_v3`. Set `REGENERATE=True` to reconstruct fold models and rerun explanations. Global SHAP uses 20 sampled held-out records; local explanations use retrospectively selected errors. They describe model behaviour, not causal effects.

## Validation and manuscript

```bash
python -m unittest discover -s tests -v
python3 scripts/build_paper.py
```

The final manuscript uses **Springer LNCS format for DESRIST** and reports the completed dynamically selected, tuned results only. All five tuned families completed 319 exclusions each, with 20 trials and three training CV folds per exclusion. Saved metrics have been independently checked against the pooled held-out probabilities. These are single-seed internal evaluations, not independent replications or external validation.

Open [main.pdf](main.pdf) for the final paper; [main.tex](main.tex) is its master source. Build the matching PDF with:

```bash
python3 scripts/build_paper.py
```

The paper XAI figures compare **fold-specific logistic regression and Stacking**. The paired script checks saved prediction agreement, SHAP additivity and local logistic coefficient reconstruction. Models agree on 292/319 classifications and share 47 errors (15 FN and 32 FP). Figure 5 compares SHAP/LIME for shared errors 156 and 178 in a compact float. LIME R² ranges from 0.296 to 0.386; some unconstrained surrogate outputs fall outside probability bounds. Full identifiers, explanations and regeneration commands are in [the shared-error report](docs/LOGISTIC_STACKING_SHARED_ERRORS.md).

The paper's completed evidence is stored in `results/dynamic_individual_v4` (four individual models) and `results/stacking_v3`. Each contains `run_config.json`, `progress.json`, `metrics.csv`, `predictions.csv`, selected parameters and per-fold trial histories. Use a **new output directory** for new experiments; the observed paper outputs are preserved. The package versions recorded in each run configuration describe the original environment; the broad requirements file alone does not guarantee identical retraining.

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

The paper compares completed dynamic-feature results for all five models; historical standalone runs are excluded. SHAP/LIME figures and notebooks compare logistic regression with Stacking. Per-patient validation metrics are in `results/stacking_v3/CV_Metrics/stacking.csv`; these differ from pooled outer-test results. Historical version-1 execution code is preserved in Git at commit `bc917c0`. The incomplete `results/lopo_v3` attempt is not submitted as completed evidence.

## Current comparison audit (24 October 2026)

The paper retains only the latest base-tuned Stacking result from `results/stacking_v4`, alongside the completed `results/dynamic_individual_v4` runs. The historical meta-tuned stacking row is excluded. All five use matching candidate fields, dynamic-k search and preprocessing; their selected subsets can differ. See [the fairness audit and optional matched-input rerun commands](docs/COMPARISON_FAIRNESS_2026-10-24.md).

## Code availability and latest execution

See [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md) for the published Optuna code version, execution entry points, completed evidence directories and commands for new matched runs. Computational code and version-4 results were published in commit `0b6b64a`; the current documentation update does not change model training.
