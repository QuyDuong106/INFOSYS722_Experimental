> Latest: search version 3 uses `PowerTransformer(method="yeo-johnson", standardize=True)` and jointly searches RFE k using `--k-values`. See LOPO_TUNING.md for the current command; older sections below document historical execution.

> Current execution uses search version 2. Stacking now tunes XGBoost, LightGBM and ridge base learners with a fixed logistic meta learner and three inner folds. Use a fresh output directory. See [LOPO_TUNING.md](LOPO_TUNING.md). The manuscript's completed version-1 results and historical workflow descriptions below are preserved as provenance.

# Final research-paper workflow

The canonical repository is `INFOSYS722_Experimental`. The final DESRIST paper is `main.tex` / `main.pdf`. Completed tuned results are in the five `results/optuna_<family>/` directories, rather than the default directory used for a new combined run. Run new experiments in a fresh output directory. The manuscript XAI figures use `scripts/build_xai_report.py` and the Stacking outer-fold models from `results/stacking_v3`. The two XAI notebooks inspect these same artifacts. The commands below document both current execution and historical stages.

## Leave One Patient Out with nested Optuna tuning

The `lopo` stage now runs `optuna_lopo.py`. For each excluded record, Optuna
chooses classifier parameters using mean ROC–AUC across three stratified inner
folds of the other 318 records. Imputation, RFE–CART selection of 20 source
features, transformation, and classification are refitted in every inner
training fold. The best pipeline is then refitted on the 318 records and
predicts the excluded record. The outer label never informs tuning.

```bash
# Main evaluation: all five predictive families plus the untuned prior dummy.
python run_workflow.py --from-stage lopo --to-stage lopo --trials 20 --inner-folds 3

# A lower-cost run for one model family.
python run_workflow.py --from-stage lopo --to-stage lopo --models logistic --trials 20

# Smoke check: one outer fold and one trial, saved in a separate directory.
python run_workflow.py --from-stage lopo --to-stage lopo --models logistic --trials 1 --max-folds 1 --output-dir results/optuna_smoke

# Also perform a separate full-data search and save inference pipelines.
python run_workflow.py --from-stage lopo --to-stage lopo --trials 20 --refit-final

# Historical fixed-parameter notebook and original output paths.
python run_workflow.py --from-stage lopo --to-stage lopo --fixed-lopo
```

Install the updated `requirements.txt` before running. The default search budget
is 20 trials per model per outer fold; complete nested evaluation, especially
stacking, can be expensive. The dummy is not tuned. Logistic regression searches
C; random forest searches tree count, depth, leaf size and feature sampling;
XGBoost/LightGBM search bounded tree, learning-rate and regularisation settings.
Stacking searches only the logistic meta-learner C, with its base learners fixed;
its own internal training folds still refit their full pipelines. The feature
count and threshold remain fixed during a run.

New results are saved under `results/lopo_optuna/`: pooled metrics and predictions,
per-fold best parameters, complete trial CSVs and atomic fold checkpoints.
Completed folds are reused when rerunning the same command. Interrupted studies
restart, rather than reuse partially completed trials. Data, options, search
version and package versions must match; changed configurations require a new
`--output-dir`. Reusing checkpoints is not supported after changing implementation
logic without incrementing SEARCH_VERSION or choosing a new output directory.
A smoke run writes partial evidence only; remove `--max-folds` with the same
settings to complete it. It does not publish full-evaluation metrics.

`--refit-final` saves separate full-data inference models under
`results/lopo_optuna/final_models/`; those models never generate outer scores.
Existing fixed results under `results/lopo/` remain historical evidence. The five completed per-family tuned runs have now been verified and are reported in the manuscript.
Legacy `holdout`, `tuning`, `errors`, `patterns` and `figures` stages still serve
the earlier experiments; they do not consume the new nested outputs. Use the
explicit `--from-stage lopo --to-stage lopo` command for the current evaluation.
The fixed notebook remains available with `--fixed-lopo`.

Each model family is tuned separately. Picking the strongest family from outer
results still does not evaluate nested family selection. The method assumes one
record per patient; repeated-patient observations require grouped outer and inner
splits. Optuna selects the best parameters among the tested trials, not a proven
global optimum.

---

Historical workflow documentation follows. The current `lopo` stage is described above.

> Relocated workflow documentation. The research manuscript is now `main.tex`; original report sections are in `report_archive/`.

# Iteration 3: reproducible gallstone modeling workflow

This submission uses Python/Jupyter to prepare the data, fit and evaluate models,
compare holdout with leave-one-out (LOPO), tune parameters, analyse errors, and
complete the 20-versus-10-predictor feedback cycle. The final report is
[`main.pdf`](main.pdf).


## Project layout

```text
Research_Paper/
├── README.md, requirements.txt
├── run_workflow.py           # Ordered execution and stage selection
├── utils.py, modeling.py     # Shared modules; stable imports for saved pickles
├── notebooks/               # Six analytical notebooks
├── data/
│   ├── raw/                 # Original CSV and spreadsheet
│   └── processed/           # Generated training-fitted partitions and plots
├── models/                  # Saved pipelines: holdout/, lopo/, tuned/
├── results/                 # Metrics, predictions, analysis and run environments
├── scripts/                 # Iteration comparison analysis
├── tests/                   # Workflow invariant checks
├──                   # LaTeX source, figures, tables and supplied PDFs
└── archive/                 # Historical work and preserved local build/cache files
```

`archive/layout_migration.json` records the old-to-new locations. Historical
archive files and stored notebook outputs may still show their original paths;
active code uses the new layout. Existing results and models were retained.
The supplied PDFs are snapshots; rebuild them after changing report sources.

## 1. Setup and required inputs

Run all commands below from the `Research_Paper` directory, which contains
`utils.py` and `run_workflow.py`. For example, from the projects directory:

```bash
cd INFOSYS722/Research_Paper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name gallstone-iteration3 --display-name "Gallstone Iteration 3"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of
`source .venv/bin/activate`; the Python commands are the same. The batch loop
below is for macOS/Linux shells.

Required input files:

- `data/raw/dataset_gallstone`: the original CSV **without a file extension**, used by
  `utils.workflow_data()` and all active modeling workflows. Keep its name and
  row order unchanged.
- `data/raw/dataset_gallstone.xlsx`: the accompanying spreadsheet and an alternative
  input recognised by the EDA notebook. It does not replace the extensionless
  CSV in the active modeling functions.
- `requirements.txt`, `utils.py` and `modeling.py`: dependencies and shared code.

The source data contain 319 records. The notebooks retain Python 3.13.5 kernel
metadata, but select the newly installed **Gallstone Iteration 3** kernel rather
than relying on the original author's local environment. `requirements.txt`
contains minimum versions, not a locked record of the original environment.
Record the environment used for your reproduction:

```bash
python --version
python -m pip freeze > reproduction-environment.txt
```

Fixed seeds make the experiment repeatable within an environment; library,
platform and numerical changes can affect exact scores and saved pickles.
The environment file above records the current run, not the historical one.

## 2. Files to run, in order

| Order | Executable file | Purpose and principal outputs |
| --- | --- | --- |
| 1 | [`notebooks/gallstone_eda.ipynb`](notebooks/gallstone_eda.ipynb) | Original data exploration and notebook plots. Include for the complete analytical workflow; later modeling does not depend on its in-memory state. |
| 2 | [`notebooks/transformation_feature_engineering.ipynb`](notebooks/transformation_feature_engineering.ipynb) | Exports training-fitted RFE–CART/CART preparation, split membership, transformations and plots to `data/processed/`. Run before holdout and tuning. |
| 3 | [`notebooks/holdout_method.ipynb`](notebooks/holdout_method.ipynb) | Five stratified holdout splits; writes `results/holdout/`, validation winners, selected test metrics and `models/holdout/`. |
| 4 | [`notebooks/leave_one_out_method.ipynb`](notebooks/leave_one_out_method.ipynb) | 319 leave-one-row-out folds per fixed model; writes `results/lopo/`, fold exports and separate full-data models in `models/lopo/`. |
| 5 | [`notebooks/tuning_model.ipynb`](notebooks/tuning_model.ipynb) | Training-only parameter search, validation selection and selected-model test evaluation; writes `results/tuning/` and `models/tuned/`. |
| 6 | [`notebooks/error_analysis.ipynb`](notebooks/error_analysis.ipynb) | Reads saved predictions and writes error-review tables and plots to `results/error_analysis/`. Run after all three evaluation notebooks for the complete comparison. |
| 7 | [`scripts/iteration_pattern_analysis.py`](scripts/iteration_pattern_analysis.py) | Refits the 20- and 10-feature logistic references on the same five training splits; compares validation results and extracts coefficients, permutation importance and CART rules into `results/iteration_patterns/`. |
| 8 | [`build_modeling_figures.py`](build_modeling_figures.py) | Reads saved results and regenerates five modeling figures plus four evidence/comparison tables in `figures/modeling/` and `generated/`. Does not train models. |
| Check | [`tests/test_notebook_workflow.py`](tests/test_notebook_workflow.py) | Checks preparation isolation, saved-pipeline prediction consistency and disease-positive label/FNR coding. Run using `unittest` as shown below. |

Supporting Python modules must be submitted, but are **imported**, not run as
standalone experiments:

- [`utils.py`](utils.py): data loading, split/export logic, model preparation,
  fitting, evaluation, persistence and error analysis.
- [`modeling.py`](modeling.py): model definitions and fixed settings used by the
  workflow helpers.

These are all active notebooks and Python files for this submission.
`archive/pre_restructure/` contains historical implementations, not additional
required execution steps.

## 3. Run interactively or from the command line

### Interactive notebook execution

```bash
python -m jupyter lab
```

Select the **Gallstone Iteration 3** kernel. Open each notebook in orders 1–6
above and use **Restart Kernel and Run All Cells**. Complete one notebook before
starting the next. Then run the two scripts and checks from a terminal in the
project directory:

```bash
python scripts/iteration_pattern_analysis.py
python build_modeling_figures.py
python -m unittest discover -s tests -v
```

### Automated execution of the complete workflow

The cross-platform runner executes stages sequentially and stops on the first
failure. It overwrites stored notebook outputs, result CSVs, preparation exports,
and models. Copy the project first if you need to preserve the supplied evidence.
It records Python and installed package versions under `results/runs/` for each run.

```bash
python run_workflow.py --list
python run_workflow.py
```

To resume or rerun a specific section:

```bash
# Resume tuning and complete all later stages (requires preparation and prior evaluations).
python run_workflow.py --from-stage tuning
# Rerun holdout only, using the existing prepared data.
python run_workflow.py --from-stage holdout --to-stage holdout
# Rebuild report figures and run checks without training.
python run_workflow.py --from-stage figures
# Run checks alone.
python run_workflow.py --from-stage check
```

Stages are `eda`, `prepare`, `holdout`, `lopo`, `tuning`, `errors`, `patterns`,
`figures`, and `check`. Stage selection does not automatically rebuild prerequisites.
After changing data, seeds, feature preparation or model settings, rerun from
`prepare` through the end; rerun EDA too when source data changes. Existing LOPO
fold exports are regenerated by the LOPO workflow. Partial or interrupted outputs
are not a completed run; restart the failed stage before proceeding.

The default kernel is `gallstone-iteration3`; use `--kernel NAME` to override it.
Notebook cells have no timeout. LOPO and stacking require many fits and can take
substantially longer than the other stages. Unit checks verify selected invariants,
not every reported score. To reproduce entirely from source, use a separate copy
containing the root Python files, requirements, notebooks, scripts, tests, `data/raw/`
and ``, then run the full workflow; generated preparation and model folders
are not required as inputs.

### Rebuild only the report visuals and PDF

When the submitted `results/` files are already present, regenerating figures
does not require retraining:

```bash
python build_modeling_figures.py
```

To compile the report, install a LaTeX distribution with `latexmk`, `pdflatex`,
Biber and the packages used in `main.tex`. Then, from the project root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd main.tex
```

Keep the complete `` directory, including all `.tex` inputs,
`references.bib`, `media/`, generated tables and figures. The modeling figure
script creates five figures (PDF and PNG) and four LaTeX tables, including the
holdout-versus-LOPO comparison. It does not regenerate the earlier EDA images
in `media/` or rewrite manually authored report tables and prose. If a
rerun changes results, reconcile those passages before compiling a new report.

## 4. Check the reproduced outputs

| Output | What to verify |
| --- | --- |
| `data/processed/manifest.json` and split membership CSVs | Seeds 42, 7, 21, 84 and 123; 223 training, 48 validation and 48 test rows per seed. |
| `results/holdout/metrics.csv` and `predictions.csv` | Full fixed run: 120 metric rows and 5,760 prediction rows across six families (including dummy), two selectors and validation/test partitions. |
| `results/holdout/validation_winners.csv` and `selected_test_metrics.csv` | Five validation-selected winners and their corresponding test results. |
| `results/lopo/metrics.csv` and `predictions.csv` | Six metric rows and 1,914 predictions: 319 held-out predictions per family, including dummy. |
| `results/tuning/seed_*_*_cv.csv` | Actual training-CV alternatives and their scores; four searched families per seed. |
| `results/tuning/metrics.csv` and `predictions.csv` | 25 metric rows and 1,200 predictions: all four validation candidates plus one selected test model per seed. |
| `results/iteration_patterns/cycle_metrics.csv` | Ten rows: five paired seeds for each of 20 and 10 predictors. |
| `results/iteration_patterns/cycle_summary.csv` | Submitted reference: mean validation AUC approximately 0.838 versus 0.816, and sensitivity 0.758 versus 0.750, for 20 versus 10 predictors. |
| `results/iteration_patterns/protocol.json` | Source-data signature, seeds, controlled change and validation-only evaluation boundary. |
| Legacy comparison source (recovery archive) | Comparison uses the same fixed RFE–CART families for holdout and LOPO; validation and test columns remain separate. |

Prediction counts are evaluation appearances, not numbers of unique patients.
Use the same data, row order, settings and package versions when investigating
any mismatch; compare metrics rather than elapsed fit times or binary pickle
contents. Preserve the original submitted CSVs before rerunning if you need an
exact before/after comparison.

## 5. Submission file checklist

Include the following, preserving the relative directory structure:

- **All six notebooks** listed in Section 2, with executed outputs visible.
- **Shared modules and workflow scripts:** `utils.py`, `modeling.py`, `run_workflow.py`,
  `scripts/iteration_pattern_analysis.py`, `tests/test_notebook_workflow.py`, and
  `build_modeling_figures.py`.
- **Setup and instructions:** `README.md`, `requirements.txt`, and the
  `reproduction-environment.txt` captured for the submitted run, if available.
- **Source data:** `data/raw/dataset_gallstone` and `data/raw/dataset_gallstone.xlsx`.
- **Evidence:** the complete `results/` directory, including
  `iteration_patterns/`, plus the complete `` directory and final PDF.
- **Saved preparation and models:** `data/processed/`, `models/holdout/`,
  `models/lopo/` and `models/tuned/` for direct inspection/loading. These can also
  be regenerated by the workflow; LOPO fold exports may make the submission
  large, so follow any course upload-size requirements.

Do not include `.venv/`, `__pycache__/`, notebook checkpoint folders or temporary
build/cache files. `archive/pre_restructure/` is optional historical material
and is not needed to reproduce the active submission. This is the technical
file checklist; course submission rules determine the required upload format.

## Data and evaluation

The source labels are **0 = gallstones present, 1 = absent**. All new modeling
notebooks recode them to **1 = present, 0 = absent**. FNR therefore measures
missed gallstone cases. A threshold of 0.5 is used throughout.

The full-data `transformed_dataset.csv` is for exploration/final fitting only.
Evaluation uses separately exported transformed partitions fitted only on
training rows. Do not randomly split the full-data transformed CSV for evaluation:
its selection and transformation have already seen every row. Source-row indices
are metadata, never predictors. Twenty selected source predictors can expand to
more than twenty columns after one-hot encoding.

Holdout seeds are 42, 7, 21, 84 and 123. Each has 223 train, 48 validation and
48 test rows. Final holdout models are refitted on 271 development rows.
Fixed logistic regression, random forest, XGBoost, LightGBM and stacking are
compared with a training-prior dummy benchmark. Both selector configurations are
reported; prefer models using validation, never the test ranking.

Individual models train on exported matrices. Stacking refits full transformations
inside its own inner folds using the same source-row membership. Tuning also
refits transformations inside inner CV; it cannot use a globally prepared matrix
without selection leakage. The tuned final classifier fits the saved development matrix.

LOPO uses one fixed seed and 319 outer folds. It assumes one row per person; use
grouped evaluation if repeated patient identifiers become available. All predictions
used for LOPO metrics are held out. Models in `models/lopo/` are separate full-data
refits for future inference, not the models used to calculate those metrics.
Expect the LOPO notebook to take longer and its cached fold datasets to use more disk space.

The training-prior dummy's pooled LOOCV AUC is 0 for this dataset: each held-out
label changes the remaining class prior in the opposite direction. This is a
fold-dependent prior artifact, not learned discrimination. The ROC chance line
represents a fixed-probability benchmark with AUC 0.5.

Summaries include accuracy, ROC-AUC, FNR, recall, specificity, precision and F1.
Repeated holdout summaries report mean, sample variance and standard deviation
(`ddof=1`). LOPO has one run, so between-run variance is undefined.
Repeated splits overlap, and tuned results reuse these internal holdout sets.

## Model loading

The supplied holdout pickle reports scikit-learn **1.9.0**. Loading it with the
local 1.7.1 environment fails during prediction (`multi_class` is missing).
Prediction with the relocated supplied model was verified successfully in the
available scikit-learn 1.9.0 environment. Use a compatible package environment
for supplied pickles, or rerun
from `prepare` to regenerate transformers and models with your installed versions.
Do not reuse the supplied preparation pickles with an incompatible scikit-learn
version. Minimum versions in `requirements.txt` do not guarantee compatibility
with historical serialized models.

Only load trusted pickle files. Keep this directory on Python's import path
because the pipelines refer to helper functions in `utils.py`.

```python
from utils import load_pickle, workflow_data, disease_probability
model = load_pickle("models/holdout/seed_42/rfe_logistic.pkl")
X, y = workflow_data()
probability_present = disease_probability(model, X)
```

Saved models accept original predictor columns and include their fitted preparation.
Error analysis is descriptive and does not establish clinical or causal explanations.

## Restructure validation

The three workflow unit tests passed in the available Python 3.13 / scikit-learn
1.7.1 environment. All six notebook startup cells and README links were checked using the available
scikit-learn 1.9.0 environment,
and `python run_workflow.py --from-stage figures` successfully regenerated report
figures/tables and ran the tests. The complete training workflow and LaTeX PDF
build were not rerun during the folder cleanup. Original notebook outputs remain
historical evidence, not proof of a fresh full run.
