# Stacking rerun with validation and experiment provenance

Use `scripts/tune_stacking.py`. It invokes the existing version-4 training engine with Stacking only and mandatory saved fold pipelines. The shared training source was not changed, so the completed individual-model source fingerprints remain valid. The retained `results/stacking_v3` experiment and the paper's current scores are preserved.

## Full rerun command

From the project root:

```sh
cd /Users/quyduong106/projects/INFOSYS722_Experimental
.venv/bin/python scripts/tune_stacking.py \
  --reference-dir results/dynamic_individual_v4 \
  --output-dir results/stacking_v4 \
  --seed 42 --trials 20 \
  --k-values 10 20 30 38
```

Three inner folds, decision threshold 0.5, raw-input pipelines, source-field RFE and saving fitted outer pipelines are enforced. The bases are XGBoost, LightGBM and Ridge. Three-fold cross-fitted outputs train a standardised logistic meta-classifier with C=1 and no raw-input passthrough. Tuning selects all three base configurations and the shared source-feature count.

Run the identical command after an interruption to resume completed folds. Do not edit the computational source, change dependencies or alter the reference experiment mid-run. Existing incompatible or nonempty uninitialised directories are rejected. No old results are overwritten. A full run involves 319 outer exclusions and 6,380 trials, plus repeated inner/base fitting; its runtime depends on the machine.

Preview the fixed protocol without fitting or creating an experiment directory:

```sh
.venv/bin/python scripts/tune_stacking.py --dry-run
```

## Fairness and transparency

The wrapper checks the completed four-model reference for matching dataset signature, target labels, source-row coverage, training-source fingerprint, package versions, seed, inner folds, feature-count candidates, objective, preprocessing declaration and sampler schedule. The complete rerun must match the reference's twenty-trial budget. The excluded record cannot influence preparation, feature selection, validation or tuning.

The manifest records search spaces directly from the actual suggestion functions. Including feature count, logistic regression searches 2 controls, random forest 5, XGBoost 7, LightGBM 7 and Stacking 14. Equal trial counts constitute a common candidate-budget comparison; they do not prove equal compute, convergence or equally exhaustive exploration. Changing only Stacking to a larger complete-run budget is rejected by this entry point. A sensitivity study should be defined separately with matched budgets across affected configurations, not substituted silently for the current comparison.

The manifest and source snapshots are written before fitting, freezing this rerun's configuration. This is not retrospective preregistration of earlier cohort exploration. All families remain candidate configurations in an internal comparison; selecting a winner from their outer results is not independent validation of that selection.

## Saved evidence

```text
results/stacking_v4/
├── experiment_manifest.json             # fixed protocol, actual ranges, reference/raw/source hashes
├── run_config.json                      # engine configuration, packages, code fingerprints
├── progress.json                        # complete/incomplete and evaluated-fold coverage
├── provenance/
│   ├── source/                          # exact shared-source and entry-point snapshots
│   ├── pip-freeze.txt                    # dependency listing
│   └── invocations/*.json                # command, timing, status, platform, Python, thread environment
├── folds/
│   ├── fold_0000_stacking.json           # IDs, best parameters, all three selectors, artifact checksum
│   └── fold_0000_stacking_trials.csv     # every trial, three validation metrics, inner feature names
├── CV_Metrics/stacking.csv               # selected-trial means, sample SD and individual inner folds
├── fold_models/fold_0000_stacking.pkl    # complete evaluated outer pipeline, including all bases/meta
├── feature_selection/                   # selected flags, RFE ranks and CART scores per base/fold
├── feature_selection_all_folds.csv
├── feature_selection_frequency.csv
├── feature_count_summary.csv
├── crossfit_partitions/fold_0000.json    # outer and inner-training Stacking cross-fit maps
├── validation/
│   ├── trial_summary.csv                # all trial objectives, parameters, durations and winner flags
│   ├── all_trial_inner_metrics.csv      # all three validation metrics/feature lists for every trial
│   ├── selected_trial_summary.csv       # chosen counts, inner AUC and recorded fitting time
│   └── optimization_summary.json        # coverage, trial counts and measured timings
├── fold_best_parameters.csv
├── predictions.csv                      # only produced after full outer evaluation
├── metrics.csv                          # pooled outer-test results, full evaluation only
└── summary.csv
```

Engine checkpoints, per-trial CSVs, saved models and selected-trial CV summaries are saved as folds finish. Additional long-form validation summaries and cross-fit maps are exported when an invocation completes or exits through a handled interruption/error. Partial outputs are labelled incomplete and must not be reported as a completed evaluation. Abrupt process termination can prevent the final exports; rerun the unchanged command to resume.

Cross-fit maps are deterministically derived from saved training IDs and the declared seeded splitters, common to the three bases and all candidate trials. They are not labelled as a historical instrumented trace of every fit. The boundary test separately checks the actual implementation's nesting.

Inner validation metrics describe the selection procedure and are optimistic as performance estimates after choosing the best trial. Use `predictions.csv` and `metrics.csv` for reported outer-test performance. Saved outer models can be loaded using `feature_selection_evidence.load_fold_pipeline` for subsequent whole-pipeline XAI. Each base has its own saved feature records; do not count Stacking three times when pooling model-level retention frequencies.

## Verification

The test suite includes real two-trial Stacking fits, saving/loading evaluated pipelines, resume, source-snapshot integrity, validation exports, inner/outer partitions and rejection of unmatched full-run budgets/seeds. The final 14-test suite passed. A real-cohort two-exclusion/two-trial smoke run is saved at `results/stacking_v4_smoke_final`; both fitted pipelines were reloaded and their saved probabilities verified, with twelve inner-validation records exported. This is not a replacement for the full rerun.

For another smoke run, choose a fresh directory:

```sh
.venv/bin/python scripts/tune_stacking.py \
  --trials 2 --max-folds 2 \
  --output-dir results/stacking_v4_smoke_new
```

The lower budget is allowed only for an incomplete smoke request and is explicitly recorded as unmatched. A two-trial smoke directory cannot be resumed as a twenty-trial full experiment. Use `results/stacking_v4` for the full comparison.

## Paper interpretation and pending integration

The paper still reports completed version-3 Stacking (AUC 0.8762) and completed dynamic individual models, including logistic regression (AUC 0.8730). All five performances remain in the results table. Logistic regression has stronger observed accuracy, sensitivity and F1 at threshold 0.5; Stacking has higher observed specificity and precision, and their errors overlap. These are descriptive trade-offs, not statistically demonstrated superiority, equivalence or an independently validated preferred model.

The manuscript now says that families and search ranges were specified for their respective runs, while prior cohort-informed development makes the overall comparison retrospective. New provenance is not claimed for old results. After the full new run completes, verify it before updating every result/feature/XAI consumer from `stacking_v3` to `stacking_v4`, then regenerate tables, shared-error analysis and explanations. The rerun script does not switch the paper's sources automatically or assume its new scores will equal the old ones.
