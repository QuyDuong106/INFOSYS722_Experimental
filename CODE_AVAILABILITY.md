# Code availability and current Optuna execution

The implementation is available at [INFOSYS722_Experimental](https://github.com/QuyDuong106/INFOSYS722_Experimental). Clone URL: `https://github.com/QuyDuong106/INFOSYS722_Experimental.git`.

The latest computational update and complete five-model version-4 evidence were published in [commit 0b6b64a](https://github.com/QuyDuong106/INFOSYS722_Experimental/commit/0b6b64a). That commit contains the current Optuna execution code; this documentation update does not change the training algorithm or claim a new run.

## Execution paths

- `optuna_lopo.py`: shared nested LOPO engine and Optuna search spaces.
- `scripts/tune_dynamic_models.py`: individual models, jointly tuning RFE count and classifier parameters.
- `scripts/tune_stacking.py`: base-tuned stacking with a matched reference protocol and experiment provenance.
- `utils.py`, `modeling.py`: preparation and model definitions.
- `cv_reporting.py`, `feature_selection_evidence.py`: validation CSVs, selector records, saved pipelines and source fingerprints.
- `scripts/compare_logistic_stacking_xai.py`: paired whole-pipeline explanation analysis.

All candidates use training-only three-fold validation, seed 42, twenty trials, RFE count candidates 10/20/30/38 and Yeo–Johnson with standardisation enabled. Stacking tunes XGBoost, LightGBM and ridge bases while keeping the logistic meta learner fixed. Completed outputs remain in `results/dynamic_individual_v4` and `results/stacking_v4`; selected-trial validation CSVs are distinct from pooled outer-test metrics.

## Commands

From the repository root, use the existing configured environment:

```bash
# Inspect the stacking protocol without training.
.venv/bin/python scripts/tune_stacking.py --dry-run

# New individual-model experiment; do not overwrite the completed evidence.
.venv/bin/python scripts/tune_dynamic_models.py \
  --seed 42 --trials 20 --inner-folds 3 --k-values 10 20 30 38 \
  --output-dir results/new_individual_v4

# After that reference run completes, run matching stacking.
.venv/bin/python scripts/tune_stacking.py \
  --reference-dir results/new_individual_v4 \
  --output-dir results/new_stacking_v4 \
  --seed 42 --trials 20 --k-values 10 20 30 38
```

Resume with the identical settings and directory. Changed source, dependencies or protocol require a separate directory. The recorded source snapshots and package metadata describe the original completed runs. No dependency reinstall or full retraining is required to inspect existing results.

The manuscript is maintained locally and is not included in this code-only documentation commit. A public identifying repository link is suitable for an identified manuscript; initial double-blind submissions need an appropriate anonymous availability statement.
