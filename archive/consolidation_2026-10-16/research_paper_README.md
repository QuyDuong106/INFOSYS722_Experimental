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
Existing fixed results under `results/lopo/` and manuscript numbers remain
historical evidence until a complete tuned evaluation is run and reviewed.
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

# Research manuscript

Build from this directory with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` (requires LaTeX and Biber).

`main.tex` assembles the abstract, introduction, related work, methods, results, discussion, and conclusion. The layout is a venue-neutral single-column article; a publisher template can be applied once a venue is chosen.

`related_work.tex` combines the former related work and literature review into one thematic section with a comparison table and study positioning. `lit_review.tex` is a compatibility input, not a second section. Original review sources and bibliography are saved under `archive/related_work_merge_2026-09-29/`. The merged text synthesises the supplied literature draft; its reported study results were not independently re-verified. Duplicate bibliography keys were removed. Numerical findings of the present study remain unchanged; no experiments were rerun. The original report section sources are retained in `report_archive/` for traceability and are not included in the manuscript. Figures and generated tables are retained. The analysis code and notebooks now live here. Supporting data, saved models, and results were copied here from the original project. See `WORKFLOW.md` for environment setup and execution instructions. Run `python run_workflow.py --list` to list stages.

## Report explainability figures

Run `python scripts/build_xai_report.py` after a completed logistic Optuna run under `results/optuna_logistic/`. The script reconstructs excluded-record tuned pipelines and checks that predictions match saved evaluation probabilities. It exports the sampled global SHAP figure, false-negative and false-positive SHAP/LIME figures, attribution CSVs, local fidelity and provenance under `figures/xai/`. The manuscript integrates these via `xai_results.tex`; the original fixed-model table remains labelled separately from this tuned explanatory analysis.
