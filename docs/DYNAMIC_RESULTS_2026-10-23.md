# Dynamic feature-selection results: 23 October 2026

Completed evidence: four individual models from `results/dynamic_individual_v4` and retained Stacking from `results/stacking_v3`. Every model has 319 held-out predictions, searches k=10/20/30/38 and uses twenty trials, three inner validation folds, seed 42 and threshold 0.5. The summary script checks signatures, environments, selection counts, checkpoint probabilities and recomputed metrics.

## Predictive results

| Model | AUC | Accuracy | Sensitivity | Specificity | F1 | FN |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.8730 | 0.8150 | 0.8820 | 0.7468 | 0.8280 | 19 |
| Random forest | 0.8386 | 0.7774 | 0.7950 | 0.7595 | 0.7829 | 33 |
| XGBoost | 0.8448 | 0.7806 | 0.7950 | 0.7658 | 0.7853 | 33 |
| LightGBM | 0.8465 | 0.7743 | 0.7826 | 0.7658 | 0.7778 | 35 |
| Stacking | 0.8762 | 0.8056 | 0.8385 | 0.7722 | 0.8133 | 26 |

Logistic regression leads accuracy, sensitivity and F1; Stacking leads AUC, specificity and precision. The AUC difference between these two is approximately 0.0031. This is descriptive evidence, not a significance claim. The former feature-count restriction on individual models has been removed, while search dimensionality, compute and retrospective exploration still limit interpretation.

## Ten most frequent features

| Feature | Appearances / 1,595 | Retained (%) | Mean CART score |
|---|---:|---:|---:|
| CRP | 1595 | 100.00 | 0.4323 |
| Vitamin D | 1595 | 100.00 | 0.1325 |
| Estimated bone mass | 1595 | 100.00 | 0.0743 |
| Hyperlipidaemia | 1586 | 99.44 | 0.0431 |
| ALP | 1580 | 99.06 | 0.0117 |
| VMA (source field) | 1578 | 98.93 | 0.0318 |
| ECF/TBW ratio | 1577 | 98.87 | 0.0411 |
| Extracellular water | 1574 | 98.68 | 0.0351 |
| AST | 1557 | 97.62 | 0.0305 |
| Weight | 1515 | 94.98 | 0.0224 |

The denominator is five models × 319 exclusions. Stacking contributes once per exclusion because its three full-training selectors agree; it is not counted three times. Ties in retention count are ordered by mean CART score. A separate four-individual-model-only summary uses 1,276 opportunities. These are overlapping model fits on 319 individuals, not 1,595 independent patients.

CART score means normalised impurity importance from the final RFE tree, with zero assigned to removed features before averaging. Retention does not establish necessity or biological importance. For example, ALP is retained frequently but has a smaller CART score than several other top-ten fields. The label VMA preserves the source field `vma_kg`; no unverified physiological expansion is inferred.

## Files

- Paper: `main.pdf` (18 pages).
- Frequency and CART importance figure: `figures/selection/dynamic_top10_features.pdf` and `.png`.
- Model-specific frequency heatmap: `figures/selection/dynamic_feature_frequency_by_model.pdf` and `.png`.
- Top ten: `figures/selection/dynamic_top10_features.csv`.
- All 38 fields: `figures/selection/dynamic_all_features.csv`.
- Per-model summaries: `figures/selection/dynamic_per_model_features.csv`.
- Four-individual-model-only summary: `figures/selection/dynamic_individual_features.csv`.
- Selected-k counts: `figures/selection/dynamic_feature_count_summary.csv`.
- Recomputed model metrics: `figures/selection/dynamic_model_metrics.csv`.
- Protocol and sources: `figures/selection/dynamic_summary_protocol.json`.

Rebuild tables and charts with `.venv/bin/python scripts/update_dynamic_results.py`, then rebuild the paper with `python3 scripts/build_paper.py`. The script writes `cart_importance.tex` and `dynamic_performance.tex`. The paired SHAP/LIME update now compares logistic regression and Stacking: 292 agreements, 47 common errors (15 FN and 32 FP). See [the shared-error report](LOGISTIC_STACKING_SHARED_ERRORS.md). Regenerate explanations separately using `.venv/bin/python scripts/compare_logistic_stacking_xai.py` before rebuilding the paper.
