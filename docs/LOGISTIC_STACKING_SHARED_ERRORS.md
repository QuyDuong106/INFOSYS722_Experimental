# Logistic regression and Stacking: paired error analysis

The final paper is `main.pdf` (18 pages). Figure 5 is a compact normal float on page 15, sharing the page with discussion text. The abstract, results, explainability, discussion and conclusion now match this analysis. No tuning was rerun.

## Paired results

| Measure | Logistic regression | Stacking |
|---|---:|---:|
| AUC | 0.873025 | 0.876170 |
| Correct / 319 | 260 | 257 |
| False negatives | 19 | 26 |
| False positives | 40 | 36 |

The models agree on 292/319 predictions (91.5%): 245 jointly correct and 47 jointly wrong. Logistic regression alone is correct in 15 cases; Stacking alone is correct in 12. The 0.003145 AUC difference is descriptive, without a test of statistical equivalence or superiority.

## All shared errors

These are source-row identifiers, not verified patient identifiers. See `figures/xai/comparison/shared_errors.csv` for both probabilities, labels and raw fields.

- Shared false negatives (15): 23, 39, 43, 73, 85, 90, 95, 105, 110, 115, 117, 141, 148, 153, 156.
- Shared false positives (32): 168, 172, 174, 176, 177, 178, 192, 194, 195, 199, 200, 203, 205, 206, 207, 215, 221, 227, 239, 243, 291, 297, 299, 302, 304, 306, 309, 311, 315, 316, 317, 318.

## Explained cases

For each error type, the selected record maximises the smaller wrong-class confidence across both models. These are retrospective extreme examples and do not represent typical cases.

| Source row | Error | Logistic P(present) | Stacking P(present) | LR LIME R² | Stacking LIME R² |
|---|---|---:|---:|---:|---:|
| 156 | FN | 0.000535 | 0.088485 | 0.295961 | 0.312972 |
| 178 | FP | 0.999481 | 0.992666 | 0.316363 | 0.386375 |

For row 156, extracellular water and vitamin D push probability down in both models, while intracellular water pushes it up. For row 178, total body water pushes probability up and intracellular water pushes it down in both. These contributions explain fitted predictions relative to training backgrounds, not clinical labels or causal mechanisms. LIME also assigns positive weights to the conditions hyperlipidaemia=0 and CRP<=0 in both records and models. The thresholds are training discretisation boundaries, not clinical decision limits.

LIME uses all 38 fields and 3,000 perturbations per model and case with matched training backgrounds and conditions. Figure 5 displays four fields per panel selected by mean absolute contribution across models; full vectors are saved. Limited neighbourhood R² restricts interpretation. LR's FN surrogate estimate is -0.016954; Stacking's FP estimate is 1.027031. These unconstrained linear outputs are not valid probabilities and are reported without clipping.

## Logistic regression understanding

Logistic regression is additive in selected, transformed and encoded inputs before applying the logistic link. Its coefficients are not raw-unit clinical odds ratios. The selected C differs by outer exclusion: 6.583468 for row 156 and 1.261484 for row 178; both retain all 38 fields. Coefficient dot products plus intercept reproduce logits -7.531854 and +7.562127 and their saved probabilities. Coefficients and intercepts are saved in `logistic_coefficients.csv`.

Across the same twenty global SHAP exclusions, logistic regression's leading mean absolute fields are estimated bone mass (0.1611), CRP (0.1609) and intracellular water (0.1343). Stacking emphasises CRP (0.1683) and intracellular water (0.1341), with lower bone-mass attribution (0.0863). This is a sample of twenty exclusion-specific models, not a population importance ranking. Correlated body-composition variables complicate independent interpretation.

## Reproduction and files

From the project root:

```sh
.venv/bin/python scripts/compare_logistic_stacking_xai.py
python3 scripts/build_paper.py
```

The script validates completed run signatures, row/label alignment and saved probabilities. LR uses saved excluded-record pipelines; the two local Stacking pipelines are reconstructed from saved training-only parameters. Global Stacking SHAP reuses validated values for the same twenty rows and settings; LR SHAP is newly computed. Kernel SHAP uses 25 training-only background rows and 256 samples; seed 42. Local SHAP additivity and LR coefficient reconstruction are checked numerically.

All outputs are under `figures/xai/comparison/`:

- `paired_predictions.csv`, `shared_errors.csv`, `error_overlap.json`: complete audit.
- `protocol.json`: sources, settings, sampled and selected rows.
- `global_shap_comparison.csv`, `logistic_sampled_shap.csv`: global values.
- `local_explanation_metadata.csv`, `local_lime_weights.csv`, `logistic_coefficients.csv`: fidelity, full LIME and logistic coefficients.
- `{logistic,stacking}_{fn,fp}_shap.csv` and corresponding `_lime.html`: four complete local explanations.
- `logistic_global_shap.pdf/png`: Figure 4.
- `shared_errors_shap_lime.pdf/png`: Figure 5.

`global_shap_comparison.tex` supplies the paper's paired SHAP table. Both explanation notebooks read these outputs without regeneration by default. Earlier Stacking-only outputs remain for provenance and are not active paper figures.
