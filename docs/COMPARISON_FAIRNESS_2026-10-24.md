# Current comparison and rerun decision — 24 October 2026

## Retained paper evidence

Only one stacking configuration is reported: the latest completed `results/stacking_v4` run (XGBoost, LightGBM and ridge bases; fixed logistic meta-classifier C=1). Its 319 predictions exactly match the earlier version-3 base-tuned run: AUC 0.8761695102, accuracy 0.8056426332, recall 0.8385093168 and specificity 0.7721518987. The historical version-1 meta-tuned stack is excluded from current tables. Historical raw output remains archived for provenance rather than being deleted or substituted for current evidence.

The four individual models use `results/dynamic_individual_v4`. Report-table and evaluation-plot generators now reference `stacking_v4`. The feature-importance table and plot were checked against the latest saved selectors, with stacking counted once per exclusion rather than three times.

## Common protocol: matched

The saved configurations have identical dataset signature, training-source hashes, package versions, seed 42, 20 trials, three stratified inner folds, deterministic outer record exclusions, candidate feature counts [10,20,30,38], RFE policy, objective, threshold 0.5 and sampler-seed schedule. All five pooled test metric sets were recomputed from their 319 unique held-out predictions. The latest stacking's three full-outer-training base selectors agree; internal cross-fit selectors fit their respective training partitions independently.

Preprocessing is training-only median/mode imputation, CART-based RFE, `PowerTransformer(method="yeo-johnson", standardize=True)` for selected continuous fields and one-hot encoding for selected coded fields. The meta-classifier's extra scaler operates on base outputs, not raw features. It is part of the stacking architecture rather than a different raw-data scaling policy.

## What is not identical

The **candidate feature pool and search options** are identical; the **chosen k and feature subset** need not be. Each model selects its own k using validation within the outer training set. The selected all-38 counts are logistic 290/319, forest 126/319, XGBoost 173/319, LightGBM 151/319 and stacking 239/319. Do not state that every model used the same selected features.

This is a fair common-protocol comparison of complete model-development procedures, not an isolated classifier comparison with fixed inputs. Equal 20-trial budgets also do not give equal computation or optimization coverage: stacking has more searched controls and additional cross-fitting. Its ridge component has no standalone counterpart in the present comparison. Single-cohort retrospective family selection, uncertain patient identity and no external validation remain limitations.

## What must be rerun?

**Nothing is required to repair the earlier fixed-k versus dynamic-k mismatch:** all four individual dynamic runs and the matched stacking version-4 run are complete. No tuning was started in this audit.

For a new, stricter comparison in which every classifier receives exactly the same source fields, run a separately labelled sensitivity experiment using all 38 fields. Passing only k=38 causes RFE to retain the entire pool, while preserving matched transformation and inner-CV policies. Do not replace the existing dynamic-search results or treat this retrospective sensitivity choice as preregistered confirmation.

```bash
cd /Users/quyduong106/projects/INFOSYS722_Experimental

# Complete these four models first to establish the reference run.
.venv/bin/python scripts/tune_dynamic_models.py \
  --seed 42 --trials 20 --inner-folds 3 --k-values 38 \
  --output-dir results/all38_individual_v4

# Then fit stacking against the same reference protocol.
.venv/bin/python scripts/tune_stacking.py \
  --reference-dir results/all38_individual_v4 \
  --seed 42 --trials 20 --k-values 38 \
  --output-dir results/all38_stacking_v4
```

For an identical reduced subset, define a common training-only selector and fixed k independently of the evaluated classifier; apply that policy in every inner fit and outer refit, then rerun **all five** models. Do not select one full-cohort feature list using outcome information before outer exclusion. Additional independently prespecified seeds or external data address different questions and cannot be fabricated by averaging single-patient folds.

Machine-readable checks: `docs/audit/comparison_alignment_2026-10-24.json`.
