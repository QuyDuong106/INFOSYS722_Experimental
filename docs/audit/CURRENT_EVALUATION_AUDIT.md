# Current five-model evaluation audit

This supersedes the fixed-k comparison findings in the 22 October audit. Scope: the paper's completed `results/dynamic_individual_v4` and `results/stacking_v3` evidence, current training/selection/preprocessing, saved individual pipelines, explanation paths, and legacy runner compatibility. No model tuning, result replacement or manuscript edits were performed.

## Assessment

No direct held-out-record leakage or metric inconsistency was found in the current paper's inspected fitting path and saved evidence. The comparison is aligned on data, preprocessing policy, outer exclusions, inner-validation design, objective and decision threshold. This does not certify absence of all leakage, statistical uncertainty, selection bias or source-data problems. It is a common-protocol internal comparison, not a controlled equal-compute comparison or an external validation.

## Completed checks

| Check | Evidence/result |
|---|---|
| Data | 319 rows, 38 predictors, no missing cells or exact duplicate predictor rows; outcome/row ID excluded from predictors |
| Shared configuration | Same raw-data signature, recorded package versions, seed 42, 20 trials, three inner folds, threshold 0.5, RFE, candidate counts and tuning objective |
| Outer predictions | All five models have exactly one correctly aligned held-out prediction per source row: 1,595 total |
| Checkpoint alignment | All 1,595 checkpoints agree with row IDs, target labels, saved probabilities and sampler seed schedule |
| Trial completeness | All 31,900 trials are COMPLETE and finite; each objective agrees with the mean of its three recorded inner AUCs |
| Winner selection | All selected trials maximise recorded inner AUC; their parameters, feature counts and recorded validation AUCs agree with checkpoints |
| Individual training boundaries | All 1,276 checkpoints contain exactly the other 318 training IDs; all 3,828 saved inner splits exactly reproduce training-only stratified splits |
| Individual feature records | 76,560 trial-by-inner-fold selected feature lists have the required number of unique source names; all saved outer selectors agree with their checkpoint records |
| Evaluated individual artifacts | All 1,276 pipeline checksums verified; every pipeline reloaded and its excluded-record probability reproduced to tolerance 1e-8 |
| Preprocessing | All reloaded pipelines have impute/select/transform/model steps; continuous Yeo–Johnson with standardisation, and categorical one-hot encoding with unknown categories ignored |
| Stacking implementation | Full preparation/selection pipelines inside three bases; out-of-fold base outputs train the fixed logistic meta-classifier; no raw-feature passthrough |
| Fresh Stacking boundary test | Instrumented RFE.fit: 48 selector fits from one tuning trial and final refit. No outer test record enters fitting; inner validation records are absent from their fitting blocks; base cross-fit exclusions partition each fitting set |
| Metrics | AUC, accuracy, precision, recall, specificity, F1 and FNR independently recomputed from pooled held-out probabilities and matched saved metrics for all five models |
| Explanations | Current code queries whole excluded-record pipelines with training-only backgrounds; saved paired local SHAP vectors reconstruct probabilities; logistic coefficient reconstruction previously verified |
| Tests | Existing 11 tests passed; new Stacking fit-boundary test also passed (12 total) |

The fresh Stacking boundary test validates the current implementation on synthetic indexed observations. It is not a historical trace of the original 319 fitted Stacking models. The retained Stacking version-3 checkpoints do not store fitted outer artifacts, training IDs, inner split IDs, per-trial feature lists or original source fingerprints. Its 319 trial/checkpoint records can be verified, and selected models have been reconstructed for XAI with saved-probability agreement, but it has weaker historical provenance than the four version-4 individual runs. Do not describe it as having the same saved-artifact evidence.

## Remaining findings

### 1. Research-design selection bias remains unresolved

Each family is tuned within outer training data. Selecting a preferred family after examining all pooled outer results is an additional retrospective choice; the winning score is not an independently validated estimate of the complete selection procedure. Prior cohort exploration and architecture revision cannot be undone by correct per-fold fitting. Results should remain exploratory internal comparisons. A frozen development strategy requires independent assessment for external-performance claims.

### 2. Statistical superiority, equivalence and precision have not been established

Single-seed evaluation supplies no seed sensitivity assessment. The 319 outer training sets overlap heavily; fold-level outcomes and selected-trial validation scores are not 319 independent experiments. Avoid t-tests treating these fold scores as independent, and do not interpret CART frequency/SD as population confidence intervals. Best-trial inner scores were used for selection and should not replace outer-test results. AUC 0.8762 versus 0.8730 is a descriptive difference, not proof of superiority or equivalence. LOPO removes a single random outer split but does not remove uncertainty.

### 3. Fair protocol does not mean equal optimisation difficulty

All families have twenty trials, but logistic regression searches C plus k, while Stacking searches many base controls plus shared k and performs additional cross-fitting. No tuning-convergence or budget-sensitivity analysis establishes equally thorough optimisation. Runtime records are not controlled common-hardware benchmarks. The results compare configured procedures; they do not isolate architecture as the only cause of a difference. Missing Ridge-only and component ablations further limit attribution to particular ensemble components.

### 4. Patient independence and predictor availability are not verified

Source-row IDs are not verified patient identifiers. Exact row uniqueness cannot exclude repeated people. Repeated-person observations require grouped exclusion at outer and inner levels, including Stacking cross-fitting. The current repository cannot establish that every predictor was acquired before the intended clinical decision or independently of diagnosis/treatment. These are unresolved possibilities, not demonstrated leakage. This remains classification of existing status, not disease-onset prediction.

### 5. Further validation depends on the intended claim

Current internal evaluation is usable as technical evidence. It does not establish transfer to another hospital/population, calibration of probabilities, clinical benefit or an optimised clinical threshold. External testing and patient-provenance confirmation are needed for stronger claims; calibration and threshold development must use training/development data. Merely adding another split of this explored cohort does not create external validation.

### 6. Feature selection is correctly partitioned but is not biological validation

All models use the same CART-RFE selection policy. Their winning k and fields may differ. RFE runs on imputed raw fields before Yeo–Johnson and one-hot encoding; numeric-coded categories therefore enter tree splits as codes. Impurity rankings can depend on split opportunities and correlated substitutes. The all-fold top-ten table is a post-evaluation summary and must not be fed back as a global prefilter without a new properly separated evaluation. Frequent retention, especially when k=38, does not prove necessity. The categorical schema is an implementation choice, not independently verified clinical measurement semantics.

### 7. XAI is aligned with evaluation but remains exploratory

Global comparisons use twenty matched exclusions, not the full population. Local shared cases were selected retrospectively for high joint wrong-class confidence. LIME R² is 0.296–0.386, and unconstrained outputs include values outside [0,1]. SHAP additivity checks numerical reconstruction, not stability, causality or clinical truth. Correlated features and synthetic perturbations constrain interpretation. Current explanation notebooks read the paired paper outputs; auxiliary xai_utils.setup runs a different holdout experiment and is not interchangeable with them.

### 8. The full historical runner is not a clean reproduction of the current paper

The preparation notebook still requests numeric.named_steps["standardize"], whereas the current transformer combines transformation and standardisation in PowerTransformer. This confirmed lookup mismatch can stop a fresh notebook run. Legacy export caches also omit source/package fingerprints, so matching metadata alone does not establish current preprocessing provenance. The current raw-input LOPO entry point bypasses these exports; no contamination of the reported results was found. Historical pipelines and holdout stages are separate experiments and should not be represented as identical current-paper evaluations.

## Defensible reporting statement

"All candidate models were evaluated on common outer exclusions using training-only preprocessing, dynamic feature selection and parameter optimisation. No direct held-out-record leakage was identified in the inspected implementation and saved evidence. The results constitute an internally aligned, exploratory comparison; unequal optimisation complexity, retrospective development, unverified patient independence and absent external validation limit stronger conclusions."

## Reproduction

From the project root:

```sh
.venv/bin/python scripts/audit_current_evaluation.py
.venv/bin/python -m unittest discover -s tests -v
```

The audit writes only `docs/audit/current_evaluation_checks.json`. It reads existing models and predictions and does not rerun tuning. It checks all individual pipeline predictions but does not reconstruct every Stacking fold. The new test is `tests/test_stacking_fit_boundaries.py`.

Methodological references:

- Training-only preprocessing and selection: https://scikit-learn.org/stable/common_pitfalls.html
- Cross-validation uncertainty and grouped observations: https://scikit-learn.org/stable/modules/cross_validation.html
- Cross-fitted meta-classifier design: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
- Selection bias: Varma and Simon (2006), https://doi.org/10.1186/1471-2105-7-91
