# Preprocessing consistency audit

The current paper's dynamic individual-model training, retained Stacking reconstruction, feature-selection evidence and whole-pipeline XAI share the evaluation pipeline in `utils.make_evaluation_model`. `optuna_lopo.make_tuning_template` supplies Stacking's XGBoost, LightGBM and Ridge bases using this same factory. The shared order is training-only imputation, raw-field CART RFE, continuous Yeo–Johnson transformation with standardisation and categorical one-hot encoding, followed by the classifier.

## Actual field handling

The raw data contain 38 numerically stored predictors. Numeric dtype does not mean all variables are continuous. The schema treats 31 fields as continuous and the following seven as categorical:

| Field | Observed codes |
|---|---|
| gender | 0, 1 |
| comorbidity | 0, 1, 2, 3 |
| cad | 0, 1 |
| hypothyroidism | 0, 1 |
| hyperlipidemia | 0, 1 |
| dm | 0, 1 |
| hfa | 0, 1, 2, 3, 4 |

These are the implementation's schema assignments; this audit does not independently establish whether multicategory codes should be nominal or ordinal clinically. One-hot encoding is active, not a dormant fallback. In the saved logistic pipeline for exclusion 0, all 38 fields are selected: 31 continuous columns plus 19 one-hot columns produce 50 classifier inputs. Count k refers to raw fields before expansion. If RFE excludes categories, fewer encoded columns result.

Continuous values use median imputation and `PowerTransformer(method='yeo-johnson', standardize=True)`. Categories use mode imputation and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. CART RFE precedes one-hot encoding, so its splits use the original numeric codes. This is a common selection design across the current models, not a claim that category codes have meaningful continuous distances.

## Verified evidence

- Inspected the shared training factory and tuning call chain.
- Loaded the saved exclusion-0 pipeline for each of logistic regression, random forest, XGBoost and LightGBM with the verified loader; the loader checks the held-out probability and artifact provenance.
- Confirmed pipeline step order, fitted power-transform settings, actual encoder inputs/output names and transformed widths.
- Inspected all three Stacking base templates for the same preparation sequence.
- Verified current computational source hashes exactly match the completed individual-model run configuration.
- Fitted and predicted with a continuous-only subset of the real data: the categorical branch had zero columns and the pipeline worked normally.
- Inspected the current SHAP/LIME scripts: they query complete evaluated pipelines, using the same categorical-field function. Training-only filling provides finite raw perturbation inputs; LIME discretisation is an explanation operation, not an alternative predictor transformation.
- Plotting/summary scripts consume saved predictions or selection evidence and do not fit a different predictor transformation.

This is a code-path audit plus four saved-pipeline checks, not a reload of every saved pipeline or a new model evaluation. Machine-readable observations are in `preprocessing_consistency.json`.

## Historical and auxiliary exceptions

`modeling.make_pipeline` is an older pipeline: median/mode imputation, optional StandardScaler, one-hot encoding, then ANOVA SelectPercentile. It has no Yeo–Johnson stage and is not equivalent to the current CART-RFE evaluation pipeline. Current `make_evaluation_model` extracts its classifier only and replaces the earlier preprocessing; Stacking bases are also replaced.

`xai_utils.setup` uses the current transformation factory but a separate 85/15 evaluation; its generic Stacking path is distinct from the retained three-base paper experiment. It is not the entry point used by the current two paper-explanation notebooks. Archived code and historical results should not be described as sharing the current evaluation procedure.

No training code or saved results were changed by this audit. The current categorical treatment should remain as documented unless a deliberate schema change is followed by new evaluation and explanations.
