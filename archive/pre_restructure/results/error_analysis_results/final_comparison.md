# Final model comparison

Fixed settings from stacking.ipynb; positive class = 1; threshold = 0.5. Model/inner-CV randomness stays at 42; only split seeds vary.

## 1. 70/15/15 test — five splits, mean ± sample standard deviation

Seeds: [63990, 36547, 17892, 85181, 2641]. Each split: 223 training / 48 validation / 48 test, with final refit on 271 development rows. Sample variance is saved separately. These overlapping splits measure variability, not an independent confidence interval.

| Model | roc_auc | accuracy | precision | recall | f1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.853 ± 0.054 | 0.783 ± 0.050 | 0.794 ± 0.056 | 0.767 ± 0.063 | 0.779 ± 0.053 | 4.800 ± 1.483 | 5.600 ± 1.517 |
| Random forest | 0.873 ± 0.055 | 0.804 ± 0.052 | 0.815 ± 0.069 | 0.792 ± 0.042 | 0.802 ± 0.048 | 4.400 ± 1.949 | 5.000 ± 1.000 |
| XGBoost | 0.874 ± 0.058 | 0.808 ± 0.040 | 0.857 ± 0.080 | 0.750 ± 0.042 | 0.797 ± 0.033 | 3.200 ± 2.280 | 6.000 ± 1.000 |
| LightGBM | 0.860 ± 0.050 | 0.792 ± 0.039 | 0.816 ± 0.061 | 0.758 ± 0.054 | 0.784 ± 0.039 | 4.200 ± 1.643 | 5.800 ± 1.304 |
| Stacking | 0.891 ± 0.053 | 0.800 ± 0.032 | 0.813 ± 0.051 | 0.783 ± 0.035 | 0.797 ± 0.028 | 4.400 ± 1.517 | 5.200 ± 0.837 |

## 2. LOOCV — 319 held-out predictions per model

| Model | roc_auc | accuracy | precision | recall | f1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.863 | 0.784 | 0.807 | 0.741 | 0.772 | 28 | 41 |
| Random forest | 0.852 | 0.781 | 0.789 | 0.759 | 0.774 | 32 | 38 |
| XGBoost | 0.837 | 0.768 | 0.780 | 0.741 | 0.760 | 33 | 41 |
| LightGBM | 0.844 | 0.781 | 0.782 | 0.772 | 0.777 | 34 | 36 |
| Stacking | 0.875 | 0.781 | 0.793 | 0.753 | 0.773 | 31 | 39 |
