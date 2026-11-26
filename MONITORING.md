# Data Drift Monitoring

## Setup

Drift is checked with Evidently (`src/monitor_drift.py`), which compares a
reference dataset (`data/reference.csv`, 70% random sample of the Heart Disease
data, about 212 rows) against a current dataset (`data/current.csv`, the
remaining 30%, about 91 rows). All 13 input features are tested; the target is
excluded. The check fails the pipeline (exit code 1) if the share of drifted
features exceeds 40%, set in `configs/config.yaml`. The full HTML report is in
`reports/drift_report.html`, and in CI it is available as the `drift-reports`
artifact of each workflow run.

## Result

| Metric | Value |
|---|---|
| Features checked | 13 |
| Features drifted | 1 |
| Drift share | 7.7% |
| Threshold | 40% |
| Threshold exceeded | No |

## 1. Which features showed drift and why?

Only `slope` (the slope of the peak exercise ST segment, a 3-category feature)
was flagged. Evidently tested it with a chi-square test and got p = 0.0345,
which is below the 0.05 flag threshold. The other 12 features were clearly not
flagged, with p-values between 0.14 (`oldpeak`) and 0.92 (`trestbps`).

This flag is statistically significant for that single test (p = 0.0345 < 0.05. 
However, it is weak evidence of real drift. Thirteen features were tested, so a 
multiple-testing correction (Bonferroni: 0.05 / 13 = 0.004) would not flag `slope`. 
In this project I also know the reference and current sets are a random split of one
dataset, so no real distribution shift exists. `slope` is a small categorical
feature, and the current set has only 91 rows, so a few rows in a different
category can trigger the test. Therefore, I treat it as a nominal flag that
most likely reflects chance, not as confirmed drift.

## 2. Would this drift likely affect model performance?

Probably not. One flagged feature out of 13 is a small, low-magnitude shift,
and the model is a random forest trained with imputation and one-hot encoding,
which tolerates small changes in a single categorical feature. Because the
flag is consistent with chance, there is no evidence that the input
distribution the model sees has actually changed.

This is a judgment from the drift report, not a measurement. Model performance
on the current data has not been evaluated separately, so the conclusion
should be confirmed by comparing metrics on both sets if the situation changes.

## 3. Recommended action: continue monitoring

- Retraining is not recommeded. Retraining on a chance flag would add cost with no
  expected benefit.
- Do a quick investigation of `slope` by opening the report and checking
  that the p-value is only marginally below 0.05 and that the category
  proportions differ only slightly.
- Keep the 40% share threshold. A single flagged feature is a note, and
  the gate is meant to catch broad shifts (about 6 or more of 13 features).

Escalate to investigation or retraining if any of these occur:

- the drift share exceeds the threshold, or rises across several consecutive checks;
- a feature drifts strongly (very small p-value, large change in distribution),
  especially a clinically important one such as `age`, `cp`, `thal` or `ca`;
- accuracy, F1 or AUC on newly labeled data falls below the thresholds in
  `configs/config.yaml`.

## Limitations

The reference and current sets come from the same file, so this check
demonstrates the monitoring pipeline and not real production drift. For
meaningful monitoring, the reference should be the training data and the
current set should be newly collected data. With about 90 current rows, the
tests have limited power, so small shifts can be missed and chance flags are
common.
