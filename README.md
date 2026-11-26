## Heart Disease Prediction: MLOps Pipeline

An end-to-end MLOps pipeline that predicts heart disease from the UCI Heart
Disease dataset. It covers data versioning (DVC), data validation, automated
tests, experiment tracking (MLflow), a CI/CD workflow (GitHub Actions) that
gates on model performance, and data drift monitoring (Evidently).

## Overview

| Stage | What it does | Tool |
|---|---|---|
| Data versioning | Tracks the dataset by content hash | DVC + Git |
| Data validation | Checks columns, target values, missing values, duplicates | `src/validate_data.py` |
| Testing | Unit, data and model validation tests | pytest |
| Training | Trains a scikit-learn pipeline and logs runs | MLflow |
| Performance gate | Exits non-zero if metrics fall below thresholds | `src/train.py` |
| Drift monitoring | Compares reference and current data, fails above a threshold | Evidently |
| CI/CD | Runs everything on every push and pull request to `main` | GitHub Actions |

**Dataset:** UCI Heart Disease (Cleveland), 303 rows, 13 input features
(5 numeric, 8 categorical), binary target (`0` = no disease, `1` = disease),
with a few missing values in `ca` and `thal`.

## Project structure

```
.
├── .github/workflows/mlops-pipeline.yml   # CI/CD workflow
├── configs/config.yaml                    # data, model, MLflow, threshold and drift settings
├── data/                                  # heart_disease.csv (tracked by DVC, not in Git)
├── reports/                               # generated drift reports (not in Git)
├── src/
│   ├── download_data.py                   # fetch the dataset from UCI
│   ├── validate_data.py                   # data quality gate
│   ├── preprocessing.py                   # cleaning, encoding and quality helpers
│   ├── train.py                           # train, log to MLflow, check thresholds
│   ├── sweep.py                           # run multiple experiments
│   ├── analyze.py                         # rank runs by metric
│   ├── make_drift_data.py                 # build reference and current datasets
│   └── monitor_drift.py                   # Evidently drift detection
├── tests/
│   ├── conftest.py                        # loads the dataset once per session
│   ├── fixtures/heart_disease_sample.csv  # fallback copy for machines without DVC data
│   ├── test_preprocessing.py
│   ├── test_data_validation.py
│   ├── test_model_validation.py
│   └── test_thresholds.py
├── MONITORING.md                          # written drift analysis
├── pytest.ini
└── requirements.txt
```

## Setup

Requires Python 3.12.

```bash
git clone https://github.com/Melkaye/MLOps-pipeline.git
cd MLOps-pipeline

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Get the data with either option:

```bash
dvc pull                         # if you have access to the DVC remote
python src/download_data.py      # otherwise, download from UCI
```

If the download fails with `CERTIFICATE_VERIFY_FAILED` on macOS, run
`export SSL_CERT_FILE=$(python -m certifi)` first.

## Configuration

All settings live in `configs/config.yaml`.

| Section | Key | Meaning |
|---|---|---|
| `data` | `path`, `target`, `numeric`, `categorical` | Dataset location and column roles |
| `data` | `min_rows`, `max_missing_share` | Limits used by data validation |
| `mlflow` | `experiment`, `tracking_uri` | Experiment name and tracking server |
| `model` | `type`, `n_estimators`, `max_depth`, `test_size`, `random_state` | Model settings (`random_forest`, `gradient_boosting` or `logistic_regression`) |
| `thresholds` | `accuracy`, `f1_score`, `auc_roc` | Minimum metrics; training exits 1 if any is not met |
| `drift` | `threshold` | Maximum share of drifted features (default 0.40) |

`MLFLOW_TRACKING_URI`, if set, overrides `mlflow.tracking_uri`.

## Running the pipeline

Run everything from the project root with the virtual environment active.

```bash
# 1. Start the MLflow server (leave it running in its own terminal)
mkdir -p mlartifacts
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlartifacts \
  --host 127.0.0.1 --port 5000

# 2. In another terminal: validate, train, and check thresholds
python src/validate_data.py
python src/train.py

# 3. Compare experiments (optional)
python src/sweep.py
python src/analyze.py

# 4. Drift monitoring
python src/make_drift_data.py
python src/monitor_drift.py
```

Open http://127.0.0.1:5000 and use the **Runs** tab to view parameters,
metrics and models.

Exit codes: `validate_data.py` and `train.py` return `1` on a failed check.
`monitor_drift.py` returns `1` if the drift share exceeds the threshold and `2`
for bad input.

## Testing

```bash
python -m pytest tests/ -v
```

The suite covers:

- **Preprocessing:** missing value handling, encoding, no mutation of inputs, error handling.
- **Data validation:** expected columns, binary target, numeric feature ranges, allowed categorical values.
- **Model validation:** prediction type and shape, probability validity, minimum accuracy and AUC on a held-out split.
- **Threshold logic:** the performance gate used by `train.py`.

Tests use `data/heart_disease.csv` when present and otherwise fall back to
`tests/fixtures/heart_disease_sample.csv`, so they run on a fresh clone.

## CI/CD

`.github/workflows/mlops-pipeline.yml` runs on pushes to `main` and on pull
requests targeting `main`.

1. **`test`**: installs dependencies and runs the full pytest suite.
2. **`train`**: runs only if `test` passes. Downloads and validates the data, trains the model, and fails if a performance threshold is not met.
3. **`monitor-drift`**: runs only if `test` passes (in parallel with `train`). Runs Evidently drift detection, fails if the drift share exceeds the threshold, and uploads the HTML report as the `drift-reports` artifact.

To view a CI drift report, open the workflow run on GitHub, scroll to
**Artifacts**, download `drift-reports`, and unzip it.

## Data versioning

`data/heart_disease.csv` is tracked with DVC. Git stores only the small
`data/heart_disease.csv.dvc` pointer file.

```bash
dvc add data/heart_disease.csv     # after the dataset changes
git add data/heart_disease.csv.dvc
dvc push                           # upload to the DVC remote
```

`data/reference.csv`, `data/current.csv` and `reports/` are generated and
ignored by Git.

## Drift monitoring

`src/monitor_drift.py` compares a reference dataset against a current dataset
on all 13 features and saves an HTML report to `reports/drift_report.html` and
a JSON summary to `reports/drift_summary.json`. See [MONITORING.md](MONITORING.md)
for the analysis of the current results.

The reference and current files are currently a random 70/30 split of one
dataset, so the check demonstrates the monitoring pipeline and does not
measure real production drift.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` for a package | Activate `.venv` and run `python -m pip install -r requirements.txt` |
| `ImportError: infer_dtype_from_object` | Evidently needs pandas below 3: keep `pandas>=2.2,<3` pinned |
| `No module named 'preprocessing'` in pytest | Run `python -m pytest` from the project root with `pytest.ini` present |
| `Address already in use` on port 5000 | Another MLflow server is running; stop it or use `--port 5001` |
| `analyze.py` says the experiment is not found | Check that the experiment name and tracking URI match `configs/config.yaml` |
