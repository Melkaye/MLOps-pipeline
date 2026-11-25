import sys
import pandas as pd
import yaml

cfg = yaml.safe_load(open("configs/config.yaml"))["data"]
df = pd.read_csv(cfg["path"])
errors = []

expected = set(cfg["numeric"] + cfg["categorical"] + [cfg["target"]])
missing_cols = expected - set(df.columns)
if missing_cols:
    errors.append(f"Missing columns: {sorted(missing_cols)}")

if len(df) < cfg["min_rows"]:
    errors.append(f"Only {len(df)} rows (min {cfg['min_rows']})")

if not set(df[cfg["target"]].dropna().unique()) <= {0, 1}:
    errors.append("Target must contain only 0 and 1")

# Missing values are expected here (ca, thal), so only fail above a threshold
for col, share in df.isna().mean().items():
    if share > cfg["max_missing_share"]:
        errors.append(f"{col}: {share:.1%} missing")

if df.duplicated().sum() > 0.1 * len(df):
    errors.append("More than 10% duplicate rows")

if errors:
    print("DATA VALIDATION FAILED")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("Data validation passed")
