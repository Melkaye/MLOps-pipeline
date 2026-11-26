"""Run Evidently data drift detection and fail if drift exceeds a threshold.

Exit codes: 0 = drift within threshold, 1 = drift exceeds threshold, 2 = bad input.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# Evidently 0.7+ moved the classic API to evidently.legacy; older versions use the original paths
try:
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report
except ImportError:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

CONFIG_PATH = "configs/config.yaml"
DEFAULT_THRESHOLD = 0.40


def load_config(path=CONFIG_PATH):
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def parse_args(cfg):
    drift_cfg = cfg.get("drift", {})
    parser = argparse.ArgumentParser(description="Evidently data drift check")
    parser.add_argument("--reference", default="data/reference.csv")
    parser.add_argument("--current", default="data/current.csv")
    parser.add_argument(
        "--threshold",
        type=float,
        default=drift_cfg.get("threshold", DEFAULT_THRESHOLD),
        help="Maximum allowed share of drifted features (0 to 1). "
             "Exit code 1 if the share exceeds it.",
    )
    parser.add_argument("--report", default="reports/drift_report.html")
    parser.add_argument("--summary", default="reports/drift_summary.json")
    return parser.parse_args()


def fail_input(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(2)


def load_inputs(args, cfg):
    for label, path in [("reference", args.reference), ("current", args.current)]:
        if not Path(path).exists():
            fail_input(f"{label} file not found: {path}")
    if not 0 <= args.threshold <= 1:
        fail_input(f"threshold must be between 0 and 1, got {args.threshold}")

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    if reference.empty or current.empty:
        fail_input("reference and current datasets must not be empty")

    data_cfg = cfg.get("data", {})
    numeric = data_cfg.get("numeric", [])
    categorical = data_cfg.get("categorical", [])
    if not (numeric or categorical):
        # No config: use every column except the target
        target = data_cfg.get("target", "target")
        shared = [c for c in reference.columns if c in current.columns and c != target]
        numeric = [c for c in shared if pd.api.types.is_numeric_dtype(reference[c])]
        categorical = [c for c in shared if c not in numeric]

    # Keep only features present in both datasets (target is never included)
    present = set(reference.columns) & set(current.columns)
    dropped = [c for c in numeric + categorical if c not in present]
    if dropped:
        print(f"WARNING: skipping columns missing from one dataset: {dropped}")
    numeric = [c for c in numeric if c in present]
    categorical = [c for c in categorical if c in present]
    features = numeric + categorical
    if not features:
        fail_input("no shared feature columns between reference and current data")

    return reference[features], current[features], numeric, categorical


def run_drift(reference, current, numeric, categorical, html_path):
    # Declare the types explicitly: categorical codes like cp or thal are stored as numbers
    mapping = ColumnMapping(numerical_features=numeric, categorical_features=categorical)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(html_path)
    return report


def extract_feature_results(report):
    result = report.as_dict()
    # Find the metric holding the per-column table instead of relying on its position
    table = next(
        (m["result"] for m in result["metrics"] if "drift_by_columns" in m.get("result", {})),
        None,
    )
    if table is None:
        found = [m.get("metric") for m in result["metrics"]]
        fail_input(f"No per-feature drift table in the report. Metrics found: {found}")

    rows = []
    for name, info in table["drift_by_columns"].items():
        rows.append({
            "feature": name,
            "drifted": bool(info["drift_detected"]),
            "test": info.get("stattest_name", ""),
            "score": info.get("drift_score"),
        })
    return rows


def print_summary(rows, share, threshold, ref_rows, cur_rows):
    drifted = [r for r in rows if r["drifted"]]
    print("=" * 70)
    print("DATA DRIFT REPORT")
    print("=" * 70)
    print(f"Reference rows: {ref_rows}   Current rows: {cur_rows}")
    print(f"Features checked: {len(rows)}")
    print(f"Features drifted: {len(drifted)}/{len(rows)}  (drift share {share:.1%})")
    print(f"Threshold:        {threshold:.1%}")
    print("-" * 70)
    print(f"{'feature':<12}{'drifted':<10}{'test':<28}{'score':<10}")
    for r in sorted(rows, key=lambda r: (not r["drifted"], r["feature"])):
        score = f"{r['score']:.4f}" if isinstance(r["score"], (int, float)) else str(r["score"])
        print(f"{r['feature']:<12}{'YES' if r['drifted'] else 'no':<10}{r['test']:<28}{score:<10}")
    print("-" * 70)
    if drifted:
        print("Drifted features: " + ", ".join(r["feature"] for r in drifted))
    else:
        print("Drifted features: none")


def main():
    cfg = load_config()
    args = parse_args(cfg)
    reference, current, numeric, categorical = load_inputs(args, cfg)

    report = run_drift(reference, current, numeric, categorical, args.report)
    rows = extract_feature_results(report)

    drifted = [r for r in rows if r["drifted"]]
    share = len(drifted) / len(rows)
    exceeded = share > args.threshold

    print_summary(rows, share, args.threshold, len(reference), len(current))

    summary = {
        "features_checked": len(rows),
        "features_drifted": len(drifted),
        "drift_share": round(share, 4),
        "threshold": args.threshold,
        "threshold_exceeded": exceeded,
        "drifted_features": [r["feature"] for r in drifted],
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"\nHTML report: {args.report}")
    print(f"Summary:     {args.summary}")

    if exceeded:
        print(f"\nFAIL: drift share {share:.1%} exceeds threshold {args.threshold:.1%}")
        sys.exit(1)
    print(f"\nOK: drift share {share:.1%} is within threshold {args.threshold:.1%}")


if __name__ == "__main__":
    main()
