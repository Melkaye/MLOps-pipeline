import pandas as pd
import numpy as np

# Heart disease (UCI) columns; mirrors configs/config.yaml
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"
REQUIRED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def _require_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")


def validate_dataframe(df, required_columns=REQUIRED_COLUMNS, target_column=TARGET_COLUMN):
    """Check that a dataframe meets basic requirements."""
    _require_dataframe(df)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if len(df) == 0:
        raise ValueError("Dataframe is empty")

    return True


def clean_data(df, numeric_columns=NUMERIC_COLUMNS, categorical_columns=CATEGORICAL_COLUMNS):
    """Clean a dataframe by handling missing values. Returns a new dataframe."""
    _require_dataframe(df)
    df = df.copy()

    # Fill numeric missing values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    return df


def encode_categoricals(df, columns=CATEGORICAL_COLUMNS):
    """One-hot encode categorical columns (first category dropped)."""
    _require_dataframe(df)
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        raise ValueError(f"Columns not found in dataframe: {unknown}")

    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
    return df


def check_data_quality(df, numeric_columns=NUMERIC_COLUMNS):
    """Return a dictionary of data quality metrics."""
    _require_dataframe(df)
    if len(df) == 0:
        raise ValueError("Dataframe is empty")

    report = {
        "total_rows": len(df),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in numeric_columns:
        if col in df.columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())

    return report
