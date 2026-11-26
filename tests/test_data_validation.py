import pandas as pd
import pytest

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Generous clinical bounds (the real data sits well inside these)
NUMERIC_RANGES = {
    "age": (18, 100),
    "trestbps": (80, 220),     # resting blood pressure
    "chol": (100, 600),        # serum cholesterol
    "thalach": (60, 220),      # max heart rate
    "oldpeak": (0, 7),         # ST depression
}

ALLOWED_CATEGORICAL_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def test_expected_columns_present(heart_df):
    missing = set(EXPECTED_COLUMNS) - set(heart_df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"


def test_no_unexpected_columns(heart_df):
    extra = set(heart_df.columns) - set(EXPECTED_COLUMNS)
    assert not extra, f"Unexpected columns: {sorted(extra)}"


def test_dataset_has_enough_rows(heart_df):
    assert len(heart_df) >= 250


def test_target_contains_only_binary_values(heart_df):
    assert set(heart_df["target"].unique()) <= {0, 1}


def test_target_has_both_classes_and_is_not_extremely_imbalanced(heart_df):
    share_positive = heart_df["target"].mean()
    assert 0.30 <= share_positive <= 0.70


def test_target_has_no_missing_values(heart_df):
    assert heart_df["target"].isna().sum() == 0


@pytest.mark.parametrize("column", NUMERIC_COLUMNS)
def test_numeric_columns_have_numeric_dtype(heart_df, column):
    assert pd.api.types.is_numeric_dtype(heart_df[column])


@pytest.mark.parametrize("column,bounds", NUMERIC_RANGES.items())
def test_numeric_features_within_expected_range(heart_df, column, bounds):
    low, high = bounds
    values = heart_df[column].dropna()
    assert values.min() >= low, f"{column} min {values.min()} below {low}"
    assert values.max() <= high, f"{column} max {values.max()} above {high}"


@pytest.mark.parametrize("column,allowed", ALLOWED_CATEGORICAL_VALUES.items())
def test_categorical_features_have_expected_values(heart_df, column, allowed):
    observed = set(heart_df[column].dropna().unique())
    assert observed <= allowed, f"{column} has unexpected values: {observed - allowed}"


def test_missing_values_are_limited(heart_df):
    share_missing = heart_df.isna().mean()
    too_many = share_missing[share_missing > 0.05]
    assert too_many.empty, f"Columns with >5% missing: {too_many.to_dict()}"


def test_duplicate_rows_are_limited(heart_df):
    assert heart_df.duplicated().sum() <= 0.10 * len(heart_df)
