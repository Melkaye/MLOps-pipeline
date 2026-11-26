import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    check_data_quality,
    clean_data,
    encode_categoricals,
    validate_dataframe,
)


@pytest.fixture
def sample_df():
    """First rows of heart_disease.csv (no missing values)."""
    return pd.DataFrame({
        "age": [63, 67, 67, 37, 41, 56],
        "sex": [1, 1, 1, 1, 0, 1],
        "cp": [1, 4, 4, 3, 2, 2],
        "trestbps": [145, 160, 120, 130, 130, 120],
        "chol": [233, 286, 229, 250, 204, 236],
        "fbs": [1, 0, 0, 0, 0, 0],
        "restecg": [2, 2, 2, 0, 2, 0],
        "thalach": [150, 108, 129, 187, 172, 178],
        "exang": [0, 1, 1, 0, 0, 0],
        "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4, 0.8],
        "slope": [3, 2, 2, 3, 1, 1],
        "ca": [0.0, 3.0, 2.0, 0.0, 0.0, 0.0],
        "thal": [6.0, 3.0, 7.0, 3.0, 3.0, 3.0],
        "target": [0, 1, 1, 0, 0, 0],
    })


@pytest.fixture
def df_with_missing(sample_df):
    """Same data with one NaN in a numeric column and two in categorical columns."""
    df = sample_df.copy()
    df["chol"] = df["chol"].astype(float)
    df.loc[2, "chol"] = np.nan   # remaining values -> median 236
    df.loc[1, "ca"] = np.nan     # remaining mode is 0.0
    df.loc[3, "thal"] = np.nan   # remaining mode is 3.0
    return df


# ---------- validate_dataframe ----------

def test_validate_accepts_good_dataframe(sample_df):
    assert validate_dataframe(sample_df) is True


def test_validate_missing_required_column_raises(sample_df):
    with pytest.raises(ValueError, match="chol"):
        validate_dataframe(sample_df.drop(columns=["chol"]))


def test_validate_missing_target_raises(sample_df):
    with pytest.raises(ValueError, match="Target column"):
        validate_dataframe(sample_df.drop(columns=["target"]))


def test_validate_empty_dataframe_raises(sample_df):
    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(sample_df.iloc[0:0])


def test_validate_non_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        validate_dataframe([[1, 2, 3]])


# ---------- clean_data: missing values ----------

def test_clean_fills_numeric_with_median(df_with_missing):
    result = clean_data(df_with_missing)
    assert result.loc[2, "chol"] == 236.0


def test_clean_fills_categorical_with_mode(df_with_missing):
    result = clean_data(df_with_missing)
    assert result.loc[1, "ca"] == 0.0
    assert result.loc[3, "thal"] == 3.0


def test_clean_leaves_no_missing_values(df_with_missing):
    result = clean_data(df_with_missing)
    assert result[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].isna().sum().sum() == 0


def test_clean_does_not_modify_input(df_with_missing):
    original = df_with_missing.copy()
    clean_data(df_with_missing)
    pd.testing.assert_frame_equal(df_with_missing, original)


def test_clean_non_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        clean_data({"age": [1, 2]})


# ---------- encode_categoricals ----------

def test_encode_drops_first_category(sample_df):
    result = encode_categoricals(sample_df, ["sex", "cp"])
    assert {"sex_1", "cp_2", "cp_3", "cp_4"} <= set(result.columns)
    assert "sex_0" not in result.columns
    assert "cp_1" not in result.columns   # first category dropped


def test_encode_removes_original_columns(sample_df):
    result = encode_categoricals(sample_df, ["sex", "cp"])
    assert "sex" not in result.columns
    assert "cp" not in result.columns


def test_encode_values_are_binary(sample_df):
    result = encode_categoricals(sample_df, ["cp"])
    cp_cols = [c for c in result.columns if c.startswith("cp_")]
    assert set(np.unique(result[cp_cols].to_numpy())) <= {0, 1}


def test_encode_keeps_rows_and_other_columns(sample_df):
    result = encode_categoricals(sample_df, ["sex", "cp"])
    assert len(result) == len(sample_df)
    pd.testing.assert_series_equal(result["age"], sample_df["age"])
    pd.testing.assert_series_equal(result["target"], sample_df["target"])


def test_encode_does_not_modify_input(sample_df):
    original = sample_df.copy()
    encode_categoricals(sample_df, CATEGORICAL_COLUMNS)
    pd.testing.assert_frame_equal(sample_df, original)


def test_encode_unknown_column_raises(sample_df):
    with pytest.raises(ValueError, match="not_a_column"):
        encode_categoricals(sample_df, ["not_a_column"])


# ---------- check_data_quality ----------

def test_quality_report_counts_nulls(df_with_missing):
    report = check_data_quality(df_with_missing)
    assert report["total_rows"] == 6
    assert report["total_nulls"] == 3
    assert report["null_percentage"] == 3.57          # 3 / (6 rows * 14 cols)


def test_quality_report_counts_duplicates(sample_df):
    df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    assert check_data_quality(df)["duplicate_rows"] == 1


def test_quality_report_min_max(sample_df):
    report = check_data_quality(sample_df)
    assert report["age_min"] == 37.0
    assert report["age_max"] == 67.0
    assert report["oldpeak_min"] == 0.8
    assert report["oldpeak_max"] == 3.5


def test_quality_min_max_ignore_nulls(df_with_missing):
    report = check_data_quality(df_with_missing)
    assert report["chol_min"] == 204.0     # the NaN row is skipped
    assert report["chol_max"] == 286.0


def test_quality_skips_unknown_numeric_columns(sample_df):
    report = check_data_quality(sample_df, ["age", "not_a_column"])
    assert "age_min" in report
    assert "not_a_column_min" not in report


def test_quality_empty_dataframe_raises(sample_df):
    with pytest.raises(ValueError, match="empty"):
        check_data_quality(sample_df.iloc[0:0])
