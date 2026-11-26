from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

# Real dataset first; committed sample as a fallback for machines without DVC data
DATA_CANDIDATES = [
    ROOT / "data" / "heart_disease.csv",
    ROOT / "tests" / "fixtures" / "heart_disease_sample.csv",
]


@pytest.fixture(scope="session")
def heart_df():
    """The actual heart disease dataset, loaded once per test session."""
    for path in DATA_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)
    pytest.fail(
        "Heart disease dataset not found. Expected one of: "
        + ", ".join(str(p) for p in DATA_CANDIDATES)
        + ". Run `python src/download_data.py`, or copy the CSV to "
        "tests/fixtures/heart_disease_sample.csv."
    )
