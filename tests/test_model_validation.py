from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from train import build_model   # src/train.py (src is on the path via pytest.ini)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    # Small, fast, deterministic model for testing
    config["model"].update({
        "type": "random_forest",
        "n_estimators": 50,
        "max_depth": 6,
        "random_state": 42,
    })
    return config


@pytest.fixture(scope="module")
def split(heart_df, cfg):
    d = cfg["data"]
    X = heart_df[d["numeric"] + d["categorical"]]
    y = heart_df[d["target"]]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@pytest.fixture(scope="module")
def trained_model(cfg, split):
    X_train, _, y_train, _ = split
    model = build_model(cfg)
    model.fit(X_train, y_train)
    return model


def test_predictions_have_correct_type_and_shape(trained_model, split):
    _, X_test, _, _ = split
    preds = trained_model.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(np.unique(preds)) <= {0, 1}


def test_predicted_probabilities_are_valid(trained_model, split):
    _, X_test, _, _ = split
    proba = trained_model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_meets_minimum_performance(trained_model, split):
    _, X_test, _, y_test = split
    preds = trained_model.predict(X_test)
    proba = trained_model.predict_proba(X_test)[:, 1]
    assert accuracy_score(y_test, preds) >= 0.70
    assert roc_auc_score(y_test, proba) >= 0.80


def test_model_handles_missing_values_at_prediction_time(trained_model, split, cfg):
    _, X_test, _, _ = split
    X_missing = X_test.copy()
    X_missing.iloc[0, X_missing.columns.get_loc("chol")] = np.nan   # numeric
    X_missing.iloc[0, X_missing.columns.get_loc("thal")] = np.nan   # categorical
    preds = trained_model.predict(X_missing)
    assert preds.shape == (len(X_missing),)


def test_training_is_reproducible(cfg, split):
    X_train, X_test, y_train, _ = split
    first = build_model(cfg).fit(X_train, y_train).predict_proba(X_test)
    second = build_model(cfg).fit(X_train, y_train).predict_proba(X_test)
    assert np.allclose(first, second)
