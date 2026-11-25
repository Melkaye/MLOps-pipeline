import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from skops.io import dumps, get_untrusted_types
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG_PATH = "configs/config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg):
    d = cfg["data"]
    df = pd.read_csv(d["path"])
    X = df[d["numeric"] + d["categorical"]]
    y = df[d["target"]]
    return X, y


def build_model(cfg):
    d, m = cfg["data"], cfg["model"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, d["numeric"]),
        ("cat", categorical_pipe, d["categorical"]),
    ])

    model_type = m["type"]
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=m["n_estimators"],
            max_depth=m["max_depth"],
            random_state=m["random_state"],
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=m["n_estimators"],
            max_depth=min(m["max_depth"], 5),
            random_state=m["random_state"],
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=m["random_state"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


def run_experiment(cfg):
    m = cfg["model"]

    # Tracking URI comes from the MLFLOW_TRACKING_URI env var if set,
    # then from the optional mlflow.tracking_uri key in the config.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    X, y = load_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=m["test_size"],
        random_state=m["random_state"],
        stratify=y,
    )

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": m["type"],
            "n_estimators": m["n_estimators"],
            "max_depth": m["max_depth"],
            "test_size": m["test_size"],
            "random_state": m["random_state"],
            "n_rows": len(X),
            "n_numeric": len(cfg["data"]["numeric"]),
            "n_categorical": len(cfg["data"]["categorical"]),
        })

        model = build_model(cfg)
        print(f"Training {m['type']}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, round(value, 4))

        # Trust only the types found inside this freshly trained model
        trusted_types = get_untrusted_types(data=dumps(model))
        mlflow.sklearn.log_model(
            model,
            "model",
            serialization_format="skops",
            skops_trusted_types=trusted_types,
        )

        # Keep a copy of the config with the run
        snapshot = "config_snapshot.json"
        with open(snapshot, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        mlflow.log_artifact(snapshot)
        os.remove(snapshot)

        print("=" * 50)
        for name, value in metrics.items():
            print(f"{name:<10} {value:.4f}")
        print("=" * 50)
        print(f"MLflow run ID: {run.info.run_id}")

        return run.info.run_id


if __name__ == "__main__":
    run_experiment(load_config())
