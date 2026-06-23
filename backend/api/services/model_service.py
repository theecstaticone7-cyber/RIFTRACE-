"""Loads the trained Random Forest model once, and serves predictions and
metrics.

Feature vectors are looked up per-request from a SQLite-backed store
(feature_store.py) instead of a fully materialized in-memory DataFrame, to
keep the API's memory footprint flat regardless of dataset size -- this used
to hold a ~272MB DataFrame resident for the life of the process just to
support tx_id lookups.

Random Forest is the best-performing model from the Phase 1/Phase 2
evaluation (F1=0.81, ROC-AUC=0.94 on the temporal test split), so it's what
/predict and /stats serve. Swapping the served model later only means
changing MODEL_PATH/MODEL_NAME here, not the router code.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import feature_store

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW = BACKEND_DIR / "data" / "raw"
DATA_PROCESSED = BACKEND_DIR / "data" / "processed"
SAVED_DIR = BACKEND_DIR / "models" / "saved"

CLASSES_PATH = DATA_RAW / "elliptic_txs_classes.csv"
EDGELIST_PATH = DATA_RAW / "elliptic_txs_edgelist.csv"
TEST_PATH = DATA_PROCESSED / "test.csv"
MODEL_PATH = SAVED_DIR / "random_forest.joblib"
DATASET_META_PATH = DATA_PROCESSED / "dataset_meta.json"
FEATURE_STORE_PATH = feature_store.STORE_PATH

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 166)]
MODEL_NAME = "random_forest"
CLASS_LABELS = {"1": "illicit", "2": "licit"}

# Files this service needs present at startup -- checked by api/main.py's
# fail-fast validation before the app is allowed to accept traffic.
REQUIRED_FILES = [
    MODEL_PATH,
    CLASSES_PATH,
    EDGELIST_PATH,
    DATASET_META_PATH,
    FEATURE_STORE_PATH,
    TEST_PATH,
]

# Populated on first use by _load() (module-level cache).
_model = None
_classes_by_tx_id = None
_edge_count = None
_dataset_meta = None

# Test-set predictions, computed once and shared by get_model_metrics() and
# get_flagged_transactions() instead of each re-reading test.csv and
# re-running predict() independently.
_test_predictions = None  # list of {tx_id, label, y_pred, probability_illicit}
_flagged_cache = None  # _test_predictions filtered to predicted-illicit, sorted desc


def _load() -> None:
    global _model, _classes_by_tx_id, _edge_count, _dataset_meta
    if _model is not None:
        return

    _model = joblib.load(MODEL_PATH)

    classes = pd.read_csv(CLASSES_PATH)
    _classes_by_tx_id = classes.set_index("txId")

    _edge_count = len(pd.read_csv(EDGELIST_PATH))
    _dataset_meta = json.loads(DATASET_META_PATH.read_text())


def _compute_test_predictions() -> None:
    """Runs the model once over the full temporal test set (steps 35-49)
    and caches per-row results. Both /stats and /flagged read from this
    instead of re-loading test.csv and re-calling predict() on every request.
    """
    global _test_predictions, _flagged_cache
    if _test_predictions is not None:
        return

    _load()
    test = pd.read_csv(TEST_PATH)
    X_test = test[FEATURE_COLUMNS]
    y_score = _model.predict_proba(X_test)[:, 1]

    _test_predictions = [
        {
            "tx_id": int(tx_id),
            "label": int(label),
            "probability_illicit": float(prob),
            "predicted_illicit": prob >= 0.5,
        }
        for tx_id, label, prob in zip(test["txId"], test["label"], y_score)
    ]

    flagged = [
        {
            "tx_id": row["tx_id"],
            "probability_illicit": row["probability_illicit"],
            "known_class": "illicit" if row["label"] == 1 else "licit",
        }
        for row in _test_predictions
        if row["predicted_illicit"]
    ]
    flagged.sort(key=lambda row: row["probability_illicit"], reverse=True)
    _flagged_cache = flagged


def warm_up() -> None:
    """Eagerly loads everything this service needs, including running the
    model once over the test set to populate the /flagged and /stats
    caches. Called at app startup (not lazily on first request) so a
    broken/missing artifact fails the deploy immediately, and so /flagged
    never recomputes predictions per-request.
    """
    _load()
    _compute_test_predictions()


def get_classes_df() -> pd.DataFrame:
    """Shared, cached classes lookup. graph_service uses this too, so the
    ~13MB classes.csv is only loaded into memory once across both services.
    """
    _load()
    return _classes_by_tx_id


def get_class_label(tx_id: int) -> str:
    """Maps the raw Elliptic class value ("1"/"2"/"unknown") to a label."""
    classes = get_classes_df()
    if tx_id not in classes.index:
        return "unknown"
    raw = classes.loc[tx_id, "class"]
    return CLASS_LABELS.get(raw, "unknown")


def predict_transaction(tx_id: int) -> dict:
    _load()
    vector = feature_store.get_feature_vector(tx_id)
    if vector is None:
        raise KeyError(f"Transaction {tx_id} not found in dataset")

    # Wrapped in a DataFrame with the training-time column names so sklearn
    # doesn't warn about missing feature names on a bare ndarray.
    row = pd.DataFrame(vector.reshape(1, -1), columns=FEATURE_COLUMNS)
    probability_illicit = float(_model.predict_proba(row)[0, 1])
    prediction = "illicit" if probability_illicit >= 0.5 else "licit"

    return {
        "tx_id": tx_id,
        "prediction": prediction,
        "probability_illicit": probability_illicit,
    }


def get_model_metrics() -> dict:
    """Illicit-class precision/recall/F1/ROC-AUC + accuracy on the temporal
    test split (steps 35-49) -- the same split used throughout the project,
    never randomly resampled. Reads from the cached test-set predictions
    (see _compute_test_predictions) instead of re-running the model.
    """
    _compute_test_predictions()
    y_test = [row["label"] for row in _test_predictions]
    y_pred = [int(row["predicted_illicit"]) for row in _test_predictions]
    y_score = [row["probability_illicit"] for row in _test_predictions]

    return {
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_score),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def get_flagged_transactions(limit: int) -> list[dict]:
    """Top `limit` test-set transactions predicted illicit, sorted by
    probability descending. Served from the cache populated by warm_up() --
    never recomputes predictions per-request.
    """
    _compute_test_predictions()
    return _flagged_cache[:limit]


def get_dataset_stats() -> dict:
    _load()
    class_counts = _classes_by_tx_id["class"].value_counts()
    total = class_counts.sum()

    return {
        "total_nodes": _dataset_meta["total_nodes"],
        "total_edges": _edge_count,
        "num_time_steps": _dataset_meta["num_time_steps"],
        "feature_count": _dataset_meta["feature_count"],
        "pct_licit": float(class_counts.get("2", 0) / total * 100),
        "pct_illicit": float(class_counts.get("1", 0) / total * 100),
        "pct_unknown": float(class_counts.get("unknown", 0) / total * 100),
    }


def health_check() -> dict:
    """Confirms the model is actually loaded AND can run inference, not
    just that the process is up. Uses a synthetic zero vector so it doesn't
    depend on any specific tx_id existing in the feature store.
    """
    _load()
    dummy = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32), columns=FEATURE_COLUMNS)
    _model.predict_proba(dummy)
    return {"model_loaded": True, "model_name": MODEL_NAME}
