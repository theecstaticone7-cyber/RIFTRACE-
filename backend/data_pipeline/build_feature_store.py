"""One-time conversion of the raw features CSV into a compact SQLite
key-value store (tx_id -> packed float32 feature vector).

The API used to load the entire ~272MB feature DataFrame into memory just to
support per-tx_id lookups for /predict. That's resident for the life of the
process regardless of how many transactions are actually queried. Storing
vectors in SQLite instead lets the API do a single indexed row lookup per
request and keep steady-state memory flat, independent of dataset size.

Run once after refreshing the raw dataset:
    python data_pipeline/build_feature_store.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

FEATURES_PATH = DATA_RAW / "elliptic_txs_features.csv"
STORE_PATH = DATA_PROCESSED / "feature_store.sqlite"
META_PATH = DATA_PROCESSED / "dataset_meta.json"

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 166)]
COLUMNS = ["txId", "time_step"] + FEATURE_COLUMNS


def build() -> None:
    features = pd.read_csv(FEATURES_PATH, header=None)
    features.columns = COLUMNS

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    STORE_PATH.unlink(missing_ok=True)

    conn = sqlite3.connect(STORE_PATH)
    conn.execute("CREATE TABLE features (tx_id INTEGER PRIMARY KEY, vector BLOB NOT NULL)")

    # float32 halves the per-vector size vs. the float64 the raw CSV implies;
    # tree-based models split on thresholds that tolerate this fine (verified
    # separately by comparing predictions before/after).
    vectors = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    rows = (
        (int(tx_id), vector.tobytes())
        for tx_id, vector in zip(features["txId"], vectors)
    )
    conn.executemany("INSERT INTO features VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

    meta = {
        "total_nodes": len(features),
        "num_time_steps": int(features["time_step"].nunique()),
        "feature_count": len(FEATURE_COLUMNS),
    }
    META_PATH.write_text(json.dumps(meta))

    print(f"Wrote {len(features):,} feature vectors to {STORE_PATH}")
    print(f"({STORE_PATH.stat().st_size / 1e6:.1f} MB on disk)")
    print(f"Wrote dataset metadata to {META_PATH}: {meta}")


if __name__ == "__main__":
    build()
