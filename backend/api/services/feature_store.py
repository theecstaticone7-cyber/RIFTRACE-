"""Lazy, per-request feature-vector lookups backed by SQLite instead of a
fully materialized in-memory DataFrame.

Each call opens a short-lived connection and does a single indexed point
lookup by primary key. SQLite connections are cheap to open and the OS page
cache handles repeated reads efficiently, so this keeps steady-state API
memory flat regardless of dataset size -- unlike holding a ~272MB pandas
DataFrame resident for the life of the process.
"""

import sqlite3
from pathlib import Path

import numpy as np

DATA_PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
STORE_PATH = DATA_PROCESSED / "feature_store.sqlite"

FEATURE_DIM = 165


def get_feature_vector(tx_id: int) -> np.ndarray | None:
    conn = sqlite3.connect(STORE_PATH)
    try:
        row = conn.execute("SELECT vector FROM features WHERE tx_id = ?", (tx_id,)).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)
