"""Loads the raw Elliptic dataset and prints summary statistics.

Expects elliptic_txs_features.csv, elliptic_txs_classes.csv, and
elliptic_txs_edgelist.csv in backend/data/raw/ (see data_pipeline/README.md).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FEATURES_PATH = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_PATH = DATA_DIR / "elliptic_txs_classes.csv"
EDGELIST_PATH = DATA_DIR / "elliptic_txs_edgelist.csv"


def load_dataset():
    for path in (FEATURES_PATH, CLASSES_PATH, EDGELIST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {path}\n"
                "See backend/data_pipeline/README.md for expected file structure."
            )

    features = pd.read_csv(FEATURES_PATH, header=None)
    classes = pd.read_csv(CLASSES_PATH)
    edges = pd.read_csv(EDGELIST_PATH)

    return features, classes, edges


def inspect(features: pd.DataFrame, classes: pd.DataFrame, edges: pd.DataFrame):
    node_count = len(features)
    edge_count = len(edges)
    time_steps = features.iloc[:, 1].nunique()
    feature_count = features.shape[1] - 2  # exclude txId and time_step columns

    class_counts = classes["class"].value_counts()
    total = class_counts.sum()
    illicit_pct = class_counts.get("1", 0) / total * 100
    licit_pct = class_counts.get("2", 0) / total * 100
    unknown_pct = class_counts.get("unknown", 0) / total * 100

    print("=== Elliptic Dataset Summary ===")
    print(f"Total node count:        {node_count:,}")
    print(f"Total edge count:        {edge_count:,}")
    print(f"Number of time steps:    {time_steps}")
    print(f"Feature count per node:  {feature_count}")
    print("Class balance:")
    print(f"  Illicit:  {illicit_pct:.2f}%")
    print(f"  Licit:    {licit_pct:.2f}%")
    print(f"  Unknown:  {unknown_pct:.2f}%")


if __name__ == "__main__":
    features_df, classes_df, edges_df = load_dataset()
    inspect(features_df, classes_df, edges_df)
