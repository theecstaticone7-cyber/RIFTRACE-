"""Splits labeled Elliptic transactions into train/test sets using the native
time_step column, NOT a random split.

The transaction graph evolves over time: edges in later time steps connect to
addresses/structures that don't exist yet in earlier steps. A random split
would let the model train on rows drawn from the same time steps it's tested
on, leaking "future" graph structure and feature patterns into training that
wouldn't be available at real deployment time (where you only ever have data
up to the present). Splitting strictly on time_step instead trains on steps
1-34 and evaluates on steps 35-49, mirroring how the model is actually used:
trained on the past, tested on the future it has never seen.

"Unknown" labeled nodes are excluded from both sets since there's no ground
truth to train or evaluate against.
"""

from pathlib import Path

import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

FEATURES_PATH = DATA_RAW / "elliptic_txs_features.csv"
CLASSES_PATH = DATA_RAW / "elliptic_txs_classes.csv"

TRAIN_PATH = DATA_PROCESSED / "train.csv"
TEST_PATH = DATA_PROCESSED / "test.csv"

TRAIN_TIME_STEPS = range(1, 35)  # steps 1-34
TEST_TIME_STEPS = range(35, 50)  # steps 35-49

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 166)]
COLUMNS = ["txId", "time_step"] + FEATURE_COLUMNS


def load_labeled_data() -> pd.DataFrame:
    features = pd.read_csv(FEATURES_PATH, header=None)
    features.columns = COLUMNS

    classes = pd.read_csv(CLASSES_PATH)
    merged = features.merge(classes, on="txId")

    merged = merged[merged["class"] != "unknown"].copy()
    merged["label"] = (merged["class"] == "1").astype(int)  # 1 = illicit, 0 = licit
    return merged.drop(columns=["class"])


def split(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = merged[merged["time_step"].isin(TRAIN_TIME_STEPS)]
    test = merged[merged["time_step"].isin(TEST_TIME_STEPS)]
    return train, test


def save_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)


if __name__ == "__main__":
    merged = load_labeled_data()
    train, test = split(merged)
    save_split(train, test)

    print(f"Labeled nodes (excl. unknown): {len(merged):,}")
    print(
        f"Train set (time steps 1-34):  {len(train):,} nodes "
        f"({train['label'].mean() * 100:.2f}% illicit)"
    )
    print(
        f"Test set  (time steps 35-49): {len(test):,} nodes "
        f"({test['label'].mean() * 100:.2f}% illicit)"
    )
    print(f"Saved to {TRAIN_PATH} and {TEST_PATH}")
