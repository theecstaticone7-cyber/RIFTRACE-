"""Precision/recall tradeoff for the served Random Forest model at different
decision thresholds, on the same temporal test split (steps 35-49) used
everywhere else. Default classifiers use 0.5; this shows what's gained/lost
by moving that cutoff for the illicit class specifically.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
TEST_PATH = DATA_PROCESSED / "test.csv"
MODEL_PATH = Path(__file__).resolve().parent / "saved" / "random_forest.joblib"

NON_FEATURE_COLUMNS = {"txId", "time_step", "label"}
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def main() -> None:
    test = pd.read_csv(TEST_PATH)
    feature_columns = [c for c in test.columns if c not in NON_FEATURE_COLUMNS]
    X_test, y_test = test[feature_columns], test["label"]

    model = joblib.load(MODEL_PATH)
    y_score = model.predict_proba(X_test)[:, 1]

    print(f"{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Flagged %':>12}")
    for t in THRESHOLDS:
        y_pred = (y_score >= t).astype(int)
        flagged_pct = y_pred.mean() * 100
        print(
            f"{t:>10.2f}"
            f"{precision_score(y_test, y_pred, pos_label=1, zero_division=0):>12.4f}"
            f"{recall_score(y_test, y_pred, pos_label=1, zero_division=0):>12.4f}"
            f"{f1_score(y_test, y_pred, pos_label=1, zero_division=0):>12.4f}"
            f"{flagged_pct:>11.2f}%"
        )


if __name__ == "__main__":
    main()
