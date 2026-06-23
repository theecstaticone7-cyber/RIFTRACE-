"""Trains baseline illicit-transaction classifiers on the temporal train split
produced by data_pipeline/temporal_split.py.

Trains, in order: Logistic Regression, Random Forest, XGBoost. Class weighting
(class_weight='balanced' / scale_pos_weight) compensates for the ~2% illicit
vs ~21% licit imbalance in the labeled data.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
TRAIN_PATH = DATA_PROCESSED / "train.csv"

SAVED_DIR = Path(__file__).resolve().parent / "saved"

NON_FEATURE_COLUMNS = {"txId", "time_step", "label"}


def load_train_data() -> tuple[pd.DataFrame, pd.Series]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_PATH} not found. Run data_pipeline/temporal_split.py first."
        )
    train = pd.read_csv(TRAIN_PATH)
    feature_columns = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    return train[feature_columns], train["label"]


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=42
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    return models


def save_models(models: dict) -> None:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = SAVED_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"Saved {name} to {path}")


if __name__ == "__main__":
    X_train, y_train = load_train_data()
    models = train_models(X_train, y_train)
    save_models(models)
