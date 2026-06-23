"""Calibrates the Random Forest's probability outputs with
CalibratedClassifierCV, so the percentages RiftRace displays to analysts are
statistically meaningful, not just a confident ranking.

probability_distribution.py showed the served RF is well-ranked (test
ROC-AUC 0.94) but poorly calibrated: trees grown without depth limits
produce many unanimous (exactly 0/1) leaf votes, pushing most predictions
to the extremes regardless of true uncertainty.

Calibration must be fit on data the base estimator never saw during its own
training -- calibrating on training-seen rows would just measure how
confident the model is about data it already memorized, not its true
reliability. So this carves a calibration-only slice out of the *train*
split (steps 30-34, the same boundary already used for the GNN's
early-stopping validation slice in train_gnn.py) and fits the base RF on
the remainder (steps 1-29) only. The temporal test set (steps 35-49) is
never touched by calibration -- it's used only for the before/after
evaluation below, exactly like every other evaluation in this project.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
TRAIN_PATH = DATA_PROCESSED / "train.csv"
TEST_PATH = DATA_PROCESSED / "test.csv"
SAVED_DIR = Path(__file__).resolve().parent / "saved"

NON_FEATURE_COLUMNS = {"txId", "time_step", "label"}

FIT_TIME_STEPS = range(1, 30)  # steps 1-29: fits the base RF
CALIBRATION_TIME_STEPS = range(30, 35)  # steps 30-34: calibration-only, never seen by the base RF


def load_split():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_columns = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]

    fit_mask = train["time_step"].isin(FIT_TIME_STEPS)
    calib_mask = train["time_step"].isin(CALIBRATION_TIME_STEPS)

    X_fit, y_fit = train.loc[fit_mask, feature_columns], train.loc[fit_mask, "label"]
    X_calib, y_calib = train.loc[calib_mask, feature_columns], train.loc[calib_mask, "label"]
    X_test, y_test = test[feature_columns], test["label"]

    return X_fit, y_fit, X_calib, y_calib, X_test, y_test


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "name": name,
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_score),
        "brier_score": brier_score_loss(y_test, y_score),
    }
    return metrics, y_score


def print_metrics_table(rows) -> None:
    print(f"{'Model':<34}{'Precision':>11}{'Recall':>9}{'F1':>9}{'ROC-AUC':>10}{'Brier':>9}")
    for row in rows:
        print(
            f"{row['name']:<34}"
            f"{row['precision']:>11.4f}"
            f"{row['recall']:>9.4f}"
            f"{row['f1']:>9.4f}"
            f"{row['roc_auc']:>10.4f}"
            f"{row['brier_score']:>9.4f}"
        )


def print_distribution(y_score, name) -> None:
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999999, 1.0 + 1e-9]
    labels = [
        "[0.0-0.1)", "[0.1-0.2)", "[0.2-0.3)", "[0.3-0.4)", "[0.4-0.5)",
        "[0.5-0.6)", "[0.6-0.7)", "[0.7-0.8)", "[0.8-0.9)", "[0.9-1.0)", "exactly 1.0",
    ]
    counts, _ = np.histogram(y_score, bins=bins)
    print(f"--- {name} ---")
    for label, count in zip(labels, counts):
        pct = count / len(y_score) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:>14}: {count:>6,}  ({pct:5.2f}%)  {bar}")
    near_extreme = np.sum((y_score < 0.05) | (y_score > 0.95))
    print(
        f"  near-extreme (<=0.05 or >=0.95): {near_extreme:,} "
        f"({near_extreme / len(y_score) * 100:.2f}%)"
    )
    print()


def reliability_table(y_true, y_score, name, n_bins=10) -> None:
    """Bins predictions by predicted probability and compares the average
    predicted probability in each bin to the actual observed illicit rate
    -- a well-calibrated model has these two columns close together.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, bin_edges) - 1, 0, n_bins - 1)

    print(f"--- Reliability (predicted vs. actual rate): {name} ---")
    print(f"{'Bin':>14}{'n':>8}{'Predicted':>12}{'Actual':>10}")
    for b in range(n_bins):
        mask = bin_idx == b
        n = mask.sum()
        if n == 0:
            continue
        predicted_mean = y_score[mask].mean()
        actual_rate = np.asarray(y_true)[mask].mean()
        print(
            f"  [{bin_edges[b]:.1f}-{bin_edges[b + 1]:.1f})".rjust(14)
            + f"{n:>8,}{predicted_mean:>12.4f}{actual_rate:>10.4f}"
        )
    print()


if __name__ == "__main__":
    X_fit, y_fit, X_calib, y_calib, X_test, y_test = load_split()
    print(f"Fit set (steps 1-29):          {len(X_fit):,} rows")
    print(f"Calibration set (steps 30-34): {len(X_calib):,} rows (held out from base RF training)")
    print(f"Test set (steps 35-49):        {len(X_test):,} rows (untouched by calibration)")
    print()

    # --- Baseline: the currently-served RF, fit on ALL of steps 1-34 ---
    served_model = joblib.load(SAVED_DIR / "random_forest.joblib")
    served_metrics, served_scores = evaluate(served_model, X_test, y_test, "served (full train, uncalibrated)")

    # --- New base RF, fit on steps 1-29 only (calibration-clean) ---
    base_rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
    base_rf.fit(X_fit, y_fit)
    base_metrics, base_scores = evaluate(base_rf, X_test, y_test, "base RF (steps 1-29, uncalibrated)")

    # --- Calibrated variants, both fit on the held-out steps 30-34 slice ---
    # FrozenEstimator marks base_rf as already-fitted, so CalibratedClassifierCV
    # calibrates on exactly the data passed to .fit() below (X_calib/y_calib)
    # instead of re-fitting or internally cross-validating the base model.
    # (sklearn 1.6+ replaced the old cv="prefit" string option with this.)
    isotonic = CalibratedClassifierCV(estimator=FrozenEstimator(base_rf), method="isotonic")
    isotonic.fit(X_calib, y_calib)
    isotonic_metrics, isotonic_scores = evaluate(isotonic, X_test, y_test, "isotonic-calibrated")

    sigmoid = CalibratedClassifierCV(estimator=FrozenEstimator(base_rf), method="sigmoid")
    sigmoid.fit(X_calib, y_calib)
    sigmoid_metrics, sigmoid_scores = evaluate(sigmoid, X_test, y_test, "sigmoid-calibrated")

    print("=== Before / After: metrics on the temporal test set (steps 35-49) ===")
    print_metrics_table([served_metrics, base_metrics, isotonic_metrics, sigmoid_metrics])
    print()

    print("=== Probability distribution on the test set ===")
    print_distribution(served_scores, "served (full train, uncalibrated)")
    print_distribution(isotonic_scores, "isotonic-calibrated")
    print_distribution(sigmoid_scores, "sigmoid-calibrated")

    print("=== Reliability curves (10 bins) ===")
    reliability_table(y_test.to_numpy(), served_scores, "served (full train, uncalibrated)")
    reliability_table(y_test.to_numpy(), isotonic_scores, "isotonic-calibrated")
    reliability_table(y_test.to_numpy(), sigmoid_scores, "sigmoid-calibrated")

    # sigmoid wins: best Brier score, AUC unchanged vs its own base RF
    # (Platt scaling is rank-preserving), and a smooth reliability curve
    # across all 10 bins -- isotonic's curve has several empty bins, a sign
    # it overfit a coarse step function to the small (3,513-row) calibration
    # set. See the conversation writeup for the full reasoning.
    calibrated_path = SAVED_DIR / "random_forest_calibrated.joblib"
    joblib.dump(sigmoid, calibrated_path)
    print(f"Saved sigmoid-calibrated model to {calibrated_path}")
