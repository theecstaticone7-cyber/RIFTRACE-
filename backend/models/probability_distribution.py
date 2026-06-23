"""Diagnostic: inspects the Random Forest's illicit-probability distribution
across the full temporal test set, to check whether near-1.0 probabilities
on flagged transactions are genuine (all trees agree) vs. a display/rounding
artifact, and whether the model's output is suspiciously binary (a sign of
unconstrained trees with pure leaves, not necessarily generalization failure
-- see models/evaluate.py for held-out performance, which is what actually
indicates overfitting or not).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
TEST_PATH = DATA_PROCESSED / "test.csv"
MODEL_PATH = Path(__file__).resolve().parent / "saved" / "random_forest.joblib"

NON_FEATURE_COLUMNS = {"txId", "time_step", "label"}


def main() -> None:
    test = pd.read_csv(TEST_PATH)
    feature_columns = [c for c in test.columns if c not in NON_FEATURE_COLUMNS]
    X_test = test[feature_columns]

    model = joblib.load(MODEL_PATH)
    y_score = model.predict_proba(X_test)[:, 1]

    print("=== 1. Distribution of illicit probability across all test transactions ===")
    print(f"Total test transactions: {len(y_score):,}")
    print(f"min={y_score.min():.6f}  max={y_score.max():.6f}  "
          f"mean={y_score.mean():.4f}  median={np.median(y_score):.4f}  std={y_score.std():.4f}")
    print()

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999999, 1.0 + 1e-9]
    labels = ["[0.0-0.1)", "[0.1-0.2)", "[0.2-0.3)", "[0.3-0.4)", "[0.4-0.5)",
              "[0.5-0.6)", "[0.6-0.7)", "[0.7-0.8)", "[0.8-0.9)", "[0.9-1.0)",
              "exactly 1.0"]
    counts, _ = np.histogram(y_score, bins=bins)
    for label, count in zip(labels, counts):
        pct = count / len(y_score) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:>14}: {count:>6,}  ({pct:5.2f}%)  {bar}")
    print()

    exact_zero = np.sum(y_score == 0.0)
    exact_one = np.sum(y_score == 1.0)
    print(f"Exactly 0.0: {exact_zero:,} ({exact_zero / len(y_score) * 100:.2f}%)")
    print(f"Exactly 1.0: {exact_one:,} ({exact_one / len(y_score) * 100:.2f}%)")
    print(f"Strictly between 0 and 1: "
          f"{np.sum((y_score > 0) & (y_score < 1)):,} "
          f"({np.sum((y_score > 0) & (y_score < 1)) / len(y_score) * 100:.2f}%)")
    print()

    print("=== 2. Verifying a 'probability=1.0' case isn't a rounding artifact ===")
    top_idx = np.argmax(y_score)
    top_tx_id = int(test.iloc[top_idx]["txId"])
    top_row = X_test.iloc[[top_idx]]

    print(f"Transaction {top_tx_id}: ensemble predict_proba = {y_score[top_idx]!r} (full float precision)")

    # Ask each of the 100 trees individually, instead of trusting the
    # ensemble average -- this proves whether they actually all agree.
    # (.to_numpy(): individual tree estimators were fit without feature
    # names, so a named DataFrame here just triggers a harmless warning.)
    tree_probs = [tree.predict_proba(top_row.to_numpy())[0, 1] for tree in model.estimators_]
    tree_probs = np.array(tree_probs)
    print(f"Individual tree probabilities for this transaction (n={len(tree_probs)} trees):")
    print(f"  trees voting exactly 1.0 (fully illicit leaf): {np.sum(tree_probs == 1.0)}")
    print(f"  trees voting exactly 0.0: {np.sum(tree_probs == 0.0)}")
    print(f"  trees voting something else: {np.sum((tree_probs > 0) & (tree_probs < 1))}")
    print(f"  mean of the {len(tree_probs)} individual tree votes: {tree_probs.mean()!r}")
    print()

    print("=== 3. Sample transactions with MID-RANGE probability (0.6-0.8) ===")
    mid_mask = (y_score >= 0.6) & (y_score <= 0.8)
    mid_count = mid_mask.sum()
    print(f"Transactions with probability in [0.6, 0.8]: {mid_count}")
    if mid_count > 0:
        mid_examples = test.loc[mid_mask, ["txId", "label"]].copy()
        mid_examples["probability_illicit"] = y_score[mid_mask]
        mid_examples = mid_examples.sort_values("probability_illicit", ascending=False)
        for _, row in mid_examples.head(10).iterrows():
            true_label = "illicit" if row["label"] == 1 else "licit"
            print(f"  tx_id={int(row['txId']):>12}  "
                  f"probability={row['probability_illicit']:.4f}  true_label={true_label}")
    print()

    print("=== 4. Is the model suspiciously binary? ===")
    near_extreme = np.sum((y_score < 0.05) | (y_score > 0.95))
    print(f"Probabilities within 0.05 of an extreme (0 or 1): "
          f"{near_extreme:,} / {len(y_score):,} ({near_extreme / len(y_score) * 100:.2f}%)")
    print(f"Probabilities in the open middle (0.05-0.95): "
          f"{len(y_score) - near_extreme:,} ({(len(y_score) - near_extreme) / len(y_score) * 100:.2f}%)")


if __name__ == "__main__":
    main()
