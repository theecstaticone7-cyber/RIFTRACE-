"""Evaluates the saved baseline models AND the saved GNN checkpoints on the
temporal test split, reporting precision, recall, F1, and ROC-AUC for the
illicit class specifically, plus overall accuracy for reference.

Overall accuracy is misleading on its own here: ~93% of the test set is
licit, so a model that always predicts "licit" would score ~93% accuracy
while catching zero illicit transactions. It's included as an extra column
for context, not as the metric to optimize for.
"""

from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gnn_data_prep import load_or_build_data
from gnn_models import GCN, GraphSAGE

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
TEST_PATH = DATA_PROCESSED / "test.csv"

SAVED_DIR = Path(__file__).resolve().parent / "saved"

NON_FEATURE_COLUMNS = {"txId", "time_step", "label"}
BASELINE_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
GNN_MODEL_CLASSES = {"gcn": GCN, "graphsage": GraphSAGE}


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_PATH} not found. Run data_pipeline/temporal_split.py first."
        )
    test = pd.read_csv(TEST_PATH)
    feature_columns = [c for c in test.columns if c not in NON_FEATURE_COLUMNS]
    return test[feature_columns], test["label"]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    return {
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_score),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def evaluate_gnn_model(model, data, mask) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[mask]
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    y_true = data.y[mask].numpy()
    y_pred = preds.numpy()
    y_score = probs.numpy()

    return {
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def print_comparison(results: dict) -> None:
    print(
        f"{'Model':<22}{'Precision':>12}{'Recall':>12}{'F1':>12}"
        f"{'ROC-AUC':>12}{'Accuracy':>12}"
    )
    for name, metrics in results.items():
        print(
            f"{name:<22}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}"
            f"{metrics['roc_auc']:>12.4f}"
            f"{metrics['accuracy']:>12.4f}"
        )


if __name__ == "__main__":
    X_test, y_test = load_test_data()

    results = {}
    for name in BASELINE_MODEL_NAMES:
        model_path = SAVED_DIR / f"{name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run models/baseline_models.py first."
            )
        model = joblib.load(model_path)
        results[name] = evaluate_model(model, X_test, y_test)

    gnn_data = load_or_build_data()
    for name, model_cls in GNN_MODEL_CLASSES.items():
        ckpt_path = SAVED_DIR / f"{name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"{ckpt_path} not found. Run models/train_gnn.py first."
            )
        checkpoint = torch.load(ckpt_path, weights_only=False)
        model = model_cls(checkpoint["in_channels"], checkpoint["hidden_channels"])
        model.load_state_dict(checkpoint["model_state_dict"])
        results[name] = evaluate_gnn_model(model, gnn_data, gnn_data.test_mask)

    print("=== Illicit-Class Evaluation on Test Set (time steps 35-49) ===")
    print_comparison(results)
