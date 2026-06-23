"""Trains GCN and GraphSAGE models on the transaction graph.

Uses the same time-step philosophy as Phase 1's temporal split: steps 1-29
drive the actual gradient updates, steps 30-34 (carved out of the Phase 1
train split) are held out purely for early-stopping decisions, and steps
35-49 (the Phase 1 test split) stay untouched until evaluate.py. Picking the
validation slice by time step rather than randomly avoids leaking "future"
graph structure into the early-stopping decision, for the same reason the
train/test split itself is temporal rather than random.
"""

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from gnn_data_prep import load_or_build_data
from gnn_models import GCN, GraphSAGE

SAVED_DIR = Path(__file__).resolve().parent / "saved"

VAL_TIME_STEPS = range(30, 35)  # carved out of the train split (steps 1-34)

MAX_EPOCHS = 200
PATIENCE = 30
LOG_EVERY = 10
HIDDEN_CHANNELS = 64
LR = 0.01
WEIGHT_DECAY = 5e-4


def make_masks(data):
    val_mask = data.train_mask & torch.isin(
        data.time_step, torch.tensor(list(VAL_TIME_STEPS))
    )
    fit_mask = data.train_mask & ~val_mask
    return fit_mask, val_mask


def train_one_model(name, model, data, fit_mask, val_mask, pos_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[fit_mask], data.y[fit_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[val_mask]
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_f1 = f1_score(
                data.y[val_mask].numpy(), val_preds.numpy(), pos_label=1, zero_division=0
            )

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"[{name}] epoch {epoch:3d}  train_loss={loss.item():.4f}  "
                f"val_f1_illicit={val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(
                    f"[{name}] early stopping at epoch {epoch} "
                    f"(best val_f1_illicit={best_val_f1:.4f})"
                )
                break

    model.load_state_dict(best_state)
    return model, best_val_f1


def save_checkpoint(name, model, in_channels):
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVED_DIR / f"{name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": in_channels,
            "hidden_channels": HIDDEN_CHANNELS,
        },
        path,
    )
    print(f"Saved {name} to {path}")


if __name__ == "__main__":
    torch.manual_seed(42)

    data = load_or_build_data()
    fit_mask, val_mask = make_masks(data)

    neg = (data.y[fit_mask] == 0).sum()
    pos = (data.y[fit_mask] == 1).sum()
    pos_weight = neg / pos

    in_channels = data.x.shape[1]

    models = {
        "gcn": GCN(in_channels, HIDDEN_CHANNELS),
        "graphsage": GraphSAGE(in_channels, HIDDEN_CHANNELS),
    }

    for name, model in models.items():
        print(f"=== Training {name} ===")
        trained_model, best_val_f1 = train_one_model(
            name, model, data, fit_mask, val_mask, pos_weight
        )
        save_checkpoint(name, trained_model, in_channels)
