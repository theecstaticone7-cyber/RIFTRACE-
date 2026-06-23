"""Converts the cached transaction graph and node features into a PyTorch
Geometric Data object (x, edge_index, y) for GNN training.

Reuses the same temporal train/test split as Phase 1 (steps 1-34 train,
35-49 test). Unknown-labeled nodes are kept in the graph and only excluded
via train_mask/test_mask, rather than removed outright, since GNNs need the
full neighbor structure intact to propagate messages through and across
unlabeled nodes.

Edges are made bidirectional for message passing: the raw edgelist encodes
fund-flow direction, but GCN/SAGE aggregation benefits from a risk signal
propagating to both a transaction's senders and its receivers during
training, not just downstream.
"""

import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

FEATURES_PATH = DATA_RAW / "elliptic_txs_features.csv"
CLASSES_PATH = DATA_RAW / "elliptic_txs_classes.csv"
GRAPH_PATH = DATA_PROCESSED / "transaction_graph.gpickle"
GNN_DATA_PATH = DATA_PROCESSED / "gnn_data.pt"

TRAIN_TIME_STEPS = range(1, 35)  # steps 1-34
TEST_TIME_STEPS = range(35, 50)  # steps 35-49

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 166)]
COLUMNS = ["txId", "time_step"] + FEATURE_COLUMNS


def build_data() -> Data:
    features = pd.read_csv(FEATURES_PATH, header=None)
    features.columns = COLUMNS

    classes = pd.read_csv(CLASSES_PATH)
    merged = features.merge(classes, on="txId", how="left")

    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    # build_graph.py adds nodes in features-file order, so this re-indexing
    # keeps `merged` aligned with the node order used for edge_index below.
    node_order = list(graph.nodes())
    node_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(node_order)}
    merged = merged.set_index("txId").loc[node_order].reset_index()

    edges = list(graph.edges())
    src = [node_id_to_idx[a] for a, b in edges]
    dst = [node_id_to_idx[b] for a, b in edges]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    time_step = torch.tensor(merged["time_step"].values, dtype=torch.long)
    is_illicit = merged["class"] == "1"
    is_licit = merged["class"] == "2"
    labeled = is_illicit | is_licit

    y = torch.zeros(len(merged), dtype=torch.float)
    y[is_illicit.to_numpy(copy=True)] = 1.0

    train_mask = torch.tensor(
        (labeled & merged["time_step"].isin(TRAIN_TIME_STEPS)).values
    )
    test_mask = torch.tensor(
        (labeled & merged["time_step"].isin(TEST_TIME_STEPS)).values
    )

    # Fit scaling on the train split only, then apply to all nodes (incl.
    # unknown/test) to avoid leaking test statistics into the features.
    x_raw = merged[FEATURE_COLUMNS].values
    scaler = StandardScaler().fit(x_raw[train_mask.numpy()])
    x = torch.tensor(scaler.transform(x_raw), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        time_step=time_step,
    )


def load_or_build_data(path: Path = GNN_DATA_PATH) -> Data:
    if path.exists():
        return torch.load(path, weights_only=False)
    data = build_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


if __name__ == "__main__":
    data = load_or_build_data()
    print(f"Node features: {tuple(data.x.shape)}")
    print(f"Edge index:    {tuple(data.edge_index.shape)} (bidirectional)")
    print(
        f"Train nodes:   {int(data.train_mask.sum()):,} "
        f"({data.y[data.train_mask].mean() * 100:.2f}% illicit)"
    )
    print(
        f"Test nodes:    {int(data.test_mask.sum()):,} "
        f"({data.y[data.test_mask].mean() * 100:.2f}% illicit)"
    )
