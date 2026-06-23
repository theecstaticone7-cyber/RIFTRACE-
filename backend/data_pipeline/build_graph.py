"""Builds a directed transaction graph from the Elliptic edgelist and caches
it to disk so later phases don't need to rebuild it every run.

NetworkX removed write_gpickle/read_gpickle in 3.0, so we serialize the graph
with the standard pickle module directly (file keeps the .gpickle name/intent).
"""

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

FEATURES_PATH = DATA_RAW / "elliptic_txs_features.csv"
EDGELIST_PATH = DATA_RAW / "elliptic_txs_edgelist.csv"
GRAPH_PATH = DATA_PROCESSED / "transaction_graph.gpickle"


def build_graph() -> nx.DiGraph:
    # Include every node from the features file, not just ones that appear in
    # an edge, so isolated transactions aren't silently dropped from the graph.
    tx_ids = pd.read_csv(FEATURES_PATH, header=None, usecols=[0])[0]
    edges = pd.read_csv(EDGELIST_PATH)

    graph = nx.DiGraph()
    graph.add_nodes_from(tx_ids.tolist())
    graph.add_edges_from(edges.itertuples(index=False, name=None))
    return graph


def save_graph(graph: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    graph = build_graph()
    save_graph(graph)
    print(f"Built graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    print(f"Saved to {GRAPH_PATH}")
