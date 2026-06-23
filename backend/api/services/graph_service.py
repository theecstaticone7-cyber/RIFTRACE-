"""Loads the cached transaction graph once and serves neighbor lookups for
the graph-visualization endpoint.

Class labels are looked up via model_service's shared classes.csv cache
(get_class_label), not a separate copy -- the ~13MB classes.csv used to be
loaded independently by both services.
"""

import pickle

from .model_service import DATA_PROCESSED, get_class_label

GRAPH_PATH = DATA_PROCESSED / "transaction_graph.gpickle"

REQUIRED_FILES = [GRAPH_PATH]

# Populated on first use by _load() (module-level cache).
_graph = None


def _load() -> None:
    global _graph
    if _graph is not None:
        return

    with open(GRAPH_PATH, "rb") as f:
        _graph = pickle.load(f)


def warm_up() -> None:
    """Eagerly loads the graph at app startup -- see model_service.warm_up
    for why this matters for fail-fast behavior.
    """
    _load()


def get_neighbors(tx_id: int) -> dict:
    _load()
    if tx_id not in _graph:
        raise KeyError(f"Transaction {tx_id} not found in graph")

    neighbors = []
    for source in _graph.predecessors(tx_id):
        neighbors.append(
            {"tx_id": source, "direction": "incoming", "known_class": get_class_label(source)}
        )
    for target in _graph.successors(tx_id):
        neighbors.append(
            {"tx_id": target, "direction": "outgoing", "known_class": get_class_label(target)}
        )

    return {
        "tx_id": tx_id,
        "known_class": get_class_label(tx_id),
        "neighbors": neighbors,
        "num_neighbors": len(neighbors),
    }


def health_check() -> dict:
    """Confirms the graph is actually loaded, not just that the file exists."""
    _load()
    return {"graph_loaded": True, "graph_nodes": _graph.number_of_nodes()}
