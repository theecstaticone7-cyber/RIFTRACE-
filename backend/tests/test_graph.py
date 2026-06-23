"""GET /transaction/{tx_id}/graph"""

KNOWN_TX_ID = 232629023
UNKNOWN_TX_ID = 999


def test_graph_returns_neighbors_for_known_transaction(client):
    response = client.get(f"/transaction/{KNOWN_TX_ID}/graph")

    assert response.status_code == 200
    body = response.json()
    assert body["tx_id"] == KNOWN_TX_ID
    assert body["num_neighbors"] == len(body["neighbors"])
    assert body["num_neighbors"] > 0

    neighbor = body["neighbors"][0]
    assert neighbor["direction"] in ("incoming", "outgoing")
    assert neighbor["known_class"] in ("licit", "illicit", "unknown")


def test_graph_unknown_transaction_returns_404(client):
    response = client.get(f"/transaction/{UNKNOWN_TX_ID}/graph")

    assert response.status_code == 404
