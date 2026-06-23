"""POST /predict"""

# Fixed, known-label transactions from the Elliptic dataset -- not flaky
# samples, the model's classification of these never changes between runs.
KNOWN_ILLICIT_TX_ID = 232629023
KNOWN_LICIT_TX_ID = 232438397
UNKNOWN_TX_ID = 999


def test_predict_known_illicit_transaction(client):
    response = client.post("/predict", json={"tx_id": KNOWN_ILLICIT_TX_ID})

    assert response.status_code == 200
    body = response.json()
    assert body["tx_id"] == KNOWN_ILLICIT_TX_ID
    assert body["prediction"] == "illicit"
    assert body["probability_illicit"] >= 0.5


def test_predict_known_licit_transaction(client):
    response = client.post("/predict", json={"tx_id": KNOWN_LICIT_TX_ID})

    assert response.status_code == 200
    body = response.json()
    assert body["tx_id"] == KNOWN_LICIT_TX_ID
    assert body["prediction"] == "licit"
    assert body["probability_illicit"] < 0.5


def test_predict_unknown_transaction_returns_404(client):
    response = client.post("/predict", json={"tx_id": UNKNOWN_TX_ID})

    assert response.status_code == 404
