"""GET /stats"""


def test_stats_returns_expected_metric_fields(client):
    response = client.get("/stats")

    assert response.status_code == 200
    metrics = response.json()["metrics"]

    for field in ("precision", "recall", "f1", "roc_auc"):
        assert field in metrics
        assert 0.0 <= metrics[field] <= 1.0
