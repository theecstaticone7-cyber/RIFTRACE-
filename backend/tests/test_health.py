"""GET /health"""


def test_health_reports_model_and_graph_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["graph_loaded"] is True
