"""GET /flagged"""


def test_flagged_is_sorted_by_probability_descending(client):
    response = client.get("/flagged?limit=100")

    assert response.status_code == 200
    body = response.json()
    probabilities = [row["probability_illicit"] for row in body["flagged"]]

    assert len(probabilities) > 1
    assert probabilities == sorted(probabilities, reverse=True)


def test_flagged_respects_limit_and_reports_count(client):
    response = client.get("/flagged?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    assert len(body["flagged"]) == 5
