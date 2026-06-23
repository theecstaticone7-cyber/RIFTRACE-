"""POST /investigate/{tx_id}

Mirrors test_explain.py's approach: llm_service.generate_explanation() reads
GROQ_API_KEY fresh on every call, so monkeypatch.setenv/delenv is enough to
exercise both branches. The investigation graph calls the LLM twice (REASON,
then RECOMMEND), so the mocked client's side_effect supplies two responses
in call order.
"""

from unittest.mock import MagicMock

from api.services import llm_service

KNOWN_TX_ID = 232629023
UNKNOWN_TX_ID = 999


def _fake_response(text):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def test_investigate_unknown_transaction_returns_404(client):
    response = client.post(f"/investigate/{UNKNOWN_TX_ID}")

    assert response.status_code == 404


def test_investigate_without_groq_key_falls_back_gracefully(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = client.post(f"/investigate/{KNOWN_TX_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["tx_id"] == KNOWN_TX_ID
    assert body["prediction"] == "illicit"
    assert body["grounded"] is False
    # ANALYZE and RETRIEVE are deterministic -- they still ran even though
    # the LLM steps fell back.
    assert str(KNOWN_TX_ID) in body["analysis"]
    assert len(body["retrieved_sources"]) > 0
    assert "GROQ_API_KEY" in body["risk_assessment"]
    assert "GROQ_API_KEY" in body["recommended_actions"]


def test_investigate_with_groq_key_returns_grounded_report(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _fake_response("High risk: tied to known illicit patterns."),
        _fake_response("Trace the connected illicit neighbor for further activity."),
    ]
    monkeypatch.setattr(llm_service, "_create_client", lambda api_key: fake_client)

    response = client.post(f"/investigate/{KNOWN_TX_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["risk_assessment"] == "High risk: tied to known illicit patterns."
    assert body["recommended_actions"] == "Trace the connected illicit neighbor for further activity."
    assert len(body["retrieved_sources"]) > 0

    # REASON ran before RECOMMEND, and RECOMMEND's prompt included REASON's
    # own output -- confirms the agents actually passed data along the graph
    # rather than running independently of each other.
    calls = fake_client.chat.completions.create.call_args_list
    assert len(calls) == 2
    recommend_user_message = calls[1].kwargs["messages"][1]["content"]
    assert "High risk: tied to known illicit patterns." in recommend_user_message


def test_investigate_with_failed_groq_call_falls_back_gracefully(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(llm_service, "_create_client", lambda api_key: fake_client)

    response = client.post(f"/investigate/{KNOWN_TX_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert "unavailable" in body["risk_assessment"].lower()
    assert "unavailable" in body["recommended_actions"].lower()
