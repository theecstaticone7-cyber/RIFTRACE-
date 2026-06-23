"""POST /explain/{tx_id}

llm_service.generate_explanation() reads GROQ_API_KEY fresh on every call
(see llm_service.py), so monkeypatch.setenv/delenv around each test is
enough to exercise both branches -- no app restart needed despite `client`
being a session-scoped fixture.
"""

from unittest.mock import MagicMock

from api.services import llm_service

KNOWN_TX_ID = 232629023
UNKNOWN_TX_ID = 999


def test_explain_unknown_transaction_returns_404(client):
    response = client.post(f"/explain/{UNKNOWN_TX_ID}")

    assert response.status_code == 404


def test_explain_without_groq_key_falls_back_gracefully(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = client.post(f"/explain/{KNOWN_TX_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["tx_id"] == KNOWN_TX_ID
    assert body["prediction"] == "illicit"
    assert body["grounded"] is False
    assert "GROQ_API_KEY" in body["explanation"]
    # Retrieval is independent of the LLM key -- it should still have run.
    assert len(body["sources"]) > 0


def test_explain_with_groq_key_returns_grounded_explanation(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    fake_message = MagicMock(content="This transaction is risky because of its ties to known illicit activity.")
    fake_response = MagicMock(choices=[MagicMock(message=fake_message)])
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    # The real network call is mocked at this one seam (see
    # llm_service._create_client's docstring) so this test exercises the
    # real "key is configured" branch without ever hitting Groq.
    monkeypatch.setattr(llm_service, "_create_client", lambda api_key: fake_client)

    response = client.post(f"/explain/{KNOWN_TX_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["explanation"] == "This transaction is risky because of its ties to known illicit activity."
    assert len(body["sources"]) > 0

    # The prompt actually sent to the model included this transaction's
    # real data, not a generic placeholder.
    sent_messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    user_message = sent_messages[1]["content"]
    assert str(KNOWN_TX_ID) in user_message
    assert "illicit" in user_message


def test_explain_with_failed_groq_call_falls_back_gracefully(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(llm_service, "_create_client", lambda api_key: fake_client)

    response = client.post(f"/explain/{KNOWN_TX_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert "unavailable" in body["explanation"].lower()
