from fastapi.testclient import TestClient

import agent_platform.main as main
from agent_platform.llm_service import EmptyModelResponseError, LLMServiceError


def test_chat_endpoint_returns_generated_text(monkeypatch) -> None:
    async def fake_generate(model: str, prompt: str, temperature: float) -> str:
        return f"ok:{model}:{prompt}:{temperature}"

    monkeypatch.setattr(main.llm_service, "generate", fake_generate)

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={"model": "mistral", "prompt": "hello", "temperature": 0.2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "mistral",
        "response": "ok:mistral:hello:0.2",
    }


def test_chat_endpoint_returns_502_for_upstream_errors(monkeypatch) -> None:
    async def fake_generate(model: str, prompt: str, temperature: float) -> str:
        raise LLMServiceError("upstream failed")

    monkeypatch.setattr(main.llm_service, "generate", fake_generate)

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={"model": "mistral", "prompt": "hello", "temperature": 0.2},
    )

    assert response.status_code == 502
    assert "upstream failed" in response.json()["detail"]


def test_chat_endpoint_returns_500_for_empty_model_response(monkeypatch) -> None:
    async def fake_generate(model: str, prompt: str, temperature: float) -> str:
        raise EmptyModelResponseError("empty")

    monkeypatch.setattr(main.llm_service, "generate", fake_generate)

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={"model": "mistral", "prompt": "hello", "temperature": 0.2},
    )

    assert response.status_code == 500
    assert "empty" in response.json()["detail"]
