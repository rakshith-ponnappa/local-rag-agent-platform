import os

import httpx


class LLMServiceError(Exception):
    pass


class EmptyModelResponseError(LLMServiceError):
    pass


class OllamaService:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.timeout = timeout

    async def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            raise LLMServiceError(f"Ollama call failed: {exc}") from exc

        text = body.get("response", "").strip()
        if not text:
            raise EmptyModelResponseError("Model returned empty response")

        return text
