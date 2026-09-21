"""OpenAI-compatible provider.

One adapter covers OpenAI, Azure-through-gateway (Bifrost), Gemini's OpenAI
endpoint, Ollama, vLLM and friends: anything that speaks
`POST {base_url}/chat/completions`.
"""

from __future__ import annotations

import logging

import httpx

from app.llm.base import LLMPermanentError, LLMResponse, LLMTransientError, Message

log = logging.getLogger(__name__)


class OpenAICompatProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise LLMPermanentError("LLM_API_KEY is empty; set it or use LLM_PROVIDER=fake")
        self.default_model = model
        base = (base_url or "https://api.openai.com").rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        self._client = httpx.AsyncClient(
            base_url=base,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 600,
        json_mode: bool = False,
    ) -> LLMResponse:
        body: dict = {
            "model": model or self.default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post("/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise LLMTransientError(f"network error talking to LLM: {exc}") from exc

        if resp.status_code in (408, 409, 429) or resp.status_code >= 500:
            raise LLMTransientError(f"LLM returned {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise LLMPermanentError(f"LLM returned {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMTransientError(f"malformed LLM response: {data}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text.strip(),
            model=data.get("model", body["model"]),
            provider=self.name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
