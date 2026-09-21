"""Native Anthropic Messages API adapter (no SDK dependency, plain httpx)."""

from __future__ import annotations

import httpx

from app.llm.base import LLMPermanentError, LLMResponse, LLMTransientError, Message


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, base_url: str | None = None, timeout: float = 30.0) -> None:
        if not api_key:
            raise LLMPermanentError("LLM_API_KEY is empty")
        self.default_model = model
        self._client = httpx.AsyncClient(
            base_url=(base_url or "https://api.anthropic.com").rstrip("/"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
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
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        if json_mode:
            system += "\n\nRespond with a single valid JSON object and nothing else."
        body = {
            "model": model or self.default_model,
            "system": system,
            "messages": convo,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = await self._client.post("/v1/messages", json=body)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise LLMTransientError(str(exc)) from exc
        if resp.status_code in (408, 409, 429, 529) or resp.status_code >= 500:
            raise LLMTransientError(f"Anthropic returned {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise LLMPermanentError(f"Anthropic returned {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text.strip(),
            model=data.get("model", body["model"]),
            provider=self.name,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
