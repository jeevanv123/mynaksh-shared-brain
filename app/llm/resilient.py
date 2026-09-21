"""Retry + fallback wrapper around any LLMProvider.

Transient failures are retried with exponential backoff. If the primary model
keeps failing, an optional fallback model (or a whole fallback provider) is
tried before giving up. The pipeline above this only ever sees `LLMError`.
"""

from __future__ import annotations

import asyncio
import logging
import random

from app.llm.base import LLMError, LLMPermanentError, LLMProvider, LLMResponse, LLMTransientError, Message

log = logging.getLogger(__name__)


class ResilientLLM:
    name = "resilient"

    def __init__(
        self,
        primary: LLMProvider,
        *,
        fallback_model: str | None = None,
        fallback_provider: LLMProvider | None = None,
        max_retries: int = 2,
        base_delay: float = 0.4,
    ) -> None:
        self.primary = primary
        self.fallback_model = fallback_model
        self.fallback_provider = fallback_provider
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def _with_retries(self, provider: LLMProvider, messages, **kwargs) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return await provider.complete(messages, **kwargs)
            except LLMTransientError as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                delay = self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
                log.warning("LLM transient failure (%s); retry %d/%d in %.1fs", exc, attempt, self.max_retries, delay)
                await asyncio.sleep(delay)

    async def complete(self, messages: list[Message], **kwargs) -> LLMResponse:
        try:
            return await self._with_retries(self.primary, messages, **kwargs)
        except LLMError as primary_exc:
            log.error("primary LLM failed: %s", primary_exc)
            if self.fallback_model and kwargs.get("model") != self.fallback_model:
                try:
                    return await self._with_retries(self.primary, messages, **{**kwargs, "model": self.fallback_model})
                except LLMError as exc:
                    log.error("fallback model failed: %s", exc)
            if self.fallback_provider is not None:
                try:
                    return await self._with_retries(self.fallback_provider, messages, **{**kwargs, "model": None})
                except LLMError as exc:
                    log.error("fallback provider failed: %s", exc)
            if isinstance(primary_exc, LLMPermanentError):
                raise
            raise LLMError(f"all LLM routes failed: {primary_exc}") from primary_exc
