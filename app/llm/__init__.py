"""LLM layer: provider protocol, adapters, resilience wrapper and factory."""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMError, LLMPermanentError, LLMProvider, LLMResponse, LLMTransientError, Message
from app.llm.resilient import ResilientLLM

__all__ = [
    "LLMError",
    "LLMPermanentError",
    "LLMProvider",
    "LLMResponse",
    "LLMTransientError",
    "Message",
    "ResilientLLM",
    "build_llm",
]


def build_llm(settings: Settings) -> LLMProvider:
    """Construct the configured provider wrapped in retry/fallback logic."""
    if settings.llm_provider == "fake":
        from app.llm.fake import FakeLLM

        primary: LLMProvider = FakeLLM()
    elif settings.llm_provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        primary = AnthropicProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    else:
        from app.llm.openai_compat import OpenAICompatProvider

        primary = OpenAICompatProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    return ResilientLLM(primary, fallback_model=settings.llm_fallback_model, max_retries=settings.llm_max_retries)
