"""LLM abstraction.

The rest of the application only ever sees `LLMProvider`. Concrete adapters
(OpenAI-compatible, Anthropic, fake) implement `complete`. Swapping providers
is a config change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict = field(default_factory=dict, repr=False)


class LLMError(Exception):
    """Base class for provider failures the pipeline can react to."""


class LLMTransientError(LLMError):
    """Timeouts, 5xx, rate limits: worth a retry."""


class LLMPermanentError(LLMError):
    """Bad credentials, bad request: retrying will not help."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 600,
        json_mode: bool = False,
    ) -> LLMResponse: ...
