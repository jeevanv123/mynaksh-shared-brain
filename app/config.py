"""Application settings.

Everything tunable lives here and is read from the environment (or a `.env`
file). Nothing else in the app reads `os.environ` directly, so the surface of
"what can be configured" is exactly this file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "MyNaksh Shared Brain"
    log_level: str = "INFO"

    # --- LLM -------------------------------------------------------------
    # "openai" covers every OpenAI-compatible endpoint: OpenAI itself, Bifrost,
    # Gemini's OpenAI endpoint, Ollama, vLLM. "anthropic" is the native API.
    # "fake" is deterministic and needs no network (used by tests / offline demo).
    llm_provider: Literal["openai", "anthropic", "fake"] = "fake"
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_model: str = "gpt-4.1-mini"
    llm_fallback_model: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.6

    # Cheaper model for extraction / classification side calls. Defaults to the
    # main model so a single-model setup still works.
    llm_small_model: str | None = None

    # --- Graph store -----------------------------------------------------
    graph_store: Literal["neo4j", "memory"] = "memory"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    # When Neo4j is configured but unreachable at boot, fall back to the
    # in-memory store instead of refusing to start.
    graph_fallback_to_memory: bool = True

    # --- Memory / context knobs -----------------------------------------
    short_term_max_turns: int = 20
    recent_turns_in_prompt: int = 6
    recent_turns_in_prompt_followup: int = 10
    max_memories_in_prompt: int = 8
    context_char_budget: int = 6000
    memory_confidence_floor: float = Field(default=0.55, ge=0.0, le=1.0)
    use_llm_for_intent: bool = True

    @property
    def small_model(self) -> str:
        return self.llm_small_model or self.llm_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
