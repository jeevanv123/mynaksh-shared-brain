"""HTTP request / response contracts."""

from __future__ import annotations

from datetime import date, datetime, time

from pydantic import BaseModel, Field, field_validator

from app.brain.models import Memory, MemoryChange


class CreateUserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, examples=["user-123"])
    name: str | None = Field(default=None, max_length=120)
    date_of_birth: date | None = Field(default=None, examples=["1995-08-15"])
    time_of_birth: time | None = Field(default=None, examples=["10:30"])
    birth_place: str | None = Field(default=None, max_length=120, examples=["Delhi"])
    preferred_language: str = Field(default="English", max_length=40)

    @field_validator("date_of_birth")
    @classmethod
    def _not_in_future(cls, v: date | None) -> date | None:
        if v and v > date.today():
            raise ValueError("date_of_birth cannot be in the future")
        return v


class AstroOut(BaseModel):
    sun_sign: str
    element: str
    modality: str
    ruling_planet: str
    traits: str


class UserOut(BaseModel):
    user_id: str
    name: str | None
    date_of_birth: date | None
    time_of_birth: time | None
    birth_place: str | None
    preferred_language: str
    astrology: AstroOut | None
    missing_fields: list[str]
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, examples=["user-123"])
    session_id: str = Field(min_length=1, max_length=128, examples=["session-456"])
    message: str = Field(min_length=1, max_length=4000, examples=["What should I focus on in my career?"])

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v.strip()


class MemoryOut(BaseModel):
    id: str
    type: str
    relationship: str
    key: str
    value: str
    attributes: dict
    life_areas: list[str]
    confidence: float
    importance: float
    status: str
    mention_count: int
    evidence: str | None
    source_session_id: str | None
    created_at: datetime
    last_confirmed_at: datetime
    superseded_by: str | None

    @classmethod
    def from_memory(cls, m: Memory) -> "MemoryOut":
        return cls(
            id=m.id,
            type=m.type.value,
            relationship=m.relationship,
            key=m.key,
            value=m.value,
            attributes=m.attributes,
            life_areas=m.life_areas,
            confidence=round(m.confidence, 3),
            importance=round(m.importance, 3),
            status=m.status.value,
            mention_count=m.mention_count,
            evidence=m.evidence,
            source_session_id=m.source_session_id,
            created_at=m.created_at,
            last_confirmed_at=m.last_confirmed_at,
            superseded_by=m.superseded_by,
        )


class MemoryChangeOut(BaseModel):
    action: str
    memory: MemoryOut | None = None
    reason: str | None = None

    @classmethod
    def from_change(cls, c: MemoryChange) -> "MemoryChangeOut":
        return cls(action=c.action, memory=MemoryOut.from_memory(c.memory) if c.memory else None, reason=c.reason)


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    context_used: list[str]
    intent: str
    life_areas: list[str]
    memory_updates: list[MemoryChangeOut]
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
    approx_prompt_tokens: int = 0


class MemoriesOut(BaseModel):
    user_id: str
    count: int
    memories: list[MemoryOut]


class HealthOut(BaseModel):
    status: str
    graph_store: str
    graph_ok: bool
    llm_provider: str
    llm_model: str
