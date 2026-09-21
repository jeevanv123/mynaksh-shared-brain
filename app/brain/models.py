"""Shared Brain domain model.

A `Memory` is one fact about one user. In the graph it is an edge from the
`User` node to a shared concept node; here it is a flat record so the rest of
the app never thinks about Cypher.
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

LIFE_AREAS: tuple[str, ...] = (
    "career",
    "relationships",
    "finance",
    "health",
    "education",
    "family",
    "spirituality",
    "general",
)


class MemoryType(str, Enum):
    GOAL = "goal"
    PREFERENCE = "preference"
    INTEREST = "interest"
    LIFE_AREA = "life_area"
    ASTRO = "astro"
    FACT = "fact"
    EVENT = "event"


# Relationship type and concept label per memory type. Fixed enums, so they can
# be interpolated into Cypher safely.
REL_FOR_TYPE: dict[MemoryType, str] = {
    MemoryType.GOAL: "HAS_GOAL",
    MemoryType.PREFERENCE: "PREFERS",
    MemoryType.INTEREST: "INTERESTED_IN",
    MemoryType.LIFE_AREA: "CARES_ABOUT",
    MemoryType.ASTRO: "HAS_ATTRIBUTE",
    MemoryType.FACT: "REMEMBERS",
    MemoryType.EVENT: "EXPERIENCED",
}
LABEL_FOR_TYPE: dict[MemoryType, str] = {
    MemoryType.GOAL: "Goal",
    MemoryType.PREFERENCE: "Preference",
    MemoryType.INTEREST: "Interest",
    MemoryType.LIFE_AREA: "LifeArea",
    MemoryType.ASTRO: "AstroAttribute",
    MemoryType.FACT: "Fact",
    MemoryType.EVENT: "Event",
}
TYPE_FOR_REL = {v: k for k, v in REL_FOR_TYPE.items()}

# Types where the *key* names a slot that holds exactly one value at a time
# (preferred language, current city, sun sign). A new value supersedes the old.
EXCLUSIVE_SLOT_TYPES = {MemoryType.PREFERENCE, MemoryType.FACT, MemoryType.ASTRO}


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by a newer value for the same slot
    WITHDRAWN = "withdrawn"  # the user said it no longer holds


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_key(text: str) -> str:
    """'Career Change' -> 'career_change'. Stable across casing/punctuation."""
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text[:64] or "unknown"


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str
    type: MemoryType
    key: str
    value: str
    attributes: dict = Field(default_factory=dict)
    life_areas: list[str] = Field(default_factory=lambda: ["general"])
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE
    evidence: str | None = None
    source_session_id: str | None = None
    mention_count: int = 1
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)
    last_confirmed_at: datetime = Field(default_factory=now_utc)
    superseded_by: str | None = None

    @property
    def relationship(self) -> str:
        return REL_FOR_TYPE[self.type]

    @property
    def label(self) -> str:
        return LABEL_FOR_TYPE[self.type]

    def context_tag(self) -> str:
        """Short identifier surfaced in the API's `context_used`."""
        return f"{self.type.value}:{self.key}"

    def to_context_line(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in self.attributes.items() if v not in (None, ""))
        line = f"{self.label}: {self.value}"
        if attrs:
            line += f" ({attrs})"
        return line

    def rank_score(self, now: datetime | None = None) -> float:
        """importance x confidence x recency decay (half-life 180 days)."""
        now = now or now_utc()
        age_days = max((now - self.last_confirmed_at).total_seconds() / 86400.0, 0.0)
        recency = math.exp(-math.log(2) * age_days / 180.0)
        # Time-bound goals whose target year has passed fade harder.
        target_year = self.attributes.get("target_year")
        stale_penalty = 0.5 if isinstance(target_year, int) and target_year < now.year else 1.0
        return self.importance * self.confidence * (0.5 + 0.5 * recency) * stale_penalty


class MemoryCandidate(BaseModel):
    """What the extractor proposes, before policy decides whether to keep it."""

    type: MemoryType
    key: str | None = None
    value: str = Field(min_length=1)
    attributes: dict = Field(default_factory=dict)
    life_areas: list[str] = Field(default_factory=lambda: ["general"])
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    persistent: bool = True
    evidence: str | None = None
    replaces: str | None = None  # key of an existing memory this one supersedes / withdraws
    status: str = "active"  # "active" | "withdrawn"

    def resolved_key(self) -> str:
        return normalize_key(self.key) if self.key else normalize_key(self.value)


class ExtractionResult(BaseModel):
    memories: list[MemoryCandidate] = Field(default_factory=list)
    profile_updates: dict = Field(default_factory=dict)


class MemoryChange(BaseModel):
    """Audit record of what the update step did with one candidate."""

    action: str  # created | confirmed | superseded | withdrawn | dropped
    memory: Memory | None = None
    reason: str | None = None

    def summary(self) -> str:
        if self.memory is None:
            return f"{self.action}: {self.reason}"
        return f"{self.action}: {self.memory.to_context_line()}"
