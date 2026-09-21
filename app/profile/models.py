"""User profile model and the parsing helpers around it."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timezone

from pydantic import BaseModel, Field, field_validator

from app.profile.astrology import AstroProfile, sun_sign_for


class UserProfile(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    name: str | None = None
    date_of_birth: date | None = None
    time_of_birth: time | None = None
    birth_place: str | None = None
    preferred_language: str = "English"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("name", "birth_place", mode="before")
    @classmethod
    def _strip(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @property
    def astro(self) -> AstroProfile | None:
        return sun_sign_for(self.date_of_birth) if self.date_of_birth else None

    @property
    def is_complete(self) -> bool:
        return bool(self.name and self.date_of_birth and self.birth_place)

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.name:
            missing.append("name")
        if not self.date_of_birth:
            missing.append("date_of_birth")
        if not self.time_of_birth:
            missing.append("time_of_birth")
        if not self.birth_place:
            missing.append("birth_place")
        return missing

    def compact_summary(self) -> str:
        """One-line profile for the prompt. Only known fields, never `None`."""
        bits = []
        if self.name:
            bits.append(f"Name: {self.name}")
        if self.date_of_birth:
            bits.append(f"Born: {self.date_of_birth.isoformat()}")
        if self.time_of_birth:
            bits.append(f"Time of birth: {self.time_of_birth.strftime('%H:%M')}")
        if self.birth_place:
            bits.append(f"Birth place: {self.birth_place}")
        bits.append(f"Preferred language: {self.preferred_language}")
        if a := self.astro:
            bits.append(f"Sun sign: {a.sun_sign} ({a.element}, ruled by {a.ruling_planet})")
        return "; ".join(bits)

    def merged_with(self, updates: dict) -> "UserProfile":
        """Return a copy with non-empty `updates` applied (used for corrections)."""
        data = self.model_dump()
        for k, v in updates.items():
            if v in (None, "", []):
                continue
            if k == "date_of_birth_text":
                parsed = parse_date_text(str(v))
                if parsed:
                    data["date_of_birth"] = parsed
                continue
            if k in data and k not in ("user_id", "created_at"):
                data[k] = v
        data["updated_at"] = datetime.now(timezone.utc)
        return UserProfile(**data)


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], start=1)}


def parse_date_text(text: str) -> date | None:
    """Parse '15 August 1995', 'August 15, 1995', '1995-08-15', '15/08/1995'."""
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})", text)
    if not m:
        m2 = re.search(r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", text)
        if not m2:
            return None
        day, month, year = int(m2.group(2)), m2.group(1), int(m2.group(3))
    else:
        day, month, year = int(m.group(1)), m.group(2), int(m.group(3))
    mon = _MONTHS.get(month.lower()) or _MONTHS.get(next((k for k in _MONTHS if k.startswith(month.lower()[:3])), ""), None)
    if not mon:
        return None
    try:
        return date(year, mon, day)
    except ValueError:
        return None
