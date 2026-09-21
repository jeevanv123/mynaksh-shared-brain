"""Query understanding: what kind of message is this, and which parts of the
user's life does answering it need?

Rules first (free, deterministic, cover the common cases), LLM second (only
when rules are unsure and the message is long enough to be worth a call).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.brain.models import LIFE_AREAS
from app.chat import prompts
from app.llm.base import LLMError, LLMProvider, Message

log = logging.getLogger(__name__)

AREA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "career": ("career", "job", "work", "promotion", "interview", "boss", "salary", "startup", "business", "profession", "office", "switch"),
    "relationships": ("love", "marriage", "marry", "partner", "relationship", "wife", "husband", "girlfriend", "boyfriend", "dating", "wedding", "divorce"),
    "finance": ("money", "invest", "investment", "finance", "financial", "wealth", "savings", "loan", "debt", "property"),
    "health": ("health", "fitness", "illness", "stress", "anxiety", "sleep", "diet", "wellbeing", "well-being"),
    "education": ("exam", "study", "studies", "college", "university", "degree", "course", "learning", "school"),
    "family": ("family", "parents", "mother", "father", "children", "kids", "son", "daughter", "sibling"),
    "spirituality": ("spiritual", "meditation", "karma", "puja", "temple", "prayer", "peace", "purpose"),
}

_FOLLOWUP = re.compile(
    r"^(why|how come|what do you mean|can you explain|explain|elaborate|tell me more|really\??|"
    r"are you sure|why do you say that|what makes you say|and\b|so\b|ok(ay)? but)",
    re.I,
)
_ANAPHORA = re.compile(r"\b(that|this|it|those|these)\b", re.I)
_RECALL = re.compile(r"\b(remember|recall|what do you know about me|my goals?|what did i tell you|do you know)\b", re.I)
_PROFILE_DECL = re.compile(r"\b(my name is|i was born|born on|i am \d{1,2} years|my birth|i live in|i prefer|reply in)\b", re.I)
_ASTRO = re.compile(r"\b(horoscope|zodiac|sign|planet|saturn|jupiter|rashi|kundli|kundali|nakshatra|astrolog|stars?)\b", re.I)


@dataclass
class QueryUnderstanding:
    intent: str  # advice | recall | followup | smalltalk | profile_update
    life_areas: list[str] = field(default_factory=lambda: ["general"])
    needs_astrology: bool = True
    source: str = "rules"  # rules | llm

    @property
    def is_followup(self) -> bool:
        return self.intent == "followup"


class QueryUnderstander:
    def __init__(self, llm: LLMProvider | None, *, model: str | None = None, use_llm: bool = True) -> None:
        self.llm = llm
        self.model = model
        self.use_llm = use_llm and llm is not None

    async def understand(self, message: str, previous_areas: list[str] | None = None) -> QueryUnderstanding:
        text = message.strip()
        low = text.lower()
        words = low.split()

        areas = [area for area, kws in AREA_KEYWORDS.items() if any(k in low for k in kws)]

        if _RECALL.search(low):
            return QueryUnderstanding("recall", areas or list(LIFE_AREAS), needs_astrology=False)

        if len(words) <= 8 and (_FOLLOWUP.match(text) or (_ANAPHORA.search(low) and not areas)):
            return QueryUnderstanding("followup", previous_areas or ["general"], needs_astrology=False)

        if _PROFILE_DECL.search(low) and not text.endswith("?"):
            return QueryUnderstanding("profile_update", areas or ["general"], needs_astrology=False)

        if areas:
            return QueryUnderstanding("advice", areas, needs_astrology=True)

        if len(words) <= 3:
            return QueryUnderstanding("smalltalk", ["general"], needs_astrology=False)

        if self.use_llm and len(words) >= 5:
            llm_result = await self._classify_with_llm(text)
            if llm_result:
                return llm_result

        return QueryUnderstanding("advice", ["general"], needs_astrology=bool(_ASTRO.search(low)) or True)

    async def _classify_with_llm(self, text: str) -> QueryUnderstanding | None:
        try:
            resp = await self.llm.complete(
                [Message("system", prompts.INTENT_SYSTEM), Message("user", text)],
                model=self.model,
                temperature=0.0,
                max_tokens=120,
                json_mode=True,
            )
            data = json.loads(re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.S))
        except (LLMError, json.JSONDecodeError, TypeError) as exc:
            log.info("intent LLM unavailable, using rules: %s", exc)
            return None
        intent = str(data.get("intent", "advice")).lower()
        if intent not in {"advice", "recall", "followup", "smalltalk", "profile_update"}:
            intent = "advice"
        areas = [a for a in data.get("life_areas", []) if a in LIFE_AREAS] or ["general"]
        return QueryUnderstanding(intent, areas, bool(data.get("needs_astrology", True)), source="llm")
