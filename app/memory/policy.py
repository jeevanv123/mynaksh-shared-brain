"""What is worth remembering.

The extractor proposes; this module decides. Rules are deliberately explicit
and cheap so they can be reasoned about in a code review, which an LLM
judgement cannot. The LLM already tagged each candidate with `persistent` and
`confidence`; the policy is the second, deterministic gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.brain.models import LIFE_AREAS, MemoryCandidate, MemoryType

# Types the brain is allowed to persist from chat. LIFE_AREA and ASTRO are
# derived by the system, not accepted from extraction.
ALLOWED_FROM_CHAT = {MemoryType.GOAL, MemoryType.PREFERENCE, MemoryType.INTEREST, MemoryType.FACT, MemoryType.EVENT}

# Profile fields belong on the User node. If the extractor also emits them as
# facts they would duplicate the profile, so they are dropped here.
PROFILE_FIELD_KEYS = {
    "name", "full_name", "first_name", "user_name",
    "date_of_birth", "dob", "birth_date", "birthdate", "birthday",
    "time_of_birth", "birth_time",
    "birth_place", "birthplace", "place_of_birth", "born_in",
}

# Catches extractor variants like date_of_birth_text, birth_year, born_in, birthday.
_PROFILE_KEY_PATTERN = re.compile(r"birth|^dob$|\bborn|^(full_|first_|user_)?name$", re.I)

KNOWN_LANGUAGES = {
    "english", "hindi", "hinglish", "tamil", "telugu", "kannada", "malayalam", "marathi",
    "gujarati", "bengali", "punjabi", "odia", "urdu", "assamese",
}

_TRANSIENT_PATTERNS = re.compile(
    r"\b(today|tonight|right now|this morning|this evening|at the moment|currently feeling|"
    r"tired|sleepy|hungry|bored|headache)\b",
    re.I,
)
_QUESTION_WORDS = re.compile(r"^(what|why|how|when|where|who|which|should|can|could|would|will|is|are|do|does|did)\b", re.I)
# A first-person *declaration*: something the user states about themselves.
# "my" alone is not enough ("what about my career?" is still just a question).
_DECLARATION = re.compile(
    r"\b(i am|i'm|i was|i have|i've|i will|i'll|i plan to|i'm planning|i am planning|i want|i prefer|i like|i love|"
    r"i live|i work|i moved|i got|i started|i decided|i no longer|my name is|my \w+ is|my \w+ are)\b",
    re.I,
)


@dataclass
class PolicyDecision:
    keep: bool
    reason: str


class MemoryPolicy:
    def __init__(self, confidence_floor: float = 0.55, max_value_len: int = 80) -> None:
        self.confidence_floor = confidence_floor
        self.max_value_len = max_value_len

    # ---- pre-extraction gate --------------------------------------------

    def worth_extracting(self, message: str) -> bool:
        """Cheap check that skips the extraction LLM call for pure questions."""
        text = message.strip()
        if len(text) < 8:
            return False
        is_question = text.endswith("?") or bool(_QUESTION_WORDS.match(text))
        if is_question:
            # A question only carries a memory if it also states something
            # ("Should I take the job? I have an offer from Google.").
            return bool(_DECLARATION.search(text))
        return True

    # ---- post-extraction gate -------------------------------------------

    def evaluate(self, candidate: MemoryCandidate) -> PolicyDecision:
        if candidate.type not in ALLOWED_FROM_CHAT:
            return PolicyDecision(False, f"type {candidate.type.value} is system-derived, not accepted from chat")
        if candidate.status == "withdrawn":
            # Withdrawals are always processed; they remove, never add.
            return PolicyDecision(True, "withdrawal of an existing memory")
        if candidate.type == MemoryType.FACT and self.is_profile_field(candidate.key):
            return PolicyDecision(False, f"{candidate.key} is a profile field; stored on the user profile, not as a memory")
        if not candidate.persistent:
            return PolicyDecision(False, "extractor marked it non-persistent")
        if candidate.confidence < self.confidence_floor:
            return PolicyDecision(False, f"confidence {candidate.confidence:.2f} below floor {self.confidence_floor:.2f}")
        value = candidate.value.strip()
        if len(value) > self.max_value_len:
            return PolicyDecision(False, "value too long to be a memory; looks like a transcript")
        if value.endswith("?") or _QUESTION_WORDS.match(value):
            return PolicyDecision(False, "value is a question, not a fact")
        if _TRANSIENT_PATTERNS.search(value) and candidate.type not in (MemoryType.GOAL, MemoryType.EVENT):
            return PolicyDecision(False, "transient state (mood / today) is not long-term memory")
        return PolicyDecision(True, "durable, confident, user-specific")

    def normalize(self, candidate: MemoryCandidate) -> MemoryCandidate:
        areas = [a for a in candidate.life_areas if a in LIFE_AREAS] or ["general"]
        key = candidate.key
        value = candidate.value.strip()
        type_ = candidate.type
        # A language preference must land in one slot, whatever the extractor
        # called it (preference/language, fact/preferred_language, ...), so a
        # new language supersedes the old one instead of duplicating it.
        if type_ in (MemoryType.PREFERENCE, MemoryType.FACT) and (
            (key and "lang" in key.lower()) or value.lower() in KNOWN_LANGUAGES
        ):
            type_ = MemoryType.PREFERENCE
            key = "language"
            value = value.title()
        return candidate.model_copy(update={"type": type_, "life_areas": areas, "value": value, "key": key})

    @staticmethod
    def is_profile_field(key: str | None) -> bool:
        k = (key or "").lower()
        return k in PROFILE_FIELD_KEYS or bool(_PROFILE_KEY_PATTERN.search(k))

    def importance_for(self, candidate: MemoryCandidate) -> float:
        base = {
            MemoryType.GOAL: 0.9,
            MemoryType.EVENT: 0.8,
            MemoryType.FACT: 0.7,
            MemoryType.PREFERENCE: 0.6,
            MemoryType.INTEREST: 0.5,
        }.get(candidate.type, 0.5)
        if candidate.attributes.get("target_year") or candidate.attributes.get("timeframe"):
            base = min(1.0, base + 0.05)
        return base
