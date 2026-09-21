"""Deterministic fake LLM.

Used by the test-suite and by the offline demo (`LLM_PROVIDER=fake`). It looks
for the task marker the prompt builder puts in the system message and answers
each task with small, predictable heuristics:

- memory extraction: regex over the user's message, emitting the same JSON
  schema the real extractor prompt requests;
- intent classification: keyword map;
- chat: a templated reply that echoes the context it was given, so a human
  running the demo can see which memories reached the prompt.

It is intentionally dumb. Its job is to make the pipeline testable without a
network, not to be clever.
"""

from __future__ import annotations

import json
import re
from datetime import date

from app.llm.base import LLMError, LLMResponse, Message
from app.chat import prompts


class FakeLLM:
    name = "fake"

    def __init__(self, *, fail: bool = False, script: dict[str, str] | None = None) -> None:
        self.fail = fail
        self.script = script or {}
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message], *, model=None, temperature=0.6, max_tokens=600, json_mode=False) -> LLMResponse:
        self.calls.append(messages)
        if self.fail:
            raise LLMError("fake LLM configured to fail")
        system = "\n".join(m.content for m in messages if m.role == "system")
        user_msgs = [m.content for m in messages if m.role == "user"]
        last_user = user_msgs[-1] if user_msgs else ""

        if prompts.EXTRACT_MARKER in system:
            text = json.dumps(self._extract(last_user))
        elif prompts.INTENT_MARKER in system:
            text = json.dumps(self._intent(last_user))
        else:
            text = self._chat(system, last_user)
        return LLMResponse(text=text, model="fake-1", provider=self.name, prompt_tokens=len(system) // 4, completion_tokens=len(text) // 4)

    # ---- heuristics -----------------------------------------------------

    def _extract(self, raw: str) -> dict:
        # The extractor prompt wraps the message; pull the message body out.
        m = re.search(r"USER MESSAGE:\s*(.*?)(?:\n\s*\n|\Z)", raw, flags=re.S)
        text = (m.group(1) if m else raw).strip()
        low = text.lower()
        memories: list[dict] = []
        profile: dict = {}
        year = date.today().year

        if mt := re.search(r"my name is ([A-Z][a-z]+)", text, flags=re.I):
            profile["name"] = mt.group(1).title()
        if mt := re.search(r"born on (\d{1,2} \w+ \d{4})(?: in ([A-Z][a-zA-Z]+))?", text, flags=re.I):
            profile["date_of_birth_text"] = mt.group(1)
            if mt.group(2):
                profile["birth_place"] = mt.group(2)
        if mt := re.search(r"\bi (?:now )?live in ([A-Z][a-zA-Z]+)|moved to ([A-Z][a-zA-Z]+)", text):
            city = mt.group(1) or mt.group(2)
            memories.append(_mem("fact", f"Lives in {city}", ["general"], 0.85, key="current_city"))

        withdrawing = bool(re.search(r"\b(no longer|not planning|decided not|actually,? i(?:'m| am) not|changed my mind)\b", low))
        if re.search(r"switch(?:ing)? jobs|change (?:my )?career|career change|new job|quit my job", low):
            if withdrawing:
                memories.append({**_mem("goal", "Career Change", ["career"], 0.9), "persistent": True, "replaces": "career_change", "status": "withdrawn"})
            else:
                attrs = {}
                if "next year" in low:
                    attrs["target_year"] = year + 1
                elif mt := re.search(r"\b(20\d\d)\b", low):
                    attrs["target_year"] = int(mt.group(1))
                memories.append(_mem("goal", "Career Change", ["career"], 0.9, attrs))
        if "product management interview" in low or "pm interview" in low:
            attrs = {"timeframe": "next month"} if "next month" in low else {}
            memories.append(_mem("goal", "Product Management Interview", ["career"], 0.85, attrs))
        if mt := re.search(r"(?:prefer|reply in|answer in|talk to me in) (hindi|english|tamil|marathi)", low):
            memories.append(_mem("preference", mt.group(1).title(), ["general"], 0.9, key="language"))
        if mt := re.search(r"interested in ([a-z ]+?)(?:[.,]|$)", low):
            memories.append(_mem("interest", mt.group(1).strip().title(), ["career"], 0.8))
        if mt := re.search(r"(?:getting married|my wedding is) (?:in|next) (\w+)", low):
            memories.append(_mem("event", "Wedding", ["relationships", "family"], 0.85, {"timeframe": mt.group(1)}))
        if "started a business" in low or "my startup" in low:
            memories.append(_mem("interest", "Entrepreneurship", ["career", "finance"], 0.8))
        return {"memories": memories, "profile_updates": profile}

    def _intent(self, raw: str) -> dict:
        low = raw.lower()
        areas = [a for a, kws in _AREA_KEYWORDS.items() if any(k in low for k in kws)] or ["general"]
        return {"intent": "advice", "life_areas": areas, "needs_astrology": True}

    def _chat(self, system: str, last_user: str) -> str:
        if "chat" in self.script:
            return self.script["chat"]
        ctx = re.search(r"## Relevant long-term memories\n(.*?)(?:\n## |\Z)", system, flags=re.S)
        prof = re.search(r"## User profile\n(.*?)(?:\n## |\Z)", system, flags=re.S)
        parts = ["[fake-llm]"]
        if prof:
            parts.append("Profile: " + " ".join(prof.group(1).split()))
        if ctx:
            parts.append("Using memories: " + " ".join(ctx.group(1).split()))
        else:
            parts.append("No long-term memories used.")
        parts.append(f"Answering: {last_user.strip()}")
        return " | ".join(parts)


_AREA_KEYWORDS = {
    "career": ["career", "job", "work", "interview", "promotion"],
    "relationships": ["love", "marriage", "partner", "relationship"],
    "finance": ["money", "invest", "finance", "wealth"],
    "health": ["health", "fitness", "stress"],
}


def _mem(type_: str, value: str, areas: list[str], conf: float, attrs: dict | None = None, key: str | None = None) -> dict:
    return {
        "type": type_,
        "key": key,
        "value": value,
        "attributes": attrs or {},
        "life_areas": areas,
        "confidence": conf,
        "persistent": True,
        "evidence": value,
    }
