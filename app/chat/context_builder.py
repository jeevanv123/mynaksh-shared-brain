"""Select relevant context and assemble the prompt under a budget.

Priority order when the budget is tight: profile > astrology > memories > recent
turns. Each block that makes it into the prompt is reported in `context_used`,
so the API caller (and the tests) can see exactly what the model saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.brain.models import Memory
from app.chat import prompts
from app.chat.intent import QueryUnderstanding
from app.llm.base import Message
from app.memory.short_term import Turn
from app.profile.models import UserProfile


@dataclass
class BuiltContext:
    messages: list[Message]
    context_used: list[str] = field(default_factory=list)
    memories_used: list[Memory] = field(default_factory=list)
    approx_tokens: int = 0


class ContextBuilder:
    def __init__(self, *, max_memories: int = 8, char_budget: int = 6000) -> None:
        self.max_memories = max_memories
        self.char_budget = char_budget

    def select_memories(self, memories: list[Memory], query: QueryUnderstanding) -> list[Memory]:
        """Filter by life area (already done by the store) and rank."""
        if query.intent == "followup":
            # Follow-ups lean on the short-term buffer; keep a few memories so the
            # model can still ground "why" in the user's goals.
            limit = min(3, self.max_memories)
        elif query.intent == "recall":
            limit = max(self.max_memories, 20)
        else:
            limit = self.max_memories
        ranked = sorted(memories, key=lambda m: m.rank_score(), reverse=True)
        return ranked[:limit]

    def build(
        self,
        *,
        profile: UserProfile,
        query: QueryUnderstanding,
        memories: list[Memory],
        recent_turns: list[Turn],
        message: str,
        degraded_note: str | None = None,
    ) -> BuiltContext:
        used: list[str] = []
        sections: list[str] = [prompts.SYSTEM_PERSONA]

        # 1. Profile (always; even an empty one tells the model what to ask for)
        if profile.name or profile.date_of_birth or profile.birth_place:
            sections.append(f"## User profile\n{profile.compact_summary()}")
            used.append("user_profile")
        else:
            missing = ", ".join(profile.missing_fields())
            sections.append(f"## User profile\nUnknown. Missing: {missing}. Preferred language: {profile.preferred_language}.")
            used.append("profile_missing")

        # 2. Astrology (only when the question calls for it)
        if query.needs_astrology and profile.astro:
            a = profile.astro
            sections.append(
                f"## Astrology attributes\nSun sign {a.sun_sign}; element {a.element}; modality {a.modality}; "
                f"ruling planet {a.ruling_planet}; tendencies: {a.traits}."
            )
            used.append("astrology")

        # 3. Long-term memories, budgeted
        selected = self.select_memories(memories, query)
        if selected:
            lines = [f"- {m.to_context_line()}" for m in selected]
            sections.append("## Relevant long-term memories\n" + "\n".join(lines))
            used.extend(m.context_tag() for m in selected)
        elif query.intent == "recall":
            sections.append("## Relevant long-term memories\n(none stored yet)")
            used.append("no_memories")

        if query.intent == "recall":
            sections.append("## Task\nThe user is asking what you remember. Answer from the memories above only.")
        elif query.intent == "followup":
            sections.append("## Task\nThis is a follow-up to your previous reply. Explain or expand on it; stay consistent with what you said.")

        if degraded_note:
            sections.append(f"## Note\n{degraded_note}")

        system = "\n\n".join(sections)

        # 4. Recent turns, trimmed to whatever budget is left
        convo: list[Message] = []
        remaining = self.char_budget - len(system) - len(message)
        for turn in reversed(recent_turns):
            if remaining - len(turn.content) < 0:
                break
            convo.insert(0, Message("assistant" if turn.role == "assistant" else "user", turn.content))
            remaining -= len(turn.content)
        if convo:
            used.append("recent_conversation")

        msgs = [Message("system", system), *convo, Message("user", message)]
        approx = sum(len(m.content) for m in msgs) // 4
        return BuiltContext(messages=msgs, context_used=used, memories_used=selected, approx_tokens=approx)
