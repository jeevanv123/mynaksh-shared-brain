"""ChatService: the end-to-end pipeline.

    message -> understand -> select context -> build prompt -> LLM -> respond -> update memory

Each stage degrades independently. A dead graph still yields an answer from
profile + short-term context; a dead LLM yields a polite fallback and the turn
is still recorded; a failed extraction never fails the response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.brain.models import Memory, MemoryCandidate, MemoryChange, MemoryType
from app.brain.service import SharedBrain
from app.brain.store import GraphStoreError
from app.chat.context_builder import ContextBuilder
from app.chat.intent import QueryUnderstander, QueryUnderstanding
from app.config import Settings
from app.llm.base import LLMError, LLMProvider, Message
from app.memory.extractor import MemoryExtractor
from app.memory.short_term import ConversationBuffer
from app.profile.models import UserProfile

log = logging.getLogger(__name__)

FALLBACK_REPLY = (
    "I'm having trouble reaching the stars right now. Please try again in a moment; "
    "everything you've told me is safe and I'll pick up where we left off."
)


@dataclass
class ChatResult:
    response: str
    context_used: list[str]
    intent: str
    life_areas: list[str]
    memories_used: list[Memory]
    memory_changes: list[MemoryChange] = field(default_factory=list)
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)
    model: str | None = None
    approx_prompt_tokens: int = 0
    new_user: bool = False


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMProvider,
        brain: SharedBrain,
        buffer: ConversationBuffer,
        understander: QueryUnderstander,
        extractor: MemoryExtractor,
        context_builder: ContextBuilder,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.brain = brain
        self.buffer = buffer
        self.understander = understander
        self.extractor = extractor
        self.context_builder = context_builder
        self._last_areas: dict[tuple[str, str], list[str]] = {}

    async def chat(self, user_id: str, session_id: str, message: str) -> ChatResult:
        warnings: list[str] = []
        degraded = False

        # 0. Profile (auto-create so a brand-new user still gets an answer)
        try:
            profile, is_new = await self.brain.get_or_create_user(user_id)
        except GraphStoreError as exc:
            log.error("graph unavailable while loading profile: %s", exc)
            profile, is_new = UserProfile(user_id=user_id), False
            warnings.append("graph_unavailable")
            degraded = True

        # 1. Understand the query
        prev_areas = self._last_areas.get((user_id, session_id))
        query = await self.understander.understand(message, previous_areas=prev_areas)

        # 2. Select relevant long-term context
        memories: list[Memory] = []
        if "graph_unavailable" not in warnings:
            try:
                areas = None if query.intent == "recall" else query.life_areas
                memories = await self.brain.relevant_memories(user_id, areas)
            except GraphStoreError as exc:
                log.error("graph unavailable while reading memories: %s", exc)
                warnings.append("graph_unavailable")
                degraded = True

        # 3. Build the prompt (profile + astrology + memories + recent turns)
        n_turns = self.settings.recent_turns_in_prompt_followup if query.is_followup else self.settings.recent_turns_in_prompt
        recent = self.buffer.recent(user_id, session_id, n_turns)
        built = self.context_builder.build(
            profile=profile,
            query=query,
            memories=memories,
            recent_turns=recent,
            message=message,
            degraded_note="Long-term memory is temporarily unavailable; answer from the conversation and profile only." if degraded else None,
        )

        # 4. Generate
        model_used: str | None = None
        try:
            llm_resp = await self.llm.complete(built.messages, temperature=self.settings.llm_temperature, max_tokens=700)
            reply = llm_resp.text or FALLBACK_REPLY
            model_used = llm_resp.model
        except LLMError as exc:
            log.error("LLM failed: %s", exc)
            reply = FALLBACK_REPLY
            warnings.append("llm_unavailable")
            degraded = True

        # 5. Record the turn in short-term memory
        self.buffer.append(user_id, session_id, "user", message)
        self.buffer.append(user_id, session_id, "assistant", reply)
        self._last_areas[(user_id, session_id)] = query.life_areas

        # 6. Update long-term memory (never fails the response)
        changes: list[MemoryChange] = []
        if "graph_unavailable" not in warnings:
            try:
                changes = await self._update_memory(profile, session_id, message, query)
            except Exception as exc:  # noqa: BLE001 - isolation is the point here
                log.exception("memory update failed: %s", exc)
                warnings.append("memory_update_failed")

        return ChatResult(
            response=reply,
            context_used=built.context_used,
            intent=query.intent,
            life_areas=query.life_areas,
            memories_used=built.memories_used,
            memory_changes=changes,
            degraded=degraded,
            warnings=warnings,
            model=model_used,
            approx_prompt_tokens=built.approx_tokens,
            new_user=is_new,
        )

    async def _update_memory(self, profile: UserProfile, session_id: str, message: str, query: QueryUnderstanding) -> list[MemoryChange]:
        if not self.brain.policy.worth_extracting(message):
            return [MemoryChange(action="skipped", reason="message carries no first-person statement; extraction not run")]
        known = await self.brain.active_keys(profile.user_id)
        extraction = await self.extractor.extract(message, profile, known)
        if not extraction.memories and not extraction.profile_updates:
            return [MemoryChange(action="nothing_to_remember", reason="no durable, user-specific fact found in this message")]

        changes: list[MemoryChange] = []
        profile_updates = dict(extraction.profile_updates)
        candidates = list(extraction.memories)

        # A stated language preference lives in two places: as a PREFERS edge
        # (so it ranks/decays like any memory) and on the profile (so every
        # prompt honours it). Keep them in sync in both directions.
        lang = profile_updates.get("preferred_language")
        if lang and str(lang).strip().lower() == (profile.preferred_language or "").lower():
            # The model echoed the current language; that is not new information.
            profile_updates.pop("preferred_language", None)
            lang = None
        if lang and not any(c.type == MemoryType.PREFERENCE and "lang" in (c.key or "").lower() for c in candidates):
            candidates.append(MemoryCandidate(type=MemoryType.PREFERENCE, key="language", value=str(lang), confidence=0.95))

        if candidates:
            saved, mem_changes = await self.brain.remember(profile.user_id, candidates, session_id)
            changes.extend(mem_changes)
            for m in saved:
                if m.type == MemoryType.PREFERENCE and m.key == "language":
                    profile_updates["preferred_language"] = m.value

        if profile_updates:
            _, profile_changes = await self.brain.apply_profile_updates(profile, profile_updates, session_id)
            changes.extend(profile_changes)
        return changes
