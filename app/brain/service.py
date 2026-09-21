"""SharedBrain: the user-intelligence layer.

Owns profile persistence, memory retrieval and the write path with conflict
resolution. Talks to a GraphStore, never to Neo4j directly.
"""

from __future__ import annotations

import logging

from app.brain.models import (
    EXCLUSIVE_SLOT_TYPES,
    Memory,
    MemoryCandidate,
    MemoryChange,
    MemoryStatus,
    MemoryType,
    normalize_key,
    now_utc,
)
from app.brain.store import GraphStore, GraphStoreError
from app.memory.policy import MemoryPolicy
from app.profile.models import UserProfile

log = logging.getLogger(__name__)


class SharedBrain:
    def __init__(self, store: GraphStore, policy: MemoryPolicy) -> None:
        self.store = store
        self.policy = policy

    # ---- profile --------------------------------------------------------

    async def get_or_create_user(self, user_id: str) -> tuple[UserProfile, bool]:
        existing = await self.store.get_user(user_id)
        if existing:
            return existing, False
        profile = await self.store.upsert_user(UserProfile(user_id=user_id))
        return profile, True

    async def save_profile(self, profile: UserProfile) -> UserProfile:
        saved = await self.store.upsert_user(profile)
        await self._sync_astro_memories(saved)
        return saved

    async def apply_profile_updates(self, profile: UserProfile, updates: dict, session_id: str | None) -> tuple[UserProfile, list[MemoryChange]]:
        if not updates:
            return profile, []
        merged = profile.merged_with(updates)
        if merged.model_dump(exclude={"updated_at"}) == profile.model_dump(exclude={"updated_at"}):
            return profile, []
        saved = await self.save_profile(merged)
        changed_fields = sorted("date_of_birth" if k == "date_of_birth_text" else k for k in updates)
        return saved, [MemoryChange(action="profile_updated", reason=", ".join(changed_fields))]

    async def _sync_astro_memories(self, profile: UserProfile) -> None:
        """Mirror derived astrology into the graph as HAS_ATTRIBUTE edges."""
        astro = profile.astro
        if not astro:
            return
        for key, value in (("sun_sign", astro.sun_sign), ("element", astro.element), ("ruling_planet", astro.ruling_planet)):
            cand = MemoryCandidate(type=MemoryType.ASTRO, key=key, value=value, life_areas=["spirituality", "general"], confidence=1.0)
            await self._upsert_one(profile.user_id, cand, session_id=None, importance=0.7)

    # ---- read path ------------------------------------------------------

    async def relevant_memories(self, user_id: str, life_areas: list[str] | None, *, include_astro: bool = False) -> list[Memory]:
        types = [t for t in MemoryType if t != MemoryType.ASTRO] if not include_astro else None
        return await self.store.get_memories(user_id, life_areas=life_areas, types=types)

    async def all_memories(self, user_id: str, *, include_inactive: bool = False) -> list[Memory]:
        statuses = list(MemoryStatus) if include_inactive else [MemoryStatus.ACTIVE]
        return await self.store.get_memories(user_id, statuses=statuses)

    async def active_keys(self, user_id: str) -> list[str]:
        return sorted({m.key for m in await self.store.get_memories(user_id) if m.type != MemoryType.ASTRO})

    # ---- write path -----------------------------------------------------

    async def remember(self, user_id: str, candidates: list[MemoryCandidate], session_id: str | None) -> tuple[list[Memory], list[MemoryChange]]:
        """Run candidates through policy, then upsert with conflict resolution."""
        saved: list[Memory] = []
        changes: list[MemoryChange] = []
        for raw in candidates:
            cand = self.policy.normalize(raw)
            decision = self.policy.evaluate(cand)
            if not decision.keep:
                changes.append(MemoryChange(action="dropped", reason=f"{cand.value!r}: {decision.reason}"))
                continue
            try:
                result = await self._upsert_one(user_id, cand, session_id, importance=self.policy.importance_for(cand))
            except GraphStoreError as exc:
                log.error("memory write failed: %s", exc)
                changes.append(MemoryChange(action="dropped", reason=f"{cand.value!r}: graph unavailable"))
                continue
            changes.extend(result)
            saved.extend(c.memory for c in result if c.memory and c.action in ("created", "confirmed", "superseded"))
        return saved, changes

    async def _upsert_one(self, user_id: str, cand: MemoryCandidate, session_id: str | None, *, importance: float) -> list[MemoryChange]:
        changes: list[MemoryChange] = []
        key = cand.resolved_key()
        now = now_utc()

        # Explicit replacement / withdrawal named by the extractor.
        if cand.replaces:
            for old in await self.store.find_active_by_key(user_id, normalize_key(cand.replaces)):
                old.status = MemoryStatus.WITHDRAWN if cand.status == "withdrawn" else MemoryStatus.SUPERSEDED
                old.updated_at = now
                await self.store.save_memory(old)
                changes.append(MemoryChange(action=old.status.value, memory=old))
        if cand.status == "withdrawn":
            if not changes:
                # Nothing matched `replaces`; try the candidate's own key.
                for old in await self.store.find_active_by_key(user_id, key):
                    old.status = MemoryStatus.WITHDRAWN
                    old.updated_at = now
                    await self.store.save_memory(old)
                    changes.append(MemoryChange(action="withdrawn", memory=old))
            return changes or [MemoryChange(action="dropped", reason=f"{cand.value!r}: withdrawal matched no active memory")]

        existing = await self.store.find_active(user_id, cand.type, key)
        if existing and existing.value.strip().lower() == cand.value.strip().lower():
            # Same fact repeated: reinforce, merge attributes, refresh recency.
            existing.mention_count += 1
            existing.confidence = min(0.99, max(existing.confidence, cand.confidence) + 0.05)
            existing.attributes = {**existing.attributes, **cand.attributes}
            existing.life_areas = sorted(set(existing.life_areas) | set(cand.life_areas))
            existing.last_confirmed_at = existing.updated_at = now
            await self.store.save_memory(existing)
            changes.append(MemoryChange(action="confirmed", memory=existing))
            return changes

        new = Memory(
            user_id=user_id,
            type=cand.type,
            key=key,
            value=cand.value,
            attributes=cand.attributes,
            life_areas=cand.life_areas,
            confidence=cand.confidence,
            importance=importance,
            evidence=cand.evidence,
            source_session_id=session_id,
        )
        if existing and cand.type in EXCLUSIVE_SLOT_TYPES:
            # Same slot, different value: the user corrected themselves.
            existing.status = MemoryStatus.SUPERSEDED
            existing.superseded_by = new.id
            existing.updated_at = now
            await self.store.save_memory(existing)
            changes.append(MemoryChange(action="superseded", memory=existing))
        await self.store.save_memory(new)
        changes.append(MemoryChange(action="created", memory=new))
        return changes
