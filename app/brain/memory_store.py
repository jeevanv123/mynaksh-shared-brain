"""In-memory GraphStore.

Same contract as the Neo4j store, backed by dicts. It exists for three reasons:
tests run without Docker, the service can boot when Neo4j is down (degraded
mode), and reviewers can try the API with zero setup. It mirrors the graph
shape (user -> memories -> life areas) closely enough that the retrieval logic
is identical.
"""

from __future__ import annotations

from app.brain.models import Memory, MemoryStatus, MemoryType
from app.profile.models import UserProfile


class InMemoryGraphStore:
    name = "memory"

    def __init__(self) -> None:
        self._users: dict[str, UserProfile] = {}
        self._memories: dict[str, dict[str, Memory]] = {}  # user_id -> memory_id -> Memory

    async def init(self) -> None:  # nothing to prepare
        return None

    async def close(self) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def get_user(self, user_id: str) -> UserProfile | None:
        return self._users.get(user_id)

    async def upsert_user(self, profile: UserProfile) -> UserProfile:
        self._users[profile.user_id] = profile
        self._memories.setdefault(profile.user_id, {})
        return profile

    async def save_memory(self, memory: Memory) -> Memory:
        self._memories.setdefault(memory.user_id, {})[memory.id] = memory
        return memory

    async def get_memories(self, user_id, *, life_areas=None, types=None, statuses=None) -> list[Memory]:
        statuses = statuses or [MemoryStatus.ACTIVE]
        out = []
        for m in self._memories.get(user_id, {}).values():
            if m.status not in statuses:
                continue
            if types and m.type not in types:
                continue
            if life_areas and not set(m.life_areas) & set(life_areas):
                continue
            out.append(m)
        return out

    async def find_active(self, user_id: str, type_: MemoryType, key: str) -> Memory | None:
        for m in self._memories.get(user_id, {}).values():
            if m.status == MemoryStatus.ACTIVE and m.type == type_ and m.key == key:
                return m
        return None

    async def find_active_by_key(self, user_id: str, key: str) -> list[Memory]:
        return [m for m in self._memories.get(user_id, {}).values() if m.status == MemoryStatus.ACTIVE and m.key == key]
