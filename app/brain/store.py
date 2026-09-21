"""GraphStore protocol: the only contract the Shared Brain service depends on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.brain.models import Memory, MemoryStatus, MemoryType
from app.profile.models import UserProfile


class GraphStoreError(Exception):
    """Raised when the backing graph is unavailable or rejects an operation."""


@runtime_checkable
class GraphStore(Protocol):
    name: str

    async def init(self) -> None: ...
    async def close(self) -> None: ...
    async def ping(self) -> bool: ...

    # ---- users ----------------------------------------------------------
    async def get_user(self, user_id: str) -> UserProfile | None: ...
    async def upsert_user(self, profile: UserProfile) -> UserProfile: ...

    # ---- memories -------------------------------------------------------
    async def save_memory(self, memory: Memory) -> Memory: ...

    async def get_memories(
        self,
        user_id: str,
        *,
        life_areas: list[str] | None = None,
        types: list[MemoryType] | None = None,
        statuses: list[MemoryStatus] | None = None,
    ) -> list[Memory]: ...

    async def find_active(self, user_id: str, type_: MemoryType, key: str) -> Memory | None: ...
    async def find_active_by_key(self, user_id: str, key: str) -> list[Memory]: ...
