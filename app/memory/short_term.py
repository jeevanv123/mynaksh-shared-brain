"""Short-term conversation context.

One bounded buffer per session. This is deliberately NOT the graph: turns are
ephemeral, high-volume and only meaningful in order. In production this is a
Redis list with a TTL; the interface stays the same.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationBuffer:
    def __init__(self, max_turns: int = 20) -> None:
        self._max = max_turns
        self._sessions: dict[tuple[str, str], deque[Turn]] = {}

    def append(self, user_id: str, session_id: str, role: str, content: str) -> None:
        key = (user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = deque(maxlen=self._max)
        self._sessions[key].append(Turn(role=role, content=content))

    def recent(self, user_id: str, session_id: str, n: int) -> list[Turn]:
        turns = self._sessions.get((user_id, session_id))
        if not turns:
            return []
        return list(turns)[-n:]

    def last_assistant(self, user_id: str, session_id: str) -> Turn | None:
        for t in reversed(self._sessions.get((user_id, session_id), ())):
            if t.role == "assistant":
                return t
        return None

    def clear(self, user_id: str, session_id: str) -> None:
        self._sessions.pop((user_id, session_id), None)
