"""Failure scenarios: bad input, dead LLM, dead graph, empty memory."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.brain.memory_store import InMemoryGraphStore
from app.brain.models import MemoryStatus, MemoryType
from app.brain.store import GraphStoreError
from app.config import Settings
from app.dependencies import build_container
from app.llm.base import LLMTransientError, Message
from app.llm.fake import FakeLLM
from app.llm.resilient import ResilientLLM
from app.main import create_app
from tests.conftest import chat, create_rahul


# ---- invalid input --------------------------------------------------------

@pytest.mark.parametrize(
    "payload, field",
    [
        ({"user_id": "u", "session_id": "s", "message": "   "}, "message"),
        ({"user_id": "", "session_id": "s", "message": "hi"}, "user_id"),
        ({"user_id": "u", "message": "hi"}, "session_id"),
        ({"user_id": "u", "session_id": "s", "message": "x" * 5000}, "message"),
    ],
)
def test_chat_rejects_invalid_input_with_readable_error(client, payload, field):
    r = client.post("/chat", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_input"
    assert any(d["field"] == field for d in body["details"])


def test_user_creation_rejects_future_dob(client):
    r = client.post("/users", json={"user_id": "u", "date_of_birth": "2999-01-01"})
    assert r.status_code == 422


def test_unknown_user_lookup_is_404(client):
    assert client.get("/users/nobody").status_code == 404
    assert client.get("/users/nobody/memories").status_code == 404


# ---- LLM failure ----------------------------------------------------------

def test_llm_failure_yields_graceful_fallback_and_keeps_conversation(settings, store):
    container = asyncio.run(build_container(settings, store=store, llm=FakeLLM(fail=True)))
    with TestClient(create_app(settings=settings, container=container)) as c:
        create_rahul(c)
        out = chat(c, "What should I focus on for my career?")
        assert out["degraded"] is True
        assert "llm_unavailable" in out["warnings"]
        assert "try again" in out["response"].lower()
        # The turn was still recorded so the next message has context.
        assert container.chat.buffer.recent("user-123", "s1", 10)[-1].role == "assistant"


class _FlakyLLM(FakeLLM):
    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures

    async def complete(self, messages, **kwargs):
        if self.failures > 0:
            self.failures -= 1
            raise LLMTransientError("503 from provider")
        return await super().complete(messages, **kwargs)


async def test_resilient_llm_retries_transient_errors():
    flaky = _FlakyLLM(failures=2)
    llm = ResilientLLM(flaky, max_retries=2, base_delay=0.0)
    resp = await llm.complete([Message("user", "hi")])
    assert resp.provider == "fake"


async def test_resilient_llm_gives_up_after_retries():
    flaky = _FlakyLLM(failures=5)
    llm = ResilientLLM(flaky, max_retries=1, base_delay=0.0)
    with pytest.raises(Exception):
        await llm.complete([Message("user", "hi")])


# ---- graph failure --------------------------------------------------------

class _BrokenStore(InMemoryGraphStore):
    """Works until `break_now` is set, then every call raises."""

    name = "broken"
    broken = False

    async def _guard(self):
        if self.broken:
            raise GraphStoreError("connection refused")

    async def get_user(self, user_id):
        await self._guard()
        return await super().get_user(user_id)

    async def get_memories(self, *a, **k):
        await self._guard()
        return await super().get_memories(*a, **k)

    async def save_memory(self, m):
        await self._guard()
        return await super().save_memory(m)

    async def ping(self):
        return not self.broken


def test_graph_outage_degrades_but_still_answers(settings):
    store = _BrokenStore()
    container = asyncio.run(build_container(settings, store=store, llm=FakeLLM()))
    with TestClient(create_app(settings=settings, container=container)) as c:
        create_rahul(c)
        chat(c, "I'm planning to switch jobs next year.")
        store.broken = True
        out = chat(c, "What should I focus on for my career?")
        assert out["degraded"] is True and "graph_unavailable" in out["warnings"]
        assert "goal:career_change" not in out["context_used"]
        assert out["response"]  # still a real answer
        assert c.get("/health").json()["status"] == "degraded"
        assert c.get("/users/user-123").status_code == 503


def test_boot_falls_back_to_memory_store_when_neo4j_is_down():
    s = Settings(
        llm_provider="fake",
        graph_store="neo4j",
        neo4j_uri="bolt://127.0.0.1:1",  # nothing listens here
        graph_fallback_to_memory=True,
        _env_file=None,
    )
    container = asyncio.run(build_container(s))
    assert container.store.name == "memory"


# ---- empty memory / no relevant context -----------------------------------

def test_recall_with_empty_memory_says_so(client):
    create_rahul(client)
    out = chat(client, "What do you remember about me?")
    assert "no_memories" in out["context_used"]


def test_astro_attributes_are_stored_as_graph_edges(container):
    profile = asyncio.run(container.brain.get_or_create_user("u-astro"))[0]
    profile = profile.merged_with({"date_of_birth_text": "15 August 1995"})
    asyncio.run(container.brain.save_profile(profile))
    astro = asyncio.run(container.store.get_memories("u-astro", types=[MemoryType.ASTRO], statuses=[MemoryStatus.ACTIVE]))
    assert {m.key: m.value for m in astro} == {"sun_sign": "Leo", "element": "Fire", "ruling_planet": "Sun"}
