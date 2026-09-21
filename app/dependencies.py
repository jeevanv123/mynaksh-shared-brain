"""Composition root.

Builds every service exactly once from `Settings` and exposes them through a
single `Container`. Tests build their own container with fakes; the app
builds one in its lifespan hook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.brain.memory_store import InMemoryGraphStore
from app.brain.service import SharedBrain
from app.brain.store import GraphStore, GraphStoreError
from app.chat.context_builder import ContextBuilder
from app.chat.intent import QueryUnderstander
from app.chat.service import ChatService
from app.config import Settings
from app.llm import LLMProvider, build_llm
from app.memory.extractor import MemoryExtractor
from app.memory.policy import MemoryPolicy
from app.memory.short_term import ConversationBuffer

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    store: GraphStore
    llm: LLMProvider
    brain: SharedBrain
    chat: ChatService

    async def close(self) -> None:
        await self.store.close()
        aclose = getattr(getattr(self.llm, "primary", None), "aclose", None)
        if aclose:
            await aclose()


async def build_store(settings: Settings) -> GraphStore:
    if settings.graph_store == "neo4j":
        from app.brain.neo4j_store import Neo4jGraphStore

        store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database)
        try:
            await store.init()
            log.info("connected to Neo4j at %s", settings.neo4j_uri)
            return store
        except GraphStoreError as exc:
            await store.close()
            if not settings.graph_fallback_to_memory:
                raise
            log.error("Neo4j unavailable (%s); falling back to in-memory graph store", exc)
    return InMemoryGraphStore()


async def build_container(settings: Settings, *, store: GraphStore | None = None, llm: LLMProvider | None = None) -> Container:
    store = store or await build_store(settings)
    llm = llm or build_llm(settings)
    policy = MemoryPolicy(confidence_floor=settings.memory_confidence_floor)
    brain = SharedBrain(store, policy)
    chat = ChatService(
        settings=settings,
        llm=llm,
        brain=brain,
        buffer=ConversationBuffer(max_turns=settings.short_term_max_turns),
        understander=QueryUnderstander(llm, model=settings.small_model, use_llm=settings.use_llm_for_intent),
        extractor=MemoryExtractor(llm, model=settings.small_model),
        context_builder=ContextBuilder(max_memories=settings.max_memories_in_prompt, char_budget=settings.context_char_budget),
    )
    return Container(settings=settings, store=store, llm=llm, brain=brain, chat=chat)
