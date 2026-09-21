"""Neo4j GraphStore.

Graph shape (see docs/design.md):

    (:User {user_id, ...profile})
    (:User)-[:HAS_GOAL|PREFERS|INTERESTED_IN|CARES_ABOUT|HAS_ATTRIBUTE|REMEMBERS|EXPERIENCED {memory props}]->(:Concept {key, value})
    (:Concept)-[:RELATES_TO]->(:LifeArea {name})

Concept nodes are shared across users; everything user-specific sits on the
edge. Relationship types and labels come from fixed enums, which is why they
can be interpolated into Cypher strings safely.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.brain.models import LABEL_FOR_TYPE, REL_FOR_TYPE, TYPE_FOR_REL, Memory, MemoryStatus, MemoryType
from app.brain.store import GraphStoreError
from app.profile.models import UserProfile

log = logging.getLogger(__name__)

_MEMORY_RELS = "|".join(REL_FOR_TYPE.values())


class Neo4jGraphStore:
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        # Notifications off: the union of relationship types in retrieval queries
        # legitimately includes types no user has created yet, and Neo4j warns
        # about each one on every query.
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF")
        self._db = database

    async def _run(self, query: str, **params):
        try:
            result = await self._driver.execute_query(query, params, database_=self._db)
            return result.records
        except (ServiceUnavailable, Neo4jError, OSError) as exc:
            raise GraphStoreError(f"neo4j: {exc}") from exc

    async def init(self) -> None:
        await self._run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
        await self._run("CREATE CONSTRAINT life_area_name IF NOT EXISTS FOR (l:LifeArea) REQUIRE l.name IS UNIQUE")
        for label in set(LABEL_FOR_TYPE.values()) - {"LifeArea"}:
            await self._run(f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (c:{label}) REQUIRE c.key IS UNIQUE")

    async def close(self) -> None:
        await self._driver.close()

    async def ping(self) -> bool:
        try:
            await self._run("RETURN 1")
            return True
        except GraphStoreError:
            return False

    # ---- users ----------------------------------------------------------

    async def get_user(self, user_id: str) -> UserProfile | None:
        records = await self._run("MATCH (u:User {user_id: $user_id}) RETURN u", user_id=user_id)
        if not records:
            return None
        return _user_from_node(dict(records[0]["u"]))

    async def upsert_user(self, profile: UserProfile) -> UserProfile:
        props = {
            "name": profile.name,
            "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
            "time_of_birth": profile.time_of_birth.strftime("%H:%M") if profile.time_of_birth else None,
            "birth_place": profile.birth_place,
            "preferred_language": profile.preferred_language,
            "sun_sign": profile.astro.sun_sign if profile.astro else None,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }
        await self._run(
            "MERGE (u:User {user_id: $user_id}) SET u += $props",
            user_id=profile.user_id,
            props={k: v for k, v in props.items() if v is not None},
        )
        return profile

    # ---- memories -------------------------------------------------------

    async def save_memory(self, memory: Memory) -> Memory:
        label = memory.label
        rel = memory.relationship
        concept_match = "{name: $key}" if label == "LifeArea" else "{key: $key}"
        query = f"""
        MATCH (u:User {{user_id: $user_id}})
        MERGE (c:{label} {concept_match})
        SET c.value = $value
        MERGE (u)-[r:{rel} {{memory_id: $memory_id}}]->(c)
        SET r += $props
        WITH c
        UNWIND $life_areas AS area
        MERGE (la:LifeArea {{name: area}})
        MERGE (c)-[:RELATES_TO]->(la)
        """
        await self._run(
            query,
            user_id=memory.user_id,
            key=memory.key,
            value=memory.value,
            memory_id=memory.id,
            life_areas=memory.life_areas,
            props={
                "key": memory.key,
                "value": memory.value,
                "attributes": json.dumps(memory.attributes),
                "life_areas": memory.life_areas,
                "confidence": memory.confidence,
                "importance": memory.importance,
                "status": memory.status.value,
                "evidence": memory.evidence,
                "source_session_id": memory.source_session_id,
                "mention_count": memory.mention_count,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
                "last_confirmed_at": memory.last_confirmed_at.isoformat(),
                "superseded_by": memory.superseded_by,
            },
        )
        return memory

    async def get_memories(self, user_id, *, life_areas=None, types=None, statuses=None) -> list[Memory]:
        statuses = [s.value for s in (statuses or [MemoryStatus.ACTIVE])]
        rel_types = [REL_FOR_TYPE[t] for t in types] if types else list(REL_FOR_TYPE.values())
        if life_areas:
            query = f"""
            MATCH (u:User {{user_id: $user_id}})-[r:{_MEMORY_RELS}]->(c)-[:RELATES_TO]->(la:LifeArea)
            WHERE r.status IN $statuses AND type(r) IN $rel_types AND la.name IN $areas
            RETURN DISTINCT type(r) AS rel, r
            """
        else:
            query = f"""
            MATCH (u:User {{user_id: $user_id}})-[r:{_MEMORY_RELS}]->(c)
            WHERE r.status IN $statuses AND type(r) IN $rel_types
            RETURN type(r) AS rel, r
            """
        records = await self._run(query, user_id=user_id, statuses=statuses, rel_types=rel_types, areas=life_areas or [])
        return [_memory_from_rel(user_id, rec["rel"], dict(rec["r"])) for rec in records]

    async def find_active(self, user_id: str, type_: MemoryType, key: str) -> Memory | None:
        rel = REL_FOR_TYPE[type_]
        records = await self._run(
            f"MATCH (u:User {{user_id: $user_id}})-[r:{rel}]->(c) WHERE r.key = $key AND r.status = 'active' RETURN type(r) AS rel, r LIMIT 1",
            user_id=user_id,
            key=key,
        )
        return _memory_from_rel(user_id, records[0]["rel"], dict(records[0]["r"])) if records else None

    async def find_active_by_key(self, user_id: str, key: str) -> list[Memory]:
        records = await self._run(
            f"MATCH (u:User {{user_id: $user_id}})-[r:{_MEMORY_RELS}]->(c) WHERE r.key = $key AND r.status = 'active' RETURN type(r) AS rel, r",
            user_id=user_id,
            key=key,
        )
        return [_memory_from_rel(user_id, rec["rel"], dict(rec["r"])) for rec in records]


def _user_from_node(props: dict) -> UserProfile:
    return UserProfile(
        user_id=props["user_id"],
        name=props.get("name"),
        date_of_birth=props.get("date_of_birth"),
        time_of_birth=props.get("time_of_birth"),
        birth_place=props.get("birth_place"),
        preferred_language=props.get("preferred_language") or "English",
        created_at=_dt(props.get("created_at")),
        updated_at=_dt(props.get("updated_at")),
    )


def _memory_from_rel(user_id: str, rel: str, props: dict) -> Memory:
    return Memory(
        id=props["memory_id"],
        user_id=user_id,
        type=TYPE_FOR_REL[rel],
        key=props["key"],
        value=props["value"],
        attributes=json.loads(props.get("attributes") or "{}"),
        life_areas=list(props.get("life_areas") or ["general"]),
        confidence=props.get("confidence", 0.7),
        importance=props.get("importance", 0.6),
        status=MemoryStatus(props.get("status", "active")),
        evidence=props.get("evidence"),
        source_session_id=props.get("source_session_id"),
        mention_count=props.get("mention_count", 1),
        created_at=_dt(props.get("created_at")),
        updated_at=_dt(props.get("updated_at")),
        last_confirmed_at=_dt(props.get("last_confirmed_at")),
        superseded_by=props.get("superseded_by"),
    )


def _dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    from app.brain.models import now_utc

    return now_utc()
