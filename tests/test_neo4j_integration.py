"""Runs the core flow against a real Neo4j. Skipped unless RUN_NEO4J_TESTS=1.

    docker compose up -d neo4j
    RUN_NEO4J_TESTS=1 pytest tests/test_neo4j_integration.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import chat

pytestmark = pytest.mark.skipif(os.getenv("RUN_NEO4J_TESTS") != "1", reason="set RUN_NEO4J_TESTS=1 with Neo4j on localhost:7687")


@pytest.fixture
def neo4j_client():
    # Let the app's lifespan build the container, so the async driver is bound
    # to the same event loop the test client uses for requests.
    settings = Settings(llm_provider="fake", graph_store="neo4j", graph_fallback_to_memory=False, _env_file=None)
    with TestClient(create_app(settings=settings)) as c:
        yield c, settings


def test_full_flow_persists_in_neo4j(neo4j_client):
    c, settings = neo4j_client
    uid = f"it-{uuid.uuid4().hex[:8]}"
    assert c.get("/health").json() == {
        "status": "ok",
        "graph_store": "neo4j",
        "graph_ok": True,
        "llm_provider": "fake",
        "llm_model": settings.llm_model,
    }

    r = c.post("/users", json={"user_id": uid, "name": "Rahul", "date_of_birth": "1995-08-15", "birth_place": "Delhi"})
    assert r.status_code == 201 and r.json()["astrology"]["sun_sign"] == "Leo"

    chat(c, "I'm planning to switch jobs next year.", user_id=uid, session_id="s1")
    out = chat(c, "What should I focus on for my career?", user_id=uid, session_id="s2")
    assert "goal:career_change" in out["context_used"]

    # Correction path through Cypher: supersede language preference
    chat(c, "I prefer Hindi replies.", user_id=uid, session_id="s2")
    chat(c, "Actually I prefer English.", user_id=uid, session_id="s2")
    mems = c.get(f"/users/{uid}/memories?include_inactive=true").json()["memories"]
    langs = {m["status"]: m["value"] for m in mems if m["key"] == "language"}
    assert langs == {"active": "English", "superseded": "Hindi"}

    # Irrelevant memory is not selected for a relationships question
    out = chat(c, "Will my marriage be happy?", user_id=uid, session_id="s3")
    assert "goal:career_change" not in out["context_used"]

    # Graph shape, checked with the sync driver: shared concept node -> LifeArea
    from neo4j import GraphDatabase

    with GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)) as driver:
        records, _, _ = driver.execute_query(
            "MATCH (u:User {user_id:$uid})-[r:HAS_GOAL]->(g:Goal)-[:RELATES_TO]->(la:LifeArea) "
            "RETURN g.key AS key, r.status AS status, collect(la.name) AS areas",
            uid=uid,
            database_=settings.neo4j_database,
        )
        assert records and records[0]["key"] == "career_change" and "career" in records[0]["areas"]
        assert records[0]["status"] == "active"
