"""Shared fixtures: an app wired with the in-memory graph and the FakeLLM."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.brain.memory_store import InMemoryGraphStore
from app.config import Settings
from app.dependencies import build_container
from app.llm.fake import FakeLLM
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake", graph_store="memory", use_llm_for_intent=True, _env_file=None)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def container(settings, store, fake_llm):
    return asyncio.run(build_container(settings, store=store, llm=fake_llm))


@pytest.fixture
def client(settings, container):
    app = create_app(settings=settings, container=container)
    with TestClient(app) as c:
        yield c


def create_rahul(client: TestClient, user_id: str = "user-123") -> dict:
    resp = client.post(
        "/users",
        json={
            "user_id": user_id,
            "name": "Rahul",
            "date_of_birth": "1995-08-15",
            "time_of_birth": "10:30",
            "birth_place": "Delhi",
            "preferred_language": "English",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def chat(client: TestClient, message: str, user_id: str = "user-123", session_id: str = "s1") -> dict:
    resp = client.post("/chat", json={"user_id": user_id, "session_id": session_id, "message": message})
    assert resp.status_code == 200, resp.text
    return resp.json()
