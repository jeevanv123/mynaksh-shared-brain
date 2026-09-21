"""The scenario suite the assignment asks for, end to end through the HTTP API.

Runs against the in-memory graph store and the deterministic FakeLLM, so it
is fast and needs no network. The assertions target pipeline behaviour
(what was selected, what was stored) rather than the model's prose.
"""

from __future__ import annotations

from tests.conftest import chat, create_rahul


def _active(client, user_id="user-123"):
    return {m["key"]: m for m in client.get(f"/users/{user_id}/memories").json()["memories"]}


# 1. New user ---------------------------------------------------------------

def test_new_user_gets_answer_and_is_asked_for_birth_details(client):
    out = chat(client, "What does my week look like?", user_id="fresh-user")
    assert out["context_used"] == ["profile_missing"]
    assert out["degraded"] is False
    assert client.get("/users/fresh-user").status_code == 200  # auto-created


def test_create_user_computes_sun_sign(client):
    user = create_rahul(client)
    assert user["astrology"]["sun_sign"] == "Leo"
    assert user["missing_fields"] == []


# 2. Creating a long-term memory -------------------------------------------

def test_statement_of_plan_becomes_goal_memory(client):
    create_rahul(client)
    out = chat(client, "My name is Rahul. I was born on 15 August 1995 in Delhi. I'm planning to switch jobs next year.")
    created = [u for u in out["memory_updates"] if u["action"] == "created"]
    assert created and created[0]["memory"]["key"] == "career_change"
    assert created[0]["memory"]["relationship"] == "HAS_GOAL"
    assert created[0]["memory"]["attributes"]["target_year"] >= 2027
    assert "career" in created[0]["memory"]["life_areas"]


def test_profile_facts_in_chat_update_profile(client):
    chat(client, "My name is Rahul. I was born on 15 August 1995 in Delhi.", user_id="u-chatty")
    user = client.get("/users/u-chatty").json()
    assert user["name"] == "Rahul"
    assert user["date_of_birth"] == "1995-08-15"
    assert user["astrology"]["sun_sign"] == "Leo"


# 3. Retrieving a memory ----------------------------------------------------

def test_career_question_pulls_career_goal_into_context(client):
    create_rahul(client)
    chat(client, "I'm planning to switch jobs next year.")
    out = chat(client, "What should I focus on for my career?")
    assert out["intent"] == "advice"
    assert "goal:career_change" in out["context_used"]
    assert "user_profile" in out["context_used"]
    assert "astrology" in out["context_used"]
    assert "Career Change" in out["response"]  # FakeLLM echoes what reached the prompt


# 4. Follow-up question -----------------------------------------------------

def test_followup_uses_recent_conversation(client):
    create_rahul(client)
    chat(client, "I'm planning to switch jobs next year.")
    chat(client, "What should I focus on for my career?")
    out = chat(client, "Why do you say that?")
    assert out["intent"] == "followup"
    assert out["life_areas"] == ["career"]  # inherited from the previous turn
    assert "recent_conversation" in out["context_used"]
    assert any(u["action"] == "skipped" for u in out["memory_updates"])  # no extraction call for a bare question


# 5. New session using previous information ---------------------------------

def test_new_session_recalls_long_term_memory(client):
    create_rahul(client)
    chat(client, "I'm planning to switch jobs next year.", session_id="s1")
    out = chat(client, "What do you remember about my career goals?", session_id="s2")
    assert out["intent"] == "recall"
    assert "goal:career_change" in out["context_used"]
    assert "recent_conversation" not in out["context_used"]  # nothing from s1 leaks in


# 6. Irrelevant memory is not injected --------------------------------------

def test_irrelevant_memory_is_left_out(client):
    create_rahul(client)
    chat(client, "I'm planning to switch jobs next year.")
    out = chat(client, "Will my marriage be happy?")
    assert out["life_areas"] == ["relationships"]
    assert "goal:career_change" not in out["context_used"]


def test_transient_mood_is_not_remembered(client):
    create_rahul(client)
    out = chat(client, "I'm feeling tired today.")
    assert all(u["action"] in ("nothing_to_remember", "dropped") for u in out["memory_updates"])
    assert "tired" not in " ".join(_active(client))


# 7. User correcting existing information -----------------------------------

def test_withdrawing_a_goal_marks_it_withdrawn(client):
    create_rahul(client)
    chat(client, "I'm planning to switch jobs next year.")
    out = chat(client, "Actually, I'm no longer planning to switch jobs.")
    assert any(u["action"] == "withdrawn" and u["memory"]["key"] == "career_change" for u in out["memory_updates"])
    assert "career_change" not in _active(client)
    everything = client.get("/users/user-123/memories?include_inactive=true").json()["memories"]
    assert any(m["key"] == "career_change" and m["status"] == "withdrawn" for m in everything)


def test_changing_a_preference_supersedes_the_old_value(client):
    create_rahul(client)
    chat(client, "I prefer Hindi replies.")
    assert client.get("/users/user-123").json()["preferred_language"] == "Hindi"
    out = chat(client, "Actually I prefer English.")
    actions = {u["action"]: u["memory"]["value"] for u in out["memory_updates"] if u["memory"]}
    assert actions.get("superseded") == "Hindi"
    assert actions.get("created") == "English"
    assert _active(client)["language"]["value"] == "English"
    assert client.get("/users/user-123").json()["preferred_language"] == "English"


def test_repeating_a_fact_reinforces_instead_of_duplicating(client):
    create_rahul(client)
    chat(client, "I'm planning to switch jobs next year.")
    out = chat(client, "Like I said, I'm planning to switch jobs next year.")
    assert any(u["action"] == "confirmed" for u in out["memory_updates"])
    goals = [m for m in client.get("/users/user-123/memories").json()["memories"] if m["key"] == "career_change"]
    assert len(goals) == 1 and goals[0]["mention_count"] == 2


# 8. Missing user information -----------------------------------------------

def test_missing_profile_is_reported_not_invented(client):
    client.post("/users", json={"user_id": "u-min", "name": "Asha"})
    out = chat(client, "What should I focus on for my career?", user_id="u-min")
    assert "user_profile" in out["context_used"]
    assert "astrology" not in out["context_used"]  # no DOB, so no sign was fabricated
    assert client.get("/users/u-min").json()["missing_fields"] == ["date_of_birth", "time_of_birth", "birth_place"]
