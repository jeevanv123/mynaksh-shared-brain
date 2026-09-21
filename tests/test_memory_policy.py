"""Unit tests for the keep/drop policy and the extraction parser."""

from __future__ import annotations

import pytest

from app.brain.models import MemoryCandidate, MemoryType
from app.memory.extractor import parse_extraction
from app.memory.policy import MemoryPolicy


@pytest.fixture
def policy() -> MemoryPolicy:
    return MemoryPolicy(confidence_floor=0.55)


@pytest.mark.parametrize(
    "message, expected",
    [
        ("What should I focus on in my career?", False),  # pure question
        ("Why do you say that?", False),
        ("How should I plan my finances this year?", False),  # "I plan" inside a question is not a plan
        ("ok", False),
        ("I'm planning to switch jobs next year.", True),
        ("My wedding is in December.", True),
        ("Should I take the job? I have an offer from Google.", True),  # question, but carries a fact
    ],
)
def test_worth_extracting_gate(policy, message, expected):
    assert policy.worth_extracting(message) is expected


def test_keeps_confident_persistent_goal(policy):
    c = MemoryCandidate(type=MemoryType.GOAL, value="Career Change", confidence=0.9)
    assert policy.evaluate(c).keep


def test_drops_low_confidence(policy):
    c = MemoryCandidate(type=MemoryType.GOAL, value="Career Change", confidence=0.3)
    d = policy.evaluate(c)
    assert not d.keep and "confidence" in d.reason


def test_drops_non_persistent(policy):
    c = MemoryCandidate(type=MemoryType.FACT, value="Feeling tired", persistent=False, confidence=0.9)
    assert not policy.evaluate(c).keep


def test_drops_transient_state_even_if_marked_persistent(policy):
    c = MemoryCandidate(type=MemoryType.FACT, value="Has a headache today", confidence=0.9)
    d = policy.evaluate(c)
    assert not d.keep and "transient" in d.reason


def test_drops_questions_and_transcripts(policy):
    assert not policy.evaluate(MemoryCandidate(type=MemoryType.FACT, value="Should I move to Pune?", confidence=0.9)).keep
    assert not policy.evaluate(MemoryCandidate(type=MemoryType.FACT, value="x" * 200, confidence=0.9)).keep


def test_system_derived_types_are_not_accepted_from_chat(policy):
    assert not policy.evaluate(MemoryCandidate(type=MemoryType.ASTRO, value="Leo", confidence=1.0)).keep


def test_withdrawals_always_pass(policy):
    c = MemoryCandidate(type=MemoryType.GOAL, value="Career Change", confidence=0.2, status="withdrawn", replaces="career_change")
    assert policy.evaluate(c).keep


def test_normalize_filters_unknown_life_areas(policy):
    c = policy.normalize(MemoryCandidate(type=MemoryType.GOAL, value=" Career Change ", life_areas=["career", "bogus"]))
    assert c.life_areas == ["career"] and c.value == "Career Change"


@pytest.mark.parametrize("key, value", [("preferred_language", "hindi"), ("reply_language", "Hindi"), (None, "hindi"), ("language", "HINDI")])
def test_language_preferences_share_one_slot(policy, key, value):
    c = policy.normalize(MemoryCandidate(type=MemoryType.PREFERENCE, key=key, value=value))
    assert (c.key, c.value) == ("language", "Hindi")


@pytest.mark.parametrize("key", ["name", "date_of_birth", "dob", "birth_place", "birthplace"])
def test_profile_fields_are_not_stored_as_facts(policy, key):
    d = policy.evaluate(MemoryCandidate(type=MemoryType.FACT, key=key, value="Lucknow", confidence=0.95))
    assert not d.keep and "profile field" in d.reason


def test_importance_ranks_goals_above_interests(policy):
    goal = MemoryCandidate(type=MemoryType.GOAL, value="Career Change")
    interest = MemoryCandidate(type=MemoryType.INTEREST, value="Chess")
    assert policy.importance_for(goal) > policy.importance_for(interest)


# ---- extraction parsing -------------------------------------------------------

def test_parse_extraction_tolerates_code_fences_and_bad_items():
    text = """```json
    {"memories": [
        {"type": "GOAL", "value": "Career Change", "attributes": {"target_year": 2027}, "life_areas": ["career"], "confidence": 0.9, "persistent": true},
        {"type": "goal", "value": ""},
        "garbage"
    ], "profile_updates": {"name": "Rahul", "birth_place": null}}
    ```"""
    result = parse_extraction(text)
    assert len(result.memories) == 1
    assert result.memories[0].type == MemoryType.GOAL
    assert result.memories[0].resolved_key() == "career_change"
    assert result.profile_updates == {"name": "Rahul"}


def test_parse_extraction_on_garbage_returns_empty():
    assert parse_extraction("Sorry, I cannot do that.").memories == []
