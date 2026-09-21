"""Query understanding: rules, LLM fallback, and graceful LLM failure."""

from __future__ import annotations

import pytest

from app.chat.intent import QueryUnderstander
from app.llm.fake import FakeLLM


@pytest.fixture
def rules_only() -> QueryUnderstander:
    return QueryUnderstander(llm=None, use_llm=False)


@pytest.mark.parametrize(
    "message, intent, areas",
    [
        ("What should I focus on for my career?", "advice", ["career"]),
        ("Will my marriage be happy?", "advice", ["relationships"]),
        ("Should I invest in property this year?", "advice", ["finance"]),
        ("What do you remember about my career goals?", "recall", ["career"]),
        ("What do you know about me?", "recall", None),  # all areas
        ("My name is Rahul. I was born on 15 August 1995 in Delhi.", "profile_update", ["general"]),
        ("hello", "smalltalk", ["general"]),
    ],
)
async def test_rules_classify_common_messages(rules_only, message, intent, areas):
    q = await rules_only.understand(message)
    assert q.intent == intent
    if areas is not None:
        assert q.life_areas == areas
    else:
        assert len(q.life_areas) >= 7


async def test_followup_inherits_previous_life_areas(rules_only):
    q = await rules_only.understand("Why do you say that?", previous_areas=["career"])
    assert q.is_followup and q.life_areas == ["career"] and q.needs_astrology is False


async def test_followup_without_history_defaults_to_general(rules_only):
    q = await rules_only.understand("Explain that again")
    assert q.is_followup and q.life_areas == ["general"]


async def test_llm_is_only_consulted_when_rules_are_unsure():
    llm = FakeLLM()
    u = QueryUnderstander(llm, use_llm=True)
    await u.understand("What should I focus on for my career?")
    assert llm.calls == []  # keyword hit, no LLM call
    q = await u.understand("Something feels off lately and I cannot place it")
    assert len(llm.calls) == 1 and q.source == "llm"


async def test_llm_failure_falls_back_to_rules():
    u = QueryUnderstander(FakeLLM(fail=True), use_llm=True)
    q = await u.understand("Something feels off lately and I cannot place it")
    assert q.intent == "advice" and q.source == "rules"
