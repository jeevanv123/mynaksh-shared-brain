"""Tiny evaluation harness for the Shared Brain.

Runs scripts/golden.json through the in-process pipeline and reports memory
precision/recall, irrelevant-context rate, intent accuracy and persistence.
Uses whatever LLM `.env` configures; set LLM_PROVIDER=fake for an offline run.

    .venv/bin/python scripts/evaluate.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.brain.memory_store import InMemoryGraphStore  # noqa: E402
from app.config import Settings  # noqa: E402
from app.dependencies import build_container  # noqa: E402
from app.main import create_app  # noqa: E402


def main() -> int:
    golden = json.loads((Path(__file__).parent / "golden.json").read_text())
    settings = Settings(graph_store="memory")
    container = asyncio.run(build_container(settings, store=InMemoryGraphStore()))
    app = create_app(settings=settings, container=container)
    uid = f"eval-{uuid.uuid4().hex[:6]}"

    tp = fp = fn = 0
    intent_hits = intent_total = 0
    irrelevant = 0
    context_slots = 0
    policy_errors = 0
    forbid_errors = 0
    attr_hits = attr_total = 0

    with TestClient(app) as c:
        c.post("/users", json={"user_id": uid, **golden["profile"]})
        for turn in golden["turns"]:
            out = c.post("/chat", json={"user_id": uid, "session_id": turn["session"], "message": turn["message"]}).json()
            created = [u["memory"] for u in out["memory_updates"] if u["action"] == "created"]

            # memory precision / recall by key alias
            expected = turn.get("expect_memories", [])
            matched_created = set()
            for exp in expected:
                hit = next(
                    (m for i, m in enumerate(created) if i not in matched_created and m["type"] == exp["type"] and (
                        m["key"] in exp.get("keys", []) or exp.get("value_contains", "\0") in m["value"].lower()
                    )),
                    None,
                )
                if hit:
                    tp += 1
                    matched_created.add(created.index(hit))
                    for k, v in exp.get("attributes", {}).items():
                        attr_total += 1
                        attr_hits += int(hit["attributes"].get(k) == v)
                else:
                    fn += 1
            fp += len(created) - len(matched_created)
            if turn.get("expect_no_new_memory") and created:
                policy_errors += 1

            if "expect_intent" in turn:
                intent_total += 1
                intent_hits += int(out["intent"] == turn["expect_intent"])

            # context relevance: memories injected whose type is forbidden for this question
            mem_tags = [t for t in out["context_used"] if ":" in t]
            context_slots += len(mem_tags)
            for t in mem_tags:
                if t.split(":")[0] in turn.get("forbid_context_types", []):
                    irrelevant += 1
                    forbid_errors += 1
            for want in turn.get("expect_context_types", []):
                if not any(t.startswith(want + ":") for t in mem_tags):
                    print(f"  MISSING context type {want!r} for: {turn['message']}")

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    print(f"\nLLM provider: {settings.llm_provider} ({settings.llm_model})")
    print(f"memory precision      : {precision:.2f}  (tp={tp} fp={fp})")
    print(f"memory recall         : {recall:.2f}  (tp={tp} fn={fn})")
    print(f"attribute accuracy    : {(attr_hits / attr_total if attr_total else 1.0):.2f}")
    print(f"intent accuracy       : {(intent_hits / intent_total if intent_total else 1.0):.2f}")
    print(f"irrelevant-context    : {irrelevant}/{context_slots} injected memories were off-topic")
    print(f"policy violations     : {policy_errors} transient statements stored")
    ok = precision >= 0.8 and recall >= 0.8 and policy_errors == 0 and forbid_errors == 0
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
