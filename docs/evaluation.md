# Evaluating whether the Shared Brain helps

The question is not "does the model sound nice" but "does remembering the user
make answers measurably better, without dragging in noise". Six dimensions,
each with a cheap metric and how this repo already exposes the signal.

| Dimension | What we measure | Signal exposed by the API |
| --- | --- | --- |
| Memory accuracy | Precision / recall of extracted memories against a golden set per scenario | `memory_updates` on every `/chat`, `/users/{id}/memories` |
| Context relevance | Fraction of injected memories a judge marks relevant to the question | `context_used` on every `/chat` |
| Personalization | Pairwise judge preference: same question, with vs. without brain | swap `GRAPH_STORE=memory` on a fresh store for the "without" arm |
| Conversation consistency | Follow-up answers do not contradict the previous turn | `intent == followup` + judge on the pair |
| Irrelevant context | Rate of off-topic memories in `context_used` for area-specific questions | `context_used` vs. `life_areas` |
| Memory persistence | Fact stated in session A is recalled in session B (and across restarts with Neo4j) | recall scenario across `session_id`s |

## The harness in this repo

`scripts/evaluate.py` runs a small golden set (`scripts/golden.json`) through
the pipeline and reports:

- memory precision / recall (by normalized key, plus attribute match),
- irrelevant-context rate (memories in `context_used` whose life areas don't
  intersect the question's),
- persistence (recall in a new session finds every golden memory),
- policy correctness (transient statements produce no memory).

Run it offline (`LLM_PROVIDER=fake`) to gate CI on pipeline regressions, and
against the real model to measure extraction quality:

```bash
.venv/bin/python scripts/evaluate.py                # uses .env (real LLM)
LLM_PROVIDER=fake .venv/bin/python scripts/evaluate.py
```

## How I would grow this in production

1. **Golden set from logs.** Sample real conversations, have a human (or a
   strong model with human spot checks) label which facts should have been
   remembered. Track precision/recall per memory type; goals matter more than
   interests.
2. **A/B on the live path.** Randomize a small slice of traffic to
   "brain off" and compare downstream signals: thumbs up/down, follow-up rate,
   session length, retention. That is the only metric that proves the brain
   pays for its latency and tokens.
3. **LLM-as-judge for relevance and consistency**, calibrated against a human
   labelled sample, run nightly on a fixed regression set so prompt changes
   can't silently regress personalization.
4. **Memory hygiene dashboards.** Superseded/withdrawn ratio, memories never
   retrieved in 90 days (decay candidates), duplicate-key clusters
   (candidates for embedding dedupe), extraction drop reasons.
5. **Token budget vs. quality curve.** Sweep `MAX_MEMORIES_IN_PROMPT` and
   `CONTEXT_CHAR_BUDGET`, measure judge score per 1k prompt tokens to pick the
   cheapest budget that holds quality.
