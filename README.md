# MyNaksh Shared Brain — Personalized Astrology Chat

A FastAPI service that answers astrology-flavoured questions **and remembers the
user**: a graph-backed *Shared Brain* (Neo4j) holds durable facts, a short-term
buffer holds the conversation, and a context-selection step decides what the
LLM actually sees on every turn.

```
message ─► understand query ─► select relevant context ─► build prompt ─► LLM ─► reply ─► update memory
              (rules + LLM)      (graph by life area)     (budgeted)                    (extract → policy → upsert)
```

- **Runs with zero setup** (`LLM_PROVIDER=fake`, in-memory graph) or with a real
  LLM + Neo4j via Docker.
- **Every response explains itself**: `context_used` lists exactly what reached
  the prompt, `memory_updates` lists what was learned, confirmed, superseded,
  withdrawn or dropped, and why.
- 69 automated tests, including the 8 scenario types the brief asks for and a
  real-Neo4j integration test.

---

## 1. Quick start

```bash
git clone <this repo> && cd mynaksh-shared-brain
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

# A) offline: no key, no Docker
make run-offline                 # fake LLM + in-memory graph on :8000

# B) real LLM + Neo4j
cp .env.example .env             # set LLM_API_KEY (+ LLM_BASE_URL for a gateway)
make neo4j-up                    # docker compose up -d neo4j
make run

# tests
make test                        # 68 unit/scenario tests, no network
make test-integration            # +1 test against the running Neo4j
bash scripts/demo.sh             # scripted walkthrough against :8000
```

Swagger UI: http://localhost:8000/docs · Neo4j browser: http://localhost:7474 (neo4j / password)

The LLM adapter is OpenAI-compatible, so `LLM_BASE_URL` can point at OpenAI,
Azure via a gateway, Gemini's OpenAI endpoint, Ollama or vLLM. A native
Anthropic adapter is included (`LLM_PROVIDER=anthropic`).

---

## 2. API

### `POST /users` — create or update a profile
```json
{"user_id": "user-123", "name": "Rahul", "date_of_birth": "1995-08-15",
 "time_of_birth": "10:30", "birth_place": "Delhi", "preferred_language": "English"}
```
→ `201` with the profile, derived astrology (`sun_sign: Leo, element: Fire, ruling_planet: Sun`)
and `missing_fields`. Profile facts can also arrive through chat ("My name is Rahul, I was born on…").

### `POST /chat`
```json
{"user_id": "user-123", "session_id": "session-456", "message": "What should I focus on for my career?"}
```
Real response (gpt-4.1-mini through a gateway, after the user said *"I'm planning to switch jobs next year. I'm also preparing for a product management interview next month."*):
```json
{
  "response": "Rahul, as a Leo with a strong Sun influence, your career focus should naturally align with roles where you can lead, innovate, and gain recognition. Product management fits well because it combines leadership, strategic thinking, and the opportunity to make a visible impact. ... Since you aim to switch jobs in 2027, use the next few years to build a strong portfolio ...",
  "user_id": "user-123",
  "session_id": "session-456",
  "context_used": ["user_profile", "astrology", "goal:product_management_interview_preparation", "goal:job_switch", "recent_conversation"],
  "intent": "advice",
  "life_areas": ["career"],
  "memory_updates": [{"action": "skipped", "reason": "message carries no first-person statement; extraction not run"}],
  "degraded": false,
  "warnings": [],
  "model": "gpt-4.1-mini-2025-04-14",
  "approx_prompt_tokens": 581
}
```

The turn *before* it, which created the memories, returned:
```json
"memory_updates": [
  {"action": "created", "memory": {"type": "goal", "relationship": "HAS_GOAL", "key": "job_switch", "value": "Job Switch",
                                   "attributes": {"target_year": 2027}, "life_areas": ["career"], "confidence": 0.95, "status": "active"}},
  {"action": "created", "memory": {"type": "goal", "key": "product_management_interview_preparation",
                                   "attributes": {"timeframe": "next month"}, "life_areas": ["career"]}}
]
```

Follow-up *"Why do you say that?"* → `intent: followup`, uses `recent_conversation`, extraction skipped.
New session *"What do you remember about my career goals?"* → `intent: recall`, answers only from the graph:
> "I remember that you have two main career-related goals: 1. You are planning a job switch targeted for the year 2027. 2. You are preparing for product management interviews within the next month."

Correction *"Actually, I've decided not to switch jobs anymore. I'll stay and aim for a promotion instead."* →
`memory_updates: [withdrawn: Job Switch, created: Aim for a promotion]`.

Hindi *"Mujhe Hindi mein jawab do please. Aur main Bangalore shift ho gaya hoon."* →
`created: Bangalore (current_city)`, `created: Hindi (language)`, `profile_updated: preferred_language`; the next reply is in Hindi.

### Other endpoints
| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/users/{id}` | profile + derived astrology + missing fields |
| `GET` | `/users/{id}/memories?include_inactive=true` | what the brain holds, ranked; inactive shows superseded/withdrawn history |
| `GET` | `/health` | graph store in use and whether it is reachable |

---

## 3. Architecture

```
app/
├── api/         routes.py, schemas.py          HTTP contracts, validation, error mapping
├── chat/        intent.py                       QueryUnderstander: rules first, LLM when unsure
│                context_builder.py              select + rank + budget → prompt; reports context_used
│                service.py                      ChatService: the pipeline, each stage degrades independently
│                prompts.py                      all prompt text, versioned in one place
├── memory/      short_term.py                   ConversationBuffer (per session, bounded)
│                extractor.py                    LLM → strict JSON candidates (tolerant parser)
│                policy.py                       keep / drop / normalize rules, importance
├── brain/       models.py                       Memory, MemoryCandidate, types ↔ relationships
│                store.py                        GraphStore protocol
│                neo4j_store.py                  Cypher implementation
│                memory_store.py                 dict implementation (tests, degraded mode)
│                service.py                      SharedBrain: retrieval, upsert, conflict resolution
├── profile/     models.py, astrology.py         UserProfile, date parsing, stubbed sun-sign engine
├── llm/         base.py                         LLMProvider protocol, error taxonomy
│                openai_compat.py, anthropic_provider.py, fake.py, resilient.py (retry + fallback)
├── config.py    pydantic-settings; the only place that reads env
├── dependencies.py                              composition root (Container)
└── main.py      FastAPI factory, exception handlers
```

Every cross-layer dependency is a `Protocol`. Tests swap Neo4j for the dict
store and the LLM for a deterministic fake without touching the pipeline.

---

## 4. Shared Brain schema (Neo4j)

```
(:User {user_id, name, date_of_birth, time_of_birth, birth_place, preferred_language, sun_sign, created_at, updated_at})

(:User)-[:HAS_GOAL      ]->(:Goal           {key, value})     e.g. career_change / "Career Change"
(:User)-[:PREFERS       ]->(:Preference     {key, value})     e.g. language / "Hindi"
(:User)-[:INTERESTED_IN ]->(:Interest       {key, value})     e.g. entrepreneurship
(:User)-[:CARES_ABOUT   ]->(:LifeArea       {name})
(:User)-[:HAS_ATTRIBUTE ]->(:AstroAttribute {key, value})     sun_sign / Leo, element / Fire, ruling_planet / Sun
(:User)-[:REMEMBERS     ]->(:Fact           {key, value})     e.g. current_city / "Bangalore"
(:User)-[:EXPERIENCED   ]->(:Event          {key, value})     e.g. wedding (timeframe=December)

(:Goal|Preference|Interest|Fact|Event|AstroAttribute)-[:RELATES_TO]->(:LifeArea {name})
      life areas: career, relationships, finance, health, education, family, spirituality, general
```

**Every memory edge carries:** `memory_id, key, value, attributes (JSON), life_areas, confidence, importance,
status (active|superseded|withdrawn), evidence, source_session_id, mention_count, created_at, updated_at,
last_confirmed_at, superseded_by`.

**Why concept nodes are shared and facts live on the edge.** "Career Change" is one
node that many users point at. That keeps the graph small, makes cross-user
questions one hop ("how many Leos are planning a career change?"), and keeps
per-user retrieval a two-hop traversal `User → concept → LifeArea`. Everything
that is *about this user* (confidence, status, when they said it) belongs on
the relationship, so two users can hold the same concept with different
confidence and lifecycle.

**Retrieval.**
```cypher
MATCH (u:User {user_id:$uid})-[r:HAS_GOAL|PREFERS|INTERESTED_IN|CARES_ABOUT|HAS_ATTRIBUTE|REMEMBERS|EXPERIENCED]->(c)
      -[:RELATES_TO]->(la:LifeArea)
WHERE r.status = 'active' AND la.name IN $areas
RETURN DISTINCT type(r), r
```
Results are ranked in Python by `importance × confidence × recency` (half-life
180 days; goals whose `target_year` has passed are penalised) and cut to
`MAX_MEMORIES_IN_PROMPT`. Constraints: unique `User.user_id`, unique `key` per
concept label, unique `LifeArea.name`.

**Why Neo4j (and why also an in-memory twin).** The data *is* a graph: a user
connected to typed concepts connected to life areas, with lifecycle on the
edges, and the natural queries are traversals. The dict-backed store implements
the same protocol so tests run in milliseconds and the service keeps serving
(degraded) if the database is down.

---

## 5. Memory strategy

**Short-term** is the `ConversationBuffer`: last 20 turns per `(user, session)`,
in process (Redis in production). Follow-ups such as *"Why do you say that?"*
are classified as `followup` and get more recent turns and fewer memories.

**Long-term** is the graph, and only durable facts get in. Four gates:

1. **Pre-gate (free).** Pure questions with no first-person declaration skip
   extraction entirely. *"What should I focus on in my career?"* → no LLM call.
   *"Should I take the job? I have an offer from Google."* → extract.
2. **LLM extraction** into a strict schema: `type, key, value, attributes,
   life_areas, confidence, persistent, evidence, replaces, status`. The prompt
   states what to remember (goals, plans, events, stable preferences, life
   circumstances, corrections) and what not to (questions, moods, the
   assistant's own advice, one-off requests).
3. **Policy** (`memory/policy.py`, deterministic): drop non-persistent, drop
   below the confidence floor (0.55), drop values that are questions or look
   like transcripts, drop transient state ("tired today") even if the model
   marked it persistent, reject system-derived types (astro) from chat,
   normalise life areas and keys.
4. **Upsert with conflict resolution** (`brain/service.py`):
   - same key, same value → **confirmed**: `mention_count++`, confidence nudged
     up, attributes merged, recency refreshed;
   - exclusive slot (preference / fact / astro) with a new value →
     old edge **superseded** (`superseded_by` set), new edge created;
   - extractor says `status: withdrawn` (+ `replaces`) → old edge **withdrawn**;
   - otherwise **created**.

Nothing is deleted. Superseded and withdrawn edges stay for audit and are never
retrieved as context. Importance is type-based (goal 0.9 > event 0.8 > fact 0.7
> preference 0.6 > interest 0.5) with a bump for time-bound items.

What the update step decided is returned on every response
(`memory_updates`), including *why* something was dropped.

---

## 6. Context selection

`QueryUnderstander` → `intent ∈ {advice, recall, followup, smalltalk,
profile_update}`, `life_areas`, `needs_astrology`.

- **Rules first**: recall phrases ("what do you remember"), follow-up anaphora
  on short messages ("why do you say that", "explain that"), profile
  declarations, keyword maps per life area.
- **LLM only when rules are unsure** and the message has ≥5 words. If the LLM
  fails, rules win. Tested: a career keyword hit makes zero LLM calls.

`ContextBuilder` assembles the system prompt under a character budget in
priority order and records each block:

| Block | When | `context_used` tag |
| --- | --- | --- |
| Profile | always (an empty profile becomes "unknown, missing: …") | `user_profile` / `profile_missing` |
| Astrology | `needs_astrology` and DOB known | `astrology` |
| Memories | active memories in the selected life areas, ranked, top-K (3 for follow-ups, 20 for recall, 8 otherwise) | `goal:career_change`, `preference:language`, … |
| Recent turns | last 6 (10 for follow-ups), trimmed to the remaining budget | `recent_conversation` |

The whole graph is never sent; the whole transcript is never sent. A prompt for
the career question above was ~580 tokens including the reply history.

---

## 7. LLM layer

`LLMProvider.complete(messages, model, temperature, max_tokens, json_mode)` is
the only interface. Adapters: `OpenAICompatProvider` (OpenAI, Azure gateways,
Gemini's OpenAI endpoint, Ollama…), `AnthropicProvider`, `FakeLLM`.
`ResilientLLM` wraps any provider with exponential-backoff retries on transient
errors (timeouts, 429, 5xx), an optional fallback model, and an optional
fallback provider. Extraction and intent use `LLM_SMALL_MODEL` when set.

---

## 8. Error handling

| Failure | Behaviour |
| --- | --- |
| Invalid input (blank message, empty ids, >4000 chars, future DOB) | `422 {"error":"invalid_input","details":[{field,message}]}` |
| Unknown user in `/chat` | auto-created bare profile; prompt says which fields are missing; model asks once |
| Missing DOB | no astrology block, nothing fabricated |
| LLM failure | retries → fallback model → fallback provider → graceful reply with `degraded: true`, `warnings: ["llm_unavailable"]`; the turn is still buffered |
| Neo4j down at boot | logged, falls back to in-memory store (`GRAPH_FALLBACK_TO_MEMORY`) |
| Neo4j down mid-flight | answer from profile + short-term context, `warnings: ["graph_unavailable"]`, `/health` reports degraded, `/users/*` returns 503 |
| Extraction / parse failure | logged, response unaffected |
| Empty memory / no relevant context | prompt states it; `context_used` shows `no_memories` on recall |

---

## 9. Tests

```
tests/test_scenarios.py         new user · create memory · retrieve · follow-up · new session ·
                                irrelevant memory · transient not stored · correction (withdraw, supersede, reinforce) · missing info
tests/test_error_handling.py    invalid input · LLM down · LLM flaky (retries) · graph outage mid-flight ·
                                boot fallback · empty memory · astro edges
tests/test_memory_policy.py     pre-gate · every drop rule · importance · tolerant JSON parsing
tests/test_intent.py            rule classification · follow-up inheritance · LLM only when unsure · LLM failure
tests/test_astrology.py         sign boundaries · date parsing · profile merge
tests/test_neo4j_integration.py the flow against real Neo4j, incl. Cypher check of the graph shape
```

`make test` → 68 passed, 1 skipped (integration) in ~1s. `make test-integration` with Neo4j up → passes.

### Evaluating the Shared Brain
See [`docs/evaluation.md`](docs/evaluation.md). `scripts/evaluate.py` runs a
golden conversation and reports memory precision/recall, attribute accuracy,
intent accuracy, irrelevant-context rate and policy violations, offline or
against the real model.

---

## 10. Key design decisions and trade-offs

- **Hybrid extraction, not pure LLM or pure rules.** The LLM understands
  language; the policy is auditable. A reviewer can read `policy.py` and know
  exactly what can never be stored.
- **Inline memory update.** The response carries `memory_updates`, which is
  great for demos and debugging, at the cost of one extra LLM call of latency
  on turns that pass the pre-gate. In production: queue it (the `ChatService`
  already isolates that stage).
- **Keys are normalised strings, not embeddings.** Simple, deterministic,
  explainable. The cost: the model may say `job_switch` today and
  `career_change` tomorrow. Mitigation today: the extractor is shown the
  user's current keys and asked to reuse them. Next step: embedding similarity
  on `(type, value)` at upsert time.
- **Supersede, never delete.** Corrections keep history (`superseded_by`),
  which makes "what did the user believe in March" answerable and bugs
  recoverable.
- **Shared concept nodes.** Enables cross-user analytics and keeps the graph
  compact; the price is that per-user data must live on edges (JSON attributes).
- **Sun-sign stub.** Deliberately minimal; `AstroProfile` is the seam where a
  real Vedic engine (rashi, nakshatra, dasha) plugs in.

---

## 11. Production considerations

- **State**: move `ConversationBuffer` to Redis (list + TTL); keep Neo4j as the
  brain; add an outbox/queue for the memory-update stage.
- **Scale reads**: cache the per-user active subgraph (invalidate on write),
  add an index on `r.status` via a `(:Memory)` intermediate node if edge
  property filtering becomes hot, or precompute `User -[:ACTIVE_IN]-> LifeArea`.
- **Cost/latency**: pre-gate already skips extraction on ~half the turns; route
  extraction/intent to a small model; cap `CONTEXT_CHAR_BUDGET`.
- **Quality loop**: golden set from logs, LLM-as-judge for relevance and
  consistency, "brain on/off" A/B on retention (see `docs/evaluation.md`).
- **Safety/privacy**: memories are PII; add per-user export/delete
  (`DELETE /users/{id}/memories`), field-level encryption at rest, retention
  and decay jobs (`last_confirmed_at` is already there for that).
- **Observability**: log `context_used`, `memory_updates`, token counts and
  drop reasons per request; alert on extraction-drop spikes and degraded rate.
- **Multilingual**: `preferred_language` flows into the prompt today (Hindi
  verified); extraction prompt handles Hinglish in testing, but a golden set
  per language is needed before relying on it.
