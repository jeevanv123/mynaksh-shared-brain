# Design: Personalized Astrology Chat with a Shared Brain

Date: 2026-09-21. Scope: the MyNaksh SDE2 machine-coding assignment.

## Goal

A FastAPI service that answers astrology-flavoured questions while remembering
the user across sessions. The interesting part is not astrology, it is the
memory pipeline:

```
message → understand query → select relevant context → build prompt → LLM → respond → update memory
```

The system must remember what matters, forget what does not, and never dump the
whole graph or whole transcript into the prompt.

## Architecture (one process, five layers)

| Layer | Package | Responsibility |
| --- | --- | --- |
| API | `app/api` | HTTP contracts, validation, error mapping |
| Chat orchestration | `app/chat` | query understanding, context selection, prompt building, pipeline |
| Memory | `app/memory` | short-term buffer, LLM extraction, keep/drop/supersede policy |
| Shared Brain | `app/brain` | graph schema, `GraphStore` protocol, Neo4j + in-memory implementations |
| LLM | `app/llm` | `LLMProvider` protocol, OpenAI-compatible + Anthropic adapters, retry/fallback, fake |

Profile and stubbed astrology live in `app/profile`.

Every cross-layer dependency is a `Protocol`, so each layer is testable alone
and swappable (Neo4j to in-memory, Bifrost to OpenAI to a fake).

## Shared Brain schema

Concept nodes are shared across users. User-specific facts live on the edge.

```
(:User {user_id, name, date_of_birth, time_of_birth, birth_place,
        preferred_language, sun_sign, created_at, updated_at})

(:User)-[:HAS_GOAL      {…edge props}]->(:Goal       {key, value})
(:User)-[:PREFERS       {…}]->(:Preference {key, value})
(:User)-[:INTERESTED_IN {…}]->(:Interest   {key, value})
(:User)-[:CARES_ABOUT   {…}]->(:LifeArea   {name})
(:User)-[:HAS_ATTRIBUTE {…}]->(:AstroAttribute {key, value})
(:User)-[:REMEMBERS     {…}]->(:Fact       {key, value})
(:User)-[:EXPERIENCED   {…}]->(:Event      {key, value})

(:Goal|Preference|Interest|Fact|Event|AstroAttribute)-[:RELATES_TO]->(:LifeArea {name})
```

Edge props: `memory_id, confidence, importance, status (active|superseded),
attributes (JSON: timeframe, target_year, …), evidence, source_session_id,
created_at, updated_at, last_confirmed_at, superseded_by`.

Why edges carry the facts: the concept "Career Change" is one node. A thousand
users can point at it, which makes cross-user questions cheap ("how many Leos
are planning a career change?") and keeps per-user retrieval a two-hop
traversal `User → concept → LifeArea`.

Retrieval is `MATCH (u:User {user_id})-[r]->(c)-[:RELATES_TO]->(la:LifeArea)
WHERE la.name IN $areas AND r.status = 'active'`, ranked in Python by
`importance × confidence × recency`.

## Memory strategy

Two stores with different lifetimes:

- Short-term: `ConversationBuffer`, per session, last N turns, process memory
  (Redis in production). Answers "Why do you say that?".
- Long-term: the graph. Only durable, user-specific facts.

Extraction is hybrid:

1. Cheap rule gate. Pure questions with no first-person declaration skip the
   extractor entirely (saves an LLM call on most turns).
2. LLM extraction into a strict JSON schema: `type, value, attributes,
   life_areas, confidence, persistent, replaces`.
3. Policy filter (`app/memory/policy.py`): drop non-persistent, drop
   confidence below the floor, drop disallowed types, drop values that are
   questions or too long, normalize keys.
4. Upsert with conflict resolution: exclusive slots (name, DOB, language…)
   supersede the old value; repeatable types (goals, interests) merge and boost
   confidence; an explicit `replaces` marks the old memory `superseded`.

Nothing is stored raw. Superseded memories are kept for audit, never returned
as context.

## Context selection

`QueryUnderstanding` produces `intent ∈ {followup, recall, advice, smalltalk,
profile_update}` plus `life_areas` and a `needs_astrology` flag. Rules run
first (anaphora detection, keyword maps); an LLM classifier is consulted only
when rules are unsure and the message is long enough to justify it.

`ContextBuilder` then assembles, in priority order and under a character
budget: compact profile, astrology attributes (when relevant), top-K active
memories in the selected life areas, and the recent turns (more of them for
follow-ups). The response reports exactly what was used in `context_used`.

## Error handling

| Failure | Behaviour |
| --- | --- |
| Invalid input | 422 with a readable message |
| Unknown user | auto-created bare profile, model told to ask for birth details |
| LLM failure | retries with backoff, provider fallback chain, then a graceful degraded reply (`degraded: true`) |
| Neo4j unreachable at boot | warning, fall back to in-memory store |
| Neo4j error at runtime | context degrades to profile + short-term, response still served |
| Empty memory / no relevant context | prompt states it plainly, response still personalized by profile |

## Testing

Scenario tests (new user, create memory, retrieve, follow-up, new session,
irrelevant memory, correction, missing info, failures) run against the
in-memory store and a deterministic `FakeLLM`. An integration test runs the
same flow against real Neo4j when it is reachable and skips otherwise.

## Trade-offs

- Extraction is inline, so the response carries `memory_updates`. Costs one
  LLM call of latency; in production it moves to a queue.
- Concept nodes are keyed by normalized string, not embeddings. Simple and
  auditable; loses fuzzy matching ("job switch" vs "career change") beyond
  what the extractor normalizes. Embedding-based dedupe is the next step.
- Session buffer is in-process. Fine for one replica; Redis for many.
