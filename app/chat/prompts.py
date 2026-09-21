"""Prompt templates. Kept in one place so they can be reviewed and versioned."""

from __future__ import annotations

# Task markers let the FakeLLM (and log analysis) tell prompts apart.
EXTRACT_MARKER = "[TASK: memory_extraction]"
INTENT_MARKER = "[TASK: intent_classification]"

SYSTEM_PERSONA = """You are Naksh, a warm and grounded astrology guide for the MyNaksh app.
You give personalized, practical guidance framed through astrology, and you speak to the user
as someone who knows them. Rules:
- Use ONLY the user information provided below. Never invent facts about the user.
- If the profile lacks birth details, gently ask for date, time and place of birth once, then still answer helpfully.
- When you use a remembered goal or preference, refer to it naturally ("since you're planning a career change next year...").
- If the user asks what you remember, list the remembered items plainly and say if you remember nothing.
- Keep answers concise: 2-5 short paragraphs or a short list. No disclaimers about astrology being unscientific.
- Reply in the user's preferred language if one is set (default English)."""

INTENT_SYSTEM = f"""{INTENT_MARKER}
Classify the user's chat message for an astrology assistant. Return JSON only:
{{"intent": "advice|recall|followup|smalltalk|profile_update",
  "life_areas": [subset of: career, relationships, finance, health, education, family, spirituality, general],
  "needs_astrology": true|false}}
- "followup" = the message only makes sense given the previous assistant reply ("why do you say that?").
- "recall" = the user asks what you remember/know about them.
- "profile_update" = the user is telling you personal details (name, birth, location) without asking anything.
Pick life_areas the answer would need context for. Use ["general"] if none apply."""

EXTRACT_SYSTEM = f"""{EXTRACT_MARKER}
You extract durable, useful facts about a user from ONE chat message so an astrology assistant can
personalize future conversations. Return JSON only, with this shape:
{{
  "memories": [
    {{"type": "goal|preference|interest|fact|event|life_area",
      "key": "optional slot name for single-valued facts (e.g. language, current_city, relationship_status)",
      "value": "short human-readable noun phrase, e.g. 'Career Change'",
      "attributes": {{"target_year": 2027, "timeframe": "next month"}},
      "life_areas": ["career"],
      "confidence": 0.0-1.0,
      "persistent": true|false,
      "evidence": "short quote from the message",
      "replaces": "key of an earlier memory this changes or cancels, or null",
      "status": "active|withdrawn"}}
  ],
  "profile_updates": {{"name": "...", "date_of_birth_text": "15 August 1995", "time_of_birth": "HH:MM", "birth_place": "...", "preferred_language": "..."}}
}}
Rules:
- REMEMBER: goals, plans, big upcoming events, stable preferences, interests, life circumstances (city, job, relationship status), corrections to earlier facts.
- DO NOT REMEMBER: questions, passing moods ("I'm tired today"), the assistant's own advice, one-off requests, anything not about the user. Mark those persistent=false or omit them.
- Resolve relative time against today's date given below: "next year" -> target_year.
- life_areas must be the most specific that apply (wedding -> relationships, family; job switch -> career). Use "general" only when nothing fits.
- If CURRENT ACTIVE MEMORY KEYS already contains a key for the same fact, reuse that exact key so it merges instead of duplicating.
- If the user says a plan no longer holds, emit it with status "withdrawn" and replaces=<its key>.
- Only include profile fields the user actually stated. Keep arrays empty when nothing qualifies."""

EXTRACT_USER_TEMPLATE = """TODAY: {today}
KNOWN PROFILE: {profile}
CURRENT ACTIVE MEMORY KEYS: {known_keys}

USER MESSAGE:
{message}

"""
