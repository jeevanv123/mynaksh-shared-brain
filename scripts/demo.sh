#!/usr/bin/env bash
# Scripted walkthrough of the assignment's example conversation.
# Usage: make run (or make run-offline) in another terminal, then: bash scripts/demo.sh
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"
UID_="${USER_ID:-rahul-$RANDOM}"

say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
post() { curl -s -X POST "$BASE$1" -H 'Content-Type: application/json' -d "$2" | python3 -m json.tool; }
chat() { post /chat "{\"user_id\":\"$UID_\",\"session_id\":\"$1\",\"message\":\"$2\"}"; }

say "health"; curl -s "$BASE/health" | python3 -m json.tool

say "1. create profile"
post /users "{\"user_id\":\"$UID_\",\"name\":\"Rahul\",\"date_of_birth\":\"1995-08-15\",\"time_of_birth\":\"10:30\",\"birth_place\":\"Delhi\"}"

say "2. first conversation: a plan becomes a long-term goal"
chat s1 "I'm planning to switch jobs next year."

say "3. same session: career question pulls the goal into context"
chat s1 "What should I focus on for my career?"

say "4. follow-up: answered from short-term context"
chat s1 "Why do you say that?"

say "5. transient state is NOT remembered"
chat s1 "I'm feeling tired today."

say "6. new session: long-term recall"
chat s2 "What do you remember about my career goals?"

say "7. correction: the goal is withdrawn, not deleted"
chat s2 "Actually, I'm no longer planning to switch jobs."

say "8. irrelevant memory stays out of an unrelated question"
chat s3 "Will my marriage be happy?"

say "9. what the Shared Brain holds (including superseded/withdrawn)"
curl -s "$BASE/users/$UID_/memories?include_inactive=true" | python3 -m json.tool
