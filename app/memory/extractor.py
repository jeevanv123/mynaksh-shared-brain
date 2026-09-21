"""LLM-based memory extraction into a strict schema."""

from __future__ import annotations

import json
import logging
import re
from datetime import date

from pydantic import ValidationError

from app.brain.models import ExtractionResult, MemoryCandidate
from app.chat import prompts
from app.llm.base import LLMError, LLMProvider, Message
from app.profile.models import UserProfile

log = logging.getLogger(__name__)


class MemoryExtractor:
    def __init__(self, llm: LLMProvider, model: str | None = None) -> None:
        self.llm = llm
        self.model = model

    async def extract(self, message: str, profile: UserProfile, known_keys: list[str]) -> ExtractionResult:
        user = prompts.EXTRACT_USER_TEMPLATE.format(
            today=date.today().isoformat(),
            profile=profile.compact_summary(),
            known_keys=", ".join(known_keys) or "(none)",
            message=message,
        )
        try:
            resp = await self.llm.complete(
                [Message("system", prompts.EXTRACT_SYSTEM), Message("user", user)],
                model=self.model,
                temperature=0.0,
                max_tokens=700,
                json_mode=True,
            )
        except LLMError as exc:
            log.warning("extraction skipped, LLM error: %s", exc)
            return ExtractionResult()
        return parse_extraction(resp.text)


def parse_extraction(text: str) -> ExtractionResult:
    """Tolerant JSON parsing: strips code fences, drops malformed candidates."""
    raw = _loads_lenient(text)
    if not isinstance(raw, dict):
        return ExtractionResult()
    candidates: list[MemoryCandidate] = []
    for item in raw.get("memories") or []:
        if not isinstance(item, dict):
            continue
        item = {k: v for k, v in item.items() if v is not None}
        if "type" in item:
            item["type"] = str(item["type"]).lower().strip()
        try:
            candidates.append(MemoryCandidate(**item))
        except ValidationError as exc:
            log.debug("dropping malformed candidate %s: %s", item, exc)
    updates = raw.get("profile_updates") or {}
    if not isinstance(updates, dict):
        updates = {}
    return ExtractionResult(memories=candidates, profile_updates={k: v for k, v in updates.items() if v})


def _loads_lenient(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    log.warning("extractor returned non-JSON output; ignoring")
    return None
