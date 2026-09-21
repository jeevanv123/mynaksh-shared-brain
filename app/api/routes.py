"""HTTP routes. Thin: validate, call the service, shape the response."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    AstroOut,
    ChatRequest,
    ChatResponse,
    CreateUserRequest,
    HealthOut,
    MemoriesOut,
    MemoryChangeOut,
    MemoryOut,
    UserOut,
)
from app.brain.store import GraphStoreError
from app.dependencies import Container
from app.profile.models import UserProfile

router = APIRouter()


def _container(request: Request) -> Container:
    return request.app.state.container


def _user_out(p: UserProfile) -> UserOut:
    return UserOut(
        user_id=p.user_id,
        name=p.name,
        date_of_birth=p.date_of_birth,
        time_of_birth=p.time_of_birth,
        birth_place=p.birth_place,
        preferred_language=p.preferred_language,
        astrology=AstroOut(**p.astro.as_dict()) if p.astro else None,
        missing_fields=p.missing_fields(),
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


@router.get("/health", response_model=HealthOut, tags=["ops"])
async def health(request: Request) -> HealthOut:
    c = _container(request)
    ok = await c.store.ping()
    return HealthOut(
        status="ok" if ok else "degraded",
        graph_store=c.store.name,
        graph_ok=ok,
        llm_provider=c.settings.llm_provider,
        llm_model=c.settings.llm_model,
    )


@router.post("/users", response_model=UserOut, status_code=201, tags=["users"])
async def create_or_update_user(body: CreateUserRequest, request: Request) -> UserOut:
    c = _container(request)
    try:
        existing = await c.store.get_user(body.user_id)
        profile = existing.merged_with(body.model_dump(exclude={"user_id"})) if existing else UserProfile(**body.model_dump())
        saved = await c.brain.save_profile(profile)
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=f"graph store unavailable: {exc}") from exc
    return _user_out(saved)


@router.get("/users/{user_id}", response_model=UserOut, tags=["users"])
async def get_user(user_id: str, request: Request) -> UserOut:
    c = _container(request)
    try:
        profile = await c.store.get_user(user_id)
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=f"graph store unavailable: {exc}") from exc
    if not profile:
        raise HTTPException(status_code=404, detail=f"user {user_id!r} not found")
    return _user_out(profile)


@router.get("/users/{user_id}/memories", response_model=MemoriesOut, tags=["brain"])
async def list_memories(user_id: str, request: Request, include_inactive: bool = False) -> MemoriesOut:
    c = _container(request)
    try:
        if not await c.store.get_user(user_id):
            raise HTTPException(status_code=404, detail=f"user {user_id!r} not found")
        memories = await c.brain.all_memories(user_id, include_inactive=include_inactive)
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=f"graph store unavailable: {exc}") from exc
    memories.sort(key=lambda m: m.rank_score(), reverse=True)
    return MemoriesOut(user_id=user_id, count=len(memories), memories=[MemoryOut.from_memory(m) for m in memories])


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    c = _container(request)
    result = await c.chat.chat(body.user_id, body.session_id, body.message)
    return ChatResponse(
        response=result.response,
        user_id=body.user_id,
        session_id=body.session_id,
        context_used=result.context_used,
        intent=result.intent,
        life_areas=result.life_areas,
        memory_updates=[MemoryChangeOut.from_change(ch) for ch in result.memory_changes],
        degraded=result.degraded,
        warnings=result.warnings,
        model=result.model,
        approx_prompt_tokens=result.approx_prompt_tokens,
    )
