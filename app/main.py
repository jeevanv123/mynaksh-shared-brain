"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.brain.store import GraphStoreError
from app.config import Settings, get_settings
from app.dependencies import Container, build_container
from app.llm.base import LLMError


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or await build_container(settings)
        try:
            yield
        finally:
            await app.state.container.close()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Personalized astrology chat with a graph-backed Shared Brain.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "invalid_input", "details": errors})

    @app.exception_handler(GraphStoreError)
    async def _graph(_: Request, exc: GraphStoreError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "graph_unavailable", "message": str(exc)})

    @app.exception_handler(LLMError)
    async def _llm(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": "llm_unavailable", "message": str(exc)})

    return app


app = create_app()
