import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routes import admin, chat, documents, health
from app.container import build_container
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.logging import request_id_var, setup_logging

logger = logging.getLogger("app.access")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    container = build_container(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        container.close()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Agentic RAG: upload documents, then chat with an agent that retrieves, "
        "grades, rewrites and falls back to web search.",
        lifespan=lifespan,
    )
    app.state.container = container
    register_error_handlers(app)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    for router in (health.router, admin.router, documents.router, chat.router):
        app.include_router(router)
    return app


def app_factory() -> FastAPI:  # uvicorn --factory app.main:app_factory
    return create_app()
