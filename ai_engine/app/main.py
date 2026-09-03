"""
FastAPI application entrypoint.

Run with:  uvicorn app.main:app --reload

This file only does app assembly: creates the FastAPI instance, builds the
DI container once at startup and stores it on app.state, registers
middleware, and includes routers. No business logic lives here.
"""
from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from app.api.middleware import RequestLoggingMiddleware
from app.api.routes import health, interpret, speech, tools
from app.container import build_container
from app.core.config import settings

logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = build_container()
    yield


app = FastAPI(
    title="AI Engine API",
    description=(
        "Standalone, reusable AI Engine exposing REST endpoints for "
        "interpreting natural language and executing backend tools "
        "through a hybrid NLP + LLM pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Flattens FastAPI's default {"detail": {...}} envelope into the same
    {error_code, message, request_id} shape used by RequestLoggingMiddleware
    for AIEngineError, so every error response from this API has one
    consistent structure regardless of where it originated.
    """
    request_id = getattr(request.state, "request_id", None)
    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        content = {**exc.detail, "request_id": request_id}
    else:
        content = {"error_code": "http_error", "message": str(exc.detail), "request_id": request_id}
    return JSONResponse(status_code=exc.status_code, content=content)


app.include_router(health.router)
app.include_router(interpret.router)
app.include_router(speech.router)
app.include_router(tools.router)
