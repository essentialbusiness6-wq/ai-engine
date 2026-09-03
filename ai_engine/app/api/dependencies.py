"""
FastAPI dependency providers.

This module bridges FastAPI's request lifecycle to the plain-Python
composition root in app/main.py (see `build_container()`). Route handlers
depend only on these functions via `Depends(...)`, never on concrete
classes directly, keeping the API layer thin and testable.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from app.core.exceptions import AIEngineError, AuthenticationError, RateLimitExceededError
from app.core.security import authenticate
from app.domain.interfaces import Principal


def get_container(request: Request):
    """Returns the app-wide container built once at startup (see main.py)."""
    return request.app.state.container


def get_principal(
    authorization: str = Header(default=""),
    container=Depends(get_container),
) -> Principal:
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    try:
        return authenticate(token)
    except AuthenticationError as e:
        raise HTTPException(status_code=e.http_status, detail={"error_code": e.error_code, "message": e.message})


def enforce_rate_limit(
    principal: Principal = Depends(get_principal),
    container=Depends(get_container),
) -> None:
    if not container.rate_limiter.allow(principal.user_id):
        raise HTTPException(
            status_code=429,
            detail={"error_code": "rate_limit_exceeded", "message": "Too many requests. Please slow down."},
        )
