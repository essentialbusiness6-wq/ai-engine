"""
Authentication & authorization.

- authenticate(): validates a bearer JWT and produces a Principal (user_id,
  tenant_id, permissions, roles). Any consuming application issues these
  JWTs (e.g. after its own login flow) using the shared secret/algorithm,
  so the AI Engine stays a stateless, reusable service.
- Permission checks themselves live on Principal.has_permission() (domain
  layer) and are enforced in the ToolExecutor -- security.py only handles
  identity, not the authorization decision for a specific tool.
"""
from __future__ import annotations

import time

import jwt

from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.domain.interfaces import Principal


def issue_token(user_id: str, tenant_id: str | None, permissions: list[str], roles: list[str]) -> str:
    """Helper mainly for tests/dev tooling to mint tokens without a full auth service."""
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "permissions": permissions,
        "roles": roles,
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.jwt_expiry_seconds,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def authenticate(bearer_token: str) -> Principal:
    if not bearer_token:
        raise AuthenticationError("Missing bearer token")
    try:
        payload = jwt.decode(bearer_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as e:
        raise AuthenticationError("Token has expired") from e
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {e}") from e

    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Token missing 'sub' claim")

    return Principal(
        user_id=user_id,
        tenant_id=payload.get("tenant_id"),
        permissions=set(payload.get("permissions", [])),
        roles=set(payload.get("roles", [])),
    )
