"""
Middleware: request/response logging plus converting any AIEngineError (or
unexpected exception) into a consistent JSON error body. Applied globally
so no route handler needs to repeat this boilerplate.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.exceptions import AIEngineError

request_log = logging.getLogger("ai_engine.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.monotonic()
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except AIEngineError as e:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            request_log.warning(
                "request_failed method=%s path=%s status=%s duration_ms=%s request_id=%s error=%s",
                request.method, request.url.path, e.http_status, duration_ms, request_id, e.message,
            )
            return JSONResponse(
                status_code=e.http_status,
                content={"error_code": e.error_code, "message": e.message, "request_id": request_id},
            )
        except Exception as e:  # noqa: BLE001 - final safety net at the HTTP boundary
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            request_log.exception(
                "request_unhandled_error method=%s path=%s duration_ms=%s request_id=%s",
                request.method, request.url.path, duration_ms, request_id,
            )
            return JSONResponse(
                status_code=500,
                content={"error_code": "internal_error", "message": "An unexpected error occurred.",
                          "request_id": request_id},
            )

        duration_ms = round((time.monotonic() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        request_log.info(
            "request_completed method=%s path=%s status=%s duration_ms=%s request_id=%s",
            request.method, request.url.path, response.status_code, duration_ms, request_id,
        )
        return response
