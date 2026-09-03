"""
Structured application-level exceptions. Each carries an HTTP-status-ready
code so the API layer can translate them into consistent JSON error bodies
instead of leaking stack traces or ad hoc error shapes.
"""
from __future__ import annotations


class AIEngineError(Exception):
    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AuthenticationError(AIEngineError):
    http_status = 401
    error_code = "authentication_failed"


class AuthorizationError(AIEngineError):
    http_status = 403
    error_code = "authorization_failed"


class RateLimitExceededError(AIEngineError):
    http_status = 429
    error_code = "rate_limit_exceeded"


class ValidationFailedError(AIEngineError):
    http_status = 422
    error_code = "validation_failed"


class ConversationNotFoundError(AIEngineError):
    http_status = 404
    error_code = "conversation_not_found"
