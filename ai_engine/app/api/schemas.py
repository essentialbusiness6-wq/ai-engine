"""
API-layer Pydantic schemas. These are deliberately separate from
app.domain.entities: domain entities are pure Python dataclasses used
internally, while these schemas define the wire format and validation
rules for the HTTP boundary (Separation of Concerns).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class InterpretRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=4000)
    confirmed: bool = False


class EngineResponseSchema(BaseModel):
    conversation_id: str
    tool_name: Optional[str]
    arguments: dict[str, Any]
    confidence: float
    status: str
    message: str
    data: Any = None
    request_id: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
