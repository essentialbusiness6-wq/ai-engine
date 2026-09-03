"""
POST /v1/interpret

The primary endpoint: takes natural language text plus a conversation id,
runs it through the full AI Engine pipeline (normalize -> intent ->
entities -> resolve -> LLM tool selection -> confidence scoring -> tool
execution), and returns a structured EngineResponse.

Auth + rate limiting are enforced via FastAPI dependencies so every route
that needs them declares it explicitly and consistently.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import enforce_rate_limit, get_container, get_principal
from app.api.schemas import EngineResponseSchema, InterpretRequest
from app.domain.interfaces import Principal

router = APIRouter(prefix="/v1", tags=["interpret"])


@router.post("/interpret", response_model=EngineResponseSchema)
def interpret(
    body: InterpretRequest,
    principal: Principal = Depends(get_principal),
    container=Depends(get_container),
    _rate_limit_ok: None = Depends(enforce_rate_limit),
) -> EngineResponseSchema:
    response = container.orchestrator.handle_text_request(
        conversation_id=body.conversation_id,
        text=body.text,
        principal=principal,
        confirmed=body.confirmed,
    )
    return EngineResponseSchema(
        conversation_id=response.conversation_id,
        tool_name=response.tool_name,
        arguments=response.arguments,
        confidence=response.confidence,
        status=response.status.value,
        message=response.message,
        data=response.data,
        request_id=response.request_id,
        timestamp=response.timestamp,
    )
