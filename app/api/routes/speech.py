"""
POST /v1/speech

Accepts an audio file upload, transcribes it (Whisper or equivalent, via
ISpeechToText), then feeds the resulting text through the EXACT SAME
orchestrator pipeline used by /v1/interpret. This guarantees speech and
text requests are handled identically from normalization onward -- there
is no separate "speech business logic."
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from app.api.dependencies import enforce_rate_limit, get_container, get_principal
from app.api.schemas import EngineResponseSchema
from app.domain.interfaces import Principal

router = APIRouter(prefix="/v1", tags=["speech"])


@router.post("/speech", response_model=EngineResponseSchema)
async def speech(
    audio: UploadFile,
    conversation_id: str = Form(...),
    confirmed: bool = Form(False),
    principal: Principal = Depends(get_principal),
    container=Depends(get_container),
    _rate_limit_ok: None = Depends(enforce_rate_limit),
) -> EngineResponseSchema:
    audio_bytes = await audio.read()
    try:
        transcript = container.speech_to_text.transcribe(
            audio_bytes, audio.content_type or "audio/wav"
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_audio", "message": str(e)})

    response = container.orchestrator.handle_text_request(
        conversation_id=conversation_id,
        text=transcript,
        principal=principal,
        confirmed=confirmed,
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
