"""
Domain entities: pure data structures representing the core concepts of the
AI Engine. These have NO dependency on FastAPI, databases, or any specific
LLM/NLP library, per Clean Architecture's dependency rule (domain is the
innermost layer and depends on nothing else in the system).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"
    REJECTED = "rejected"          # e.g. permission denied, injection detected
    PENDING_CONFIRMATION = "pending_confirmation"
    ANSWERED = "answered"          # LLM answered directly (support/FAQ), no tool executed


class EntityType(str, Enum):
    PERSON_NAME = "person_name"
    DATE = "date"
    RELATIVE_DATE = "relative_date"
    AMOUNT = "amount"
    CURRENCY = "currency"
    INVOICE_NUMBER = "invoice_number"
    PAYMENT_REFERENCE = "payment_reference"
    PRODUCT = "product"
    EMAIL = "email"
    PHONE = "phone"


@dataclass(frozen=True)
class ExtractedEntity:
    """A single piece of structured data pulled out of free text."""
    type: EntityType
    value: Any                     # normalized value (e.g. datetime, Decimal)
    raw_text: str                  # original substring matched
    confidence: float = 1.0
    start: int = -1
    end: int = -1


@dataclass
class Message:
    """A single turn in a conversation."""
    role: Role
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """
    Aggregate of everything the engine knows about an ongoing conversation:
    message history plus a scratchpad of slot values that have been resolved
    across turns (e.g. "invoice #123" mentioned two turns ago, then the user
    says "pay it" -> resolver needs that value).
    """
    conversation_id: str
    user_id: str
    tenant_id: Optional[str] = None
    messages: list[Message] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)   # resolved entities
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def recent_messages(self, n: int = 10) -> list[Message]:
        return self.messages[-n:]


@dataclass
class IntentResult:
    """Output of the Intent Recognition layer."""
    intent: str
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ToolParameter:
    name: str
    type: str
    required: bool = True
    description: str = ""


@dataclass
class ToolDefinition:
    """Metadata describing a callable backend tool."""
    name: str
    description: str
    parameters: list[ToolParameter]
    required_permission: Optional[str] = None
    # Optional full JSON-schema for `parameters` (used to describe nested/
    # array arguments, e.g. an invoice's line_items, that a flat list of
    # ToolParameter can't express). When present, this is what's sent to
    # the LLM provider's tool-use schema; `parameters` above still drives
    # simple required-field validation in the ToolExecutor.
    parameters_schema: Optional[dict[str, Any]] = None


@dataclass
class ToolCall:
    """A concrete, argument-bound invocation of a tool, chosen by the LLM."""
    tool_name: str
    arguments: dict[str, Any]
    confidence: float = 0.0


# Sentinel tool name: the LLM provider returns this (with the answer text in
# arguments["message"]) when the best response is a direct natural-language
# answer rather than a tool call -- e.g. support/FAQ questions. The
# orchestrator special-cases this and never routes it through the
# ToolRegistry/ToolExecutor, so it can never trigger business logic.
RESPOND_TOOL_NAME = "__respond__"
NO_TOOL_NAME = "__none__"


@dataclass
class ToolExecutionResult:
    status: ExecutionStatus
    data: Any = None
    error_message: Optional[str] = None


@dataclass
class EngineResponse:
    """
    The single structured response shape returned by every /interpret call,
    regardless of which tool/intent/LLM handled it underneath.
    """
    conversation_id: str
    tool_name: Optional[str]
    arguments: dict[str, Any]
    confidence: float
    status: ExecutionStatus
    message: str                     # user-facing natural language message
    data: Any = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
