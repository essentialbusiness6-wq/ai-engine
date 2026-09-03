"""
Abstract interfaces ("ports" in hexagonal-architecture terms). Every service
in app/services implements one of these. The orchestrator and API layer only
ever depend on these abstractions, never on concrete classes. That is what
lets you swap the LLM provider, the intent recognizer, or the conversation
store without touching any business logic (Open/Closed + Dependency
Inversion principles).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.domain.entities import (
    ConversationContext,
    EntityType,
    ExtractedEntity,
    IntentResult,
    Message,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
)


class IConversationStore(ABC):
    """Persistence port for conversation context."""

    @abstractmethod
    def get_or_create(self, conversation_id: str, user_id: str,
                       tenant_id: Optional[str] = None) -> ConversationContext: ...

    @abstractmethod
    def save(self, context: ConversationContext) -> None: ...

    @abstractmethod
    def append_message(self, conversation_id: str, message: Message) -> None: ...


class ITextNormalizer(ABC):
    """Cleans raw user text before any NLP happens on it."""

    @abstractmethod
    def normalize(self, text: str) -> str: ...


class IIntentRecognizer(ABC):
    @abstractmethod
    def recognize(self, normalized_text: str) -> IntentResult: ...


class IEntityExtractor(ABC):
    @abstractmethod
    def extract(self, normalized_text: str) -> list[ExtractedEntity]: ...


class IContextResolver(ABC):
    """
    Combines freshly extracted entities with conversation history and
    backend lookups to fill in anything the user left implicit
    ("pay it" -> which invoice? -> look at conversation slots).
    """

    @abstractmethod
    def resolve(
        self,
        entities: list[ExtractedEntity],
        context: ConversationContext,
    ) -> dict[str, Any]: ...


class ILLMProvider(ABC):
    """
    Port for the language model. CRITICAL CONSTRAINT: implementations of
    this interface are used ONLY for language understanding and tool
    selection. They must never be given direct DB/SQL access and the rest
    of the system must never execute code returned by the LLM as-is; the
    ToolExecutor always calls a fixed set of pre-registered pure functions.
    """

    @abstractmethod
    def select_tool(
        self,
        user_text: str,
        intent: IntentResult,
        resolved_slots: dict[str, Any],
        available_tools: list[ToolDefinition],
        history: list[Message],
    ) -> ToolCall: ...


class IToolRegistry(ABC):
    @abstractmethod
    def register(self, definition: ToolDefinition, handler) -> None: ...

    @abstractmethod
    def get_definition(self, tool_name: str) -> Optional[ToolDefinition]: ...

    @abstractmethod
    def list_definitions(self) -> list[ToolDefinition]: ...

    @abstractmethod
    def get_handler(self, tool_name: str): ...


class IToolExecutor(ABC):
    @abstractmethod
    def execute(
        self,
        tool_call: ToolCall,
        principal: "Principal",
    ) -> ToolExecutionResult: ...


class ISpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, content_type: str) -> str: ...


class IAuditLogger(ABC):
    @abstractmethod
    def log(self, event: str, principal: "Principal", details: dict[str, Any]) -> None: ...


class Principal:
    """Represents the authenticated caller, produced by the auth layer."""

    def __init__(self, user_id: str, tenant_id: Optional[str], permissions: set[str], roles: set[str]):
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.permissions = permissions
        self.roles = roles

    def has_permission(self, permission: Optional[str]) -> bool:
        if permission is None:
            return True
        return permission in self.permissions or "admin" in self.roles
