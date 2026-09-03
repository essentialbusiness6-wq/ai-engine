"""
Conversation Manager.

Responsible for storing and retrieving conversation context: message
history plus resolved "slots" (entities carried across turns). This default
implementation is in-memory and thread-safe, intended for development/tests
and single-process deployments. In production you would implement the same
IConversationStore interface backed by Redis or Postgres -- nothing else in
the system needs to change (Liskov substitution: any IConversationStore is
interchangeable).

A max-turns window and TTL-style eviction are included so conversations
don't grow unbounded in memory.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from app.domain.entities import ConversationContext, Message
from app.domain.interfaces import IConversationStore


class InMemoryConversationStore(IConversationStore):
    def __init__(self, max_messages_per_conversation: int = 50, ttl_seconds: int = 3600):
        self._store: dict[str, ConversationContext] = {}
        self._last_access: dict[str, float] = {}
        self._lock = threading.RLock()
        self._max_messages = max_messages_per_conversation
        self._ttl_seconds = ttl_seconds

    def get_or_create(
        self, conversation_id: str, user_id: str, tenant_id: Optional[str] = None
    ) -> ConversationContext:
        with self._lock:
            self._evict_expired()
            ctx = self._store.get(conversation_id)
            if ctx is None:
                ctx = ConversationContext(
                    conversation_id=conversation_id, user_id=user_id, tenant_id=tenant_id
                )
                self._store[conversation_id] = ctx
            self._last_access[conversation_id] = time.time()
            return ctx

    def save(self, context: ConversationContext) -> None:
        with self._lock:
            self._store[context.conversation_id] = context
            self._last_access[context.conversation_id] = time.time()

    def append_message(self, conversation_id: str, message: Message) -> None:
        with self._lock:
            ctx = self._store.get(conversation_id)
            if ctx is None:
                raise KeyError(f"No conversation found with id={conversation_id}")
            ctx.add_message(message)
            if len(ctx.messages) > self._max_messages:
                ctx.messages = ctx.messages[-self._max_messages:]
            self._last_access[conversation_id] = time.time()

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            cid for cid, last in self._last_access.items()
            if now - last > self._ttl_seconds
        ]
        for cid in expired:
            self._store.pop(cid, None)
            self._last_access.pop(cid, None)
