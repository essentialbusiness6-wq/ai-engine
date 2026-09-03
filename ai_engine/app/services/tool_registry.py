"""
Tool Registry.

A simple, explicit mapping from tool name -> (ToolDefinition, handler
function). This is the ONLY place tools are wired up. The LLM never
executes business logic directly; it only ever returns the NAME of one of
these registered tools plus arguments, and the ToolExecutor (see
tool_executor.py) looks the handler up here and calls it.

Handlers are expected to be thin adapters over EXISTING backend services --
the registry does not contain business logic itself, only wiring.
"""
from __future__ import annotations

from typing import Callable, Optional

from app.domain.entities import ToolDefinition
from app.domain.interfaces import IToolRegistry

ToolHandler = Callable[..., object]


class InMemoryToolRegistry(IToolRegistry):
    def __init__(self):
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool '{definition.name}' is already registered")
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        return self._definitions.get(tool_name)

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def get_handler(self, tool_name: str) -> Optional[ToolHandler]:
        return self._handlers.get(tool_name)
