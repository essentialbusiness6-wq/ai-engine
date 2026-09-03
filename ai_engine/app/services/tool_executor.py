"""
Tool Executor.

The ONLY component that actually calls backend service functions. It:
  1. Looks up the tool definition + handler from the ToolRegistry.
  2. Enforces authorization (Principal must have the tool's required
     permission) -- this is the enforcement point, independent of whether
     the LLM "decided" to call the tool.
  3. Validates that required arguments are present.
  4. Executes with retry-on-transient-failure semantics.
  5. Converts any exception into a structured ToolExecutionResult rather
     than letting it propagate as a raw stack trace to the API layer.

This is deliberately the single choke point between "LLM picked a tool"
and "business logic actually ran," which is what enforces the requirement
that the LLM never directly executes SQL or business logic.
"""
from __future__ import annotations

import logging
import time

from app.domain.entities import ExecutionStatus, ToolCall, ToolExecutionResult
from app.domain.interfaces import IAuditLogger, IToolExecutor, IToolRegistry, Principal

logger = logging.getLogger("ai_engine.tool_executor")


class TransientToolError(Exception):
    """Raised by handlers to signal a retryable failure (e.g. network blip)."""


class ValidationError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


class RetryingToolExecutor(IToolExecutor):
    def __init__(
        self,
        registry: IToolRegistry,
        audit_logger: IAuditLogger | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.1,
    ):
        self._registry = registry
        self._audit_logger = audit_logger
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    def execute(self, tool_call: ToolCall, principal: Principal) -> ToolExecutionResult:
        definition = self._registry.get_definition(tool_call.tool_name)
        if definition is None:
            return ToolExecutionResult(
                status=ExecutionStatus.FAILED,
                error_message=f"Unknown tool '{tool_call.tool_name}'",
            )

        try:
            self._check_permission(definition, principal)
            self._validate_arguments(definition, tool_call.arguments)
        except PermissionDeniedError as e:
            self._audit(tool_call, principal, "tool_rejected_permission", {"error": str(e)})
            return ToolExecutionResult(status=ExecutionStatus.REJECTED, error_message=str(e))
        except ValidationError as e:
            self._audit(tool_call, principal, "tool_rejected_validation", {"error": str(e)})
            return ToolExecutionResult(status=ExecutionStatus.NEEDS_CLARIFICATION, error_message=str(e))

        handler = self._registry.get_handler(tool_call.tool_name)
        result = self._execute_with_retry(tool_call, handler)

        self._audit(tool_call, principal, "tool_executed", {
            "status": result.status.value,
            "error": result.error_message,
        })
        return result

    # -- internals --------------------------------------------------

    def _check_permission(self, definition, principal: Principal) -> None:
        if not principal.has_permission(definition.required_permission):
            raise PermissionDeniedError(
                f"Principal '{principal.user_id}' lacks permission "
                f"'{definition.required_permission}' for tool '{definition.name}'"
            )

    def _validate_arguments(self, definition, arguments: dict) -> None:
        for param in definition.parameters:
            if param.required and param.name not in arguments:
                raise ValidationError(f"Missing required argument '{param.name}'")

    def _execute_with_retry(self, tool_call: ToolCall, handler) -> ToolExecutionResult:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                data = handler(**tool_call.arguments)
                return ToolExecutionResult(status=ExecutionStatus.SUCCESS, data=data)
            except TransientToolError as e:
                last_error = e
                logger.warning("Transient error on tool %s (attempt %d): %s",
                                tool_call.tool_name, attempt + 1, e)
                time.sleep(self._retry_backoff_seconds)
                continue
            except (LookupError, ValueError) as e:
                # Business-rule failures are NOT retried (retrying a
                # "invoice already paid" error won't help).
                return ToolExecutionResult(status=ExecutionStatus.FAILED, error_message=str(e))
            except Exception as e:  # noqa: BLE001 - final safety net
                logger.exception("Unexpected error executing tool %s", tool_call.tool_name)
                return ToolExecutionResult(status=ExecutionStatus.FAILED, error_message=f"Internal error: {e}")

        return ToolExecutionResult(
            status=ExecutionStatus.FAILED,
            error_message=f"Tool '{tool_call.tool_name}' failed after retries: {last_error}",
        )

    def _audit(self, tool_call: ToolCall, principal: Principal, event: str, details: dict) -> None:
        if self._audit_logger:
            self._audit_logger.log(event, principal, {
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                **details,
            })
