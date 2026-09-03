import pytest

from app.domain.entities import ExecutionStatus, ToolCall, ToolDefinition, ToolParameter
from app.domain.interfaces import Principal
from app.services.tool_executor import RetryingToolExecutor, TransientToolError
from app.services.tool_registry import InMemoryToolRegistry
from app.tools.backend_services import InMemoryBillingService
from app.tools.tool_definitions import register_billing_tools


@pytest.fixture
def registry():
    r = InMemoryToolRegistry()
    register_billing_tools(r, InMemoryBillingService())
    return r


@pytest.fixture
def executor(registry):
    return RetryingToolExecutor(registry)


def admin_principal():
    return Principal(user_id="u1", tenant_id="t1", permissions=set(), roles={"admin"})


def readonly_principal():
    return Principal(user_id="u2", tenant_id="t1", permissions={"invoices:read"}, roles=set())


def test_registry_lists_definitions(registry):
    names = {d.name for d in registry.list_definitions()}
    assert "get_invoice_status" in names
    assert "create_payment" in names


def test_registry_rejects_duplicate_registration(registry):
    with pytest.raises(ValueError):
        registry.register(
            ToolDefinition(name="get_invoice_status", description="dup", parameters=[]),
            handler=lambda **_: None,
        )


def test_execute_successful_tool_call(executor):
    call = ToolCall(tool_name="get_invoice_status", arguments={"invoice_number": "12345"})
    result = executor.execute(call, admin_principal())
    assert result.status == ExecutionStatus.SUCCESS
    assert result.data["invoice_number"] == "12345"


def test_execute_unknown_tool_fails(executor):
    call = ToolCall(tool_name="does_not_exist", arguments={})
    result = executor.execute(call, admin_principal())
    assert result.status == ExecutionStatus.FAILED


def test_execute_missing_argument_needs_clarification(executor):
    call = ToolCall(tool_name="get_invoice_status", arguments={})
    result = executor.execute(call, admin_principal())
    assert result.status == ExecutionStatus.NEEDS_CLARIFICATION


def test_execute_without_permission_is_rejected(executor):
    call = ToolCall(tool_name="create_payment", arguments={
        "invoice_number": "12345", "amount": 500, "currency": "USD",
    })
    result = executor.execute(call, readonly_principal())
    assert result.status == ExecutionStatus.REJECTED


def test_execute_with_permission_succeeds(executor):
    principal = Principal(user_id="u3", tenant_id="t1", permissions={"payments:write"}, roles=set())
    call = ToolCall(tool_name="create_payment", arguments={
        "invoice_number": "12345", "amount": 500, "currency": "USD",
    })
    result = executor.execute(call, principal)
    assert result.status == ExecutionStatus.SUCCESS


def test_business_rule_failure_is_not_retried_and_returns_failed(executor):
    principal = Principal(user_id="u3", tenant_id="t1", permissions={"payments:write"}, roles=set())
    call = ToolCall(tool_name="create_payment", arguments={
        "invoice_number": "does-not-exist", "amount": 500, "currency": "USD",
    })
    result = executor.execute(call, principal)
    assert result.status == ExecutionStatus.FAILED
    assert "not found" in result.error_message


def test_transient_error_is_retried_then_succeeds():
    registry = InMemoryToolRegistry()
    attempts = {"count": 0}

    def flaky_handler(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise TransientToolError("temporary network blip")
        return {"ok": True}

    registry.register(
        ToolDefinition(name="flaky_tool", description="", parameters=[]),
        handler=flaky_handler,
    )
    executor = RetryingToolExecutor(registry, max_retries=2, retry_backoff_seconds=0)
    result = executor.execute(ToolCall(tool_name="flaky_tool", arguments={}), admin_principal())
    assert result.status == ExecutionStatus.SUCCESS
    assert attempts["count"] == 2
