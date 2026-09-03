"""
Registers each backend service method as a tool the AI Engine can select
and execute. These handler functions are intentionally thin: they only
adapt argument shapes and RAISE exceptions for the executor to turn into
structured errors -- no business logic lives here, it lives in the backend
service.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.domain.entities import ToolDefinition, ToolParameter
from app.domain.interfaces import IToolRegistry
from app.tools.backend_services import InMemoryBillingService


def register_billing_tools(registry: IToolRegistry, billing_service: InMemoryBillingService) -> None:

    registry.register(
        ToolDefinition(
            name="create_invoice",
            description=(
                "Create a new invoice for a customer with one or more line items "
                "(product, quantity, unit price), and a due date."
            ),
            parameters=[
                ToolParameter("customer_name", "string", required=True),
                ToolParameter("line_items", "array", required=True,
                               description="List of {product, quantity, unit_price}"),
                ToolParameter("due_date", "string", required=True,
                               description="ISO date, e.g. 2026-08-13"),
                ToolParameter("currency", "string", required=False),
            ],
            required_permission="invoices:write",
            parameters_schema={
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string", "description": "Name of the customer being invoiced"},
                    "line_items": {
                        "type": "array",
                        "description": "One entry per product/service being billed",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product": {"type": "string", "description": "Product or service name"},
                                "quantity": {"type": "number", "description": "Number of units"},
                                "unit_price": {"type": "number", "description": "Price per single unit"},
                            },
                            "required": ["product", "quantity", "unit_price"],
                        },
                    },
                    "due_date": {"type": "string", "description": "ISO 8601 date the invoice is due, e.g. 2026-08-13"},
                    "currency": {"type": "string", "description": "3-letter currency code, defaults to USD"},
                },
                "required": ["customer_name", "line_items", "due_date"],
            },
        ),
        handler=lambda customer_name, line_items, due_date, currency="USD", **_: _invoice_to_dict(
            billing_service.create_invoice(
                customer_name=customer_name,
                line_items=line_items,
                due_date=date.fromisoformat(due_date),
                currency=currency,
            )
        ),
    )

    registry.register(
        ToolDefinition(
            name="get_invoice_status",
            description="Look up the status, amount, and due date of an invoice by its number.",
            parameters=[ToolParameter("invoice_number", "string", required=True)],
            required_permission="invoices:read",
        ),
        handler=lambda invoice_number, **_: _invoice_to_dict(billing_service.get_invoice(invoice_number)),
    )

    registry.register(
        ToolDefinition(
            name="list_invoices",
            description="List invoices, optionally filtered by status (paid/unpaid/overdue).",
            parameters=[ToolParameter("status", "string", required=False)],
            required_permission="invoices:read",
        ),
        handler=lambda status=None, **_: [
            _invoice_to_dict(i) for i in billing_service.list_invoices(status=status)
        ],
    )

    registry.register(
        ToolDefinition(
            name="create_payment",
            description="Create and process a payment against an invoice.",
            parameters=[
                ToolParameter("invoice_number", "string", required=True),
                ToolParameter("amount", "number", required=True),
                ToolParameter("currency", "string", required=True),
            ],
            required_permission="payments:write",
        ),
        handler=lambda invoice_number, amount, currency, **_: billing_service.create_payment(
            invoice_number, Decimal(str(amount)), currency
        ),
    )

    registry.register(
        ToolDefinition(
            name="cancel_payment",
            description="Cancel a previously created payment by its payment id.",
            parameters=[ToolParameter("payment_id", "string", required=True)],
            required_permission="payments:write",
        ),
        handler=lambda payment_id, **_: billing_service.cancel_payment(payment_id),
    )

    registry.register(
        ToolDefinition(
            name="get_account_balance",
            description="Get the total outstanding balance for a customer.",
            parameters=[ToolParameter("customer_name", "string", required=True)],
            required_permission="invoices:read",
        ),
        handler=lambda customer_name, **_: {
            "customer_name": customer_name,
            "balance": str(billing_service.get_account_balance(customer_name)),
        },
    )


def _invoice_to_dict(invoice) -> dict:
    if invoice is None:
        raise LookupError("Invoice not found")
    return {
        "invoice_number": invoice.invoice_number,
        "customer_name": invoice.customer_name,
        "amount": str(invoice.amount),
        "currency": invoice.currency,
        "status": invoice.status,
        "due_date": invoice.due_date.isoformat(),
        "line_items": [
            {"product": li.product, "quantity": str(li.quantity), "unit_price": str(li.unit_price),
             "line_total": str(li.line_total)}
            for li in getattr(invoice, "line_items", [])
        ],
    }
