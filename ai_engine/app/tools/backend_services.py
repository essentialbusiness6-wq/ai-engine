"""
Example "existing backend services".

In a real deployment, these would be your actual billing/payments
microservices (called over HTTP/gRPC, or imported as an internal library).
The AI Engine must NEVER reimplement this logic -- it only calls it. These
stand-ins exist so the rest of the engine (tool registry, executor,
orchestrator) has something real to invoke and test against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass
class LineItem:
    product: str
    quantity: Decimal
    unit_price: Decimal

    @property
    def line_total(self) -> Decimal:
        return self.quantity * self.unit_price


@dataclass
class Invoice:
    invoice_number: str
    customer_name: str
    amount: Decimal
    currency: str
    status: str  # "paid" | "unpaid" | "overdue"
    due_date: date
    line_items: list[LineItem] = field(default_factory=list)


class InMemoryBillingService:
    """Stand-in for a real billing microservice / repository."""

    def __init__(self):
        self._invoices: dict[str, Invoice] = {
            "12345": Invoice("12345", "John Smith", Decimal("500.00"), "USD", "unpaid", date(2026, 8, 1)),
            "67890": Invoice("67890", "Jane Doe", Decimal("1200.00"), "EUR", "overdue", date(2026, 6, 15)),
        }
        self._payments: dict[str, dict] = {}

    def get_invoice(self, invoice_number: str) -> Invoice | None:
        return self._invoices.get(invoice_number)

    def create_invoice(
        self,
        customer_name: str,
        line_items: list[dict],
        due_date: date,
        currency: str = "USD",
    ) -> Invoice:
        if not line_items:
            raise ValueError("An invoice needs at least one line item")

        items = [
            LineItem(
                product=li["product"],
                quantity=Decimal(str(li["quantity"])),
                unit_price=Decimal(str(li["unit_price"])),
            )
            for li in line_items
        ]
        total = sum((item.line_total for item in items), Decimal("0"))
        invoice_number = f"INV-{len(self._invoices) + 1000}"

        invoice = Invoice(
            invoice_number=invoice_number,
            customer_name=customer_name,
            amount=total,
            currency=currency,
            status="unpaid",
            due_date=due_date,
            line_items=items,
        )
        self._invoices[invoice_number] = invoice
        return invoice

    def list_invoices(self, status: str | None = None) -> list[Invoice]:
        invoices = list(self._invoices.values())
        if status:
            invoices = [i for i in invoices if i.status == status]
        return invoices

    def create_payment(self, invoice_number: str, amount: Decimal, currency: str) -> dict:
        invoice = self._invoices.get(invoice_number)
        if invoice is None:
            raise LookupError(f"Invoice {invoice_number} not found")
        if invoice.status == "paid":
            raise ValueError(f"Invoice {invoice_number} is already paid")
        payment_id = f"PMT-{invoice_number}-{len(self._payments) + 1}"
        self._payments[payment_id] = {
            "payment_id": payment_id, "invoice_number": invoice_number,
            "amount": amount, "currency": currency, "status": "processed",
        }
        invoice.status = "paid"
        return self._payments[payment_id]

    def cancel_payment(self, payment_id: str) -> dict:
        payment = self._payments.get(payment_id)
        if payment is None:
            raise LookupError(f"Payment {payment_id} not found")
        payment["status"] = "cancelled"
        return payment

    def get_account_balance(self, customer_name: str) -> Decimal:
        return sum(
            (inv.amount for inv in self._invoices.values()
             if inv.customer_name == customer_name and inv.status != "paid"),
            Decimal("0"),
        )

    def find_latest_open_invoice_for_customer(self, customer_name: str) -> str | None:
        open_invoices = [
            inv for inv in self._invoices.values()
            if inv.customer_name == customer_name and inv.status != "paid"
        ]
        if not open_invoices:
            return None
        return sorted(open_invoices, key=lambda i: i.due_date)[0].invoice_number
