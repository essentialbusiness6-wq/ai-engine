"""
Support-assistant tools.

These wire up a customer-support use case alongside the invoicing one, on
the SAME engine: knowledge-base lookup, support ticket creation, and
escalation to a human agent. Swap InMemoryKnowledgeBaseService for your real
helpdesk/KB system (Zendesk, Intercom, a internal docs API, etc.) the same
way you'd swap InMemoryBillingService -- only this file changes.

Design note: `search_knowledge_base` is a READ tool the LLM can call to
ground its own answer in real content (so support replies aren't just the
model improvising). The orchestrator's RESPOND_TOOL_NAME path is for when
the LLM decides no tool is needed at all (e.g. "hi", "thank you").
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class SupportTicket:
    ticket_id: str
    customer_user_id: str
    subject: str
    description: str
    status: str = "open"


class InMemoryKnowledgeBaseService:
    """Stand-in for a real KB/helpdesk search API."""

    def __init__(self):
        self._articles: dict[str, str] = {
            "reset password": "Go to Settings > Security > Reset Password, then check your email for the reset link.",
            "cancel subscription": "Go to Billing > Subscription > Cancel. Your access continues until the end of the current billing period.",
            "invoice not received": "Invoices are emailed within 5 minutes of creation. Check spam, or re-send it from Billing > Invoices > Resend.",
            "change payment method": "Go to Billing > Payment Methods > Add New, then set it as default and remove the old one.",
            "refund policy": "Refunds are available within 14 days of payment for unused services; contact support to request one.",
        }
        self._tickets: dict[str, SupportTicket] = {}

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_lower = query.lower()
        scored = []
        for topic, content in self._articles.items():
            overlap = len(set(topic.split()) & set(query_lower.split()))
            if overlap > 0 or topic in query_lower:
                scored.append((overlap, topic, content))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [{"topic": topic, "content": content} for _, topic, content in scored[:top_k]]

    def create_ticket(self, customer_user_id: str, subject: str, description: str) -> SupportTicket:
        ticket = SupportTicket(
            ticket_id=f"TCK-{uuid.uuid4().hex[:8].upper()}",
            customer_user_id=customer_user_id,
            subject=subject,
            description=description,
        )
        self._tickets[ticket.ticket_id] = ticket
        return ticket

    def escalate(self, ticket_id: str) -> SupportTicket:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise LookupError(f"Ticket {ticket_id} not found")
        ticket.status = "escalated_to_human"
        return ticket
