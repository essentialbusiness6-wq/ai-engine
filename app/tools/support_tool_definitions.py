"""
Registers support-assistant tools. Called from container.py alongside
register_billing_tools -- both sets of tools live in the SAME registry, so
one engine instance can serve as both the invoicing business assistant and
the support assistant. The LLM picks whichever tool fits a given message;
your frontend doesn't need to know which "mode" it's in.
"""
from __future__ import annotations

from app.domain.entities import ToolDefinition, ToolParameter
from app.domain.interfaces import IToolRegistry
from app.tools.support_services import InMemoryKnowledgeBaseService


def register_support_tools(registry: IToolRegistry, kb_service: InMemoryKnowledgeBaseService) -> None:

    registry.register(
        ToolDefinition(
            name="search_knowledge_base",
            description=(
                "Search the help center / knowledge base for an answer to a customer's "
                "question (e.g. how to reset a password, cancel a subscription, get a refund)."
            ),
            parameters=[ToolParameter("query", "string", required=True)],
            required_permission=None,  # read-only help content, open to any authenticated user
        ),
        handler=lambda query, **_: kb_service.search(query),
    )

    registry.register(
        ToolDefinition(
            name="create_support_ticket",
            description=(
                "Open a new support ticket when the knowledge base doesn't resolve the "
                "customer's issue and a human needs to follow up."
            ),
            parameters=[
                ToolParameter("customer_user_id", "string", required=True),
                ToolParameter("subject", "string", required=True),
                ToolParameter("description", "string", required=True),
            ],
            required_permission="support:write",
        ),
        handler=lambda customer_user_id, subject, description, **_: _ticket_to_dict(
            kb_service.create_ticket(customer_user_id, subject, description)
        ),
    )

    registry.register(
        ToolDefinition(
            name="escalate_to_human",
            description="Escalate an existing support ticket to a human agent immediately.",
            parameters=[ToolParameter("ticket_id", "string", required=True)],
            required_permission="support:write",
        ),
        handler=lambda ticket_id, **_: _ticket_to_dict(kb_service.escalate(ticket_id)),
    )


def _ticket_to_dict(ticket) -> dict:
    return {
        "ticket_id": ticket.ticket_id,
        "customer_user_id": ticket.customer_user_id,
        "subject": ticket.subject,
        "description": ticket.description,
        "status": ticket.status,
    }
