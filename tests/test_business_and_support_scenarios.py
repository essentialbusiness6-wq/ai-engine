"""
Full-pipeline integration tests for the two real-world use cases:
  1. Business assistant: free-text multi-item invoice creation.
  2. Support assistant: FAQ-style question answered directly, and a
     fallback to ticket creation.

Uses the same FakeAnthropicClient pattern as test_anthropic_llm_provider.py
so these run with no network access / API key, but exercise the REAL
orchestrator, tool registry, tool executor, and both tool-definition sets.
"""
from dataclasses import dataclass, field

from app.domain.entities import ExecutionStatus
from app.domain.interfaces import Principal
from app.orchestrator.pipeline import AIEngineOrchestrator
from app.services.confidence_scorer import WeightedConfidenceScorer
from app.services.context_resolver import SlotBasedContextResolver
from app.services.conversation_manager import InMemoryConversationStore
from app.services.entity_extractor import RegexEntityExtractor
from app.services.llm_provider import AnthropicLLMProvider
from app.services.intent_recognizer import TfidfIntentRecognizer
from app.services.text_normalizer import RuleBasedTextNormalizer
from app.services.tool_executor import RetryingToolExecutor
from app.services.tool_registry import InMemoryToolRegistry
from app.tools.backend_services import InMemoryBillingService
from app.tools.support_services import InMemoryKnowledgeBaseService
from app.tools.support_tool_definitions import register_support_tools
from app.tools.tool_definitions import register_billing_tools


@dataclass
class FakeToolUseBlock:
    type: str = "tool_use"
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class FakeResponse:
    content: list


class FakeMessagesAPI:
    def __init__(self, scripted_response):
        self._scripted_response = scripted_response

    def create(self, **kwargs):
        return self._scripted_response


class FakeAnthropicClient:
    def __init__(self, scripted_response):
        self.messages = FakeMessagesAPI(scripted_response)


def make_orchestrator(fake_llm_response):
    registry = InMemoryToolRegistry()
    register_billing_tools(registry, InMemoryBillingService())
    register_support_tools(registry, InMemoryKnowledgeBaseService())
    executor = RetryingToolExecutor(registry)
    fake_client = FakeAnthropicClient(fake_llm_response)
    llm_provider = AnthropicLLMProvider(client=fake_client)

    return AIEngineOrchestrator(
        conversation_store=InMemoryConversationStore(),
        text_normalizer=RuleBasedTextNormalizer(),
        intent_recognizer=TfidfIntentRecognizer(),
        entity_extractor=RegexEntityExtractor(),
        context_resolver=SlotBasedContextResolver(),
        llm_provider=llm_provider,
        tool_registry=registry,
        tool_executor=executor,
        confidence_scorer=WeightedConfidenceScorer(),
    )


def write_principal():
    return Principal(user_id="u1", tenant_id="t1",
                      permissions={"invoices:write", "invoices:read", "support:write"}, roles=set())


def test_business_assistant_creates_multi_item_invoice_from_free_text():
    """The exact user scenario: typo-laden, implicit multi-item order."""
    fake_response = FakeResponse(content=[FakeToolUseBlock(
        name="create_invoice",
        input={
            "customer_name": "John",
            "line_items": [{"product": "guns", "quantity": 3, "unit_price": 1000}],
            "due_date": "2026-08-13",
        },
    )])
    orchestrator = make_orchestrator(fake_response)

    response = orchestrator.handle_text_request(
        "biz-1",
        "create an invoice for jpohn, he but 3 guns 1000 for each one, "
        "the invoice should be due by 13 august 2026",
        write_principal(),
        confirmed=True,  # write tool -> would need confirmation first in a real UI
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.tool_name == "create_invoice"
    assert response.data["customer_name"] == "John"
    assert response.data["amount"] == "3000"
    assert response.data["due_date"] == "2026-08-13"
    assert response.data["line_items"][0]["quantity"] == "3"


def test_business_assistant_first_pass_requires_confirmation_for_invoice_creation():
    fake_response = FakeResponse(content=[FakeToolUseBlock(
        name="create_invoice",
        input={"customer_name": "John", "line_items": [{"product": "guns", "quantity": 3, "unit_price": 1000}],
               "due_date": "2026-08-13"},
    )])
    orchestrator = make_orchestrator(fake_response)
    response = orchestrator.handle_text_request(
        "biz-2", "create an invoice for John, 3 guns at 1000 each, due 13 aug 2026",
        write_principal(), confirmed=False,
    )
    assert response.status == ExecutionStatus.PENDING_CONFIRMATION


def test_support_assistant_answers_faq_directly_without_touching_business_logic():
    fake_response = FakeResponse(content=[FakeToolUseBlock(
        name="respond_directly",
        input={"message": "You can reset your password from Settings > Security > Reset Password."},
    )])
    orchestrator = make_orchestrator(fake_response)

    response = orchestrator.handle_text_request(
        "support-1", "how do i reset my password", write_principal(),
    )

    assert response.status == ExecutionStatus.ANSWERED
    assert response.tool_name is None          # no business logic was touched
    assert "Settings" in response.message


def test_support_assistant_creates_ticket_when_escalation_needed():
    fake_response = FakeResponse(content=[FakeToolUseBlock(
        name="create_support_ticket",
        input={"customer_user_id": "u1", "subject": "Billing issue",
               "description": "Customer was double-charged for last invoice"},
    )])
    orchestrator = make_orchestrator(fake_response)

    response = orchestrator.handle_text_request(
        "support-2", "I was charged twice for my last invoice, please help",
        write_principal(), confirmed=True,
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.tool_name == "create_support_ticket"
    assert response.data["status"] == "open"
