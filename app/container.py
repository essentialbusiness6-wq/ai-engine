"""
Composition Root.

This is the ONE place in the whole codebase where concrete implementations
are chosen and wired together. Every other module depends only on
abstractions (app.domain.interfaces). To swap the LLM provider, the intent
recognizer, or the conversation store for a different implementation, you
edit ONLY this function -- no business logic anywhere else changes.
"""
from __future__ import annotations

import os

from dataclasses import dataclass

from app.core.config import settings
from app.core.rate_limiter import SlidingWindowRateLimiter
from app.domain.interfaces import (
    IConversationStore,
    IContextResolver,
    IEntityExtractor,
    IIntentRecognizer,
    ILLMProvider,
    ISpeechToText,
    IToolExecutor,
    IToolRegistry,
    ITextNormalizer,
)
from app.orchestrator.pipeline import AIEngineOrchestrator
from app.services.audit_logger import StandardAuditLogger
from app.services.confidence_scorer import WeightedConfidenceScorer
from app.services.context_resolver import SlotBasedContextResolver
from app.services.conversation_manager import InMemoryConversationStore
from app.services.entity_extractor import RegexEntityExtractor
from app.services.intent_recognizer import TfidfIntentRecognizer
from app.services.llm_provider import AnthropicLLMProvider, MockLLMProvider
from app.services.speech_to_text import MockSpeechToText
from app.services.text_normalizer import RuleBasedTextNormalizer
from app.services.tool_executor import RetryingToolExecutor
from app.services.tool_registry import InMemoryToolRegistry
from app.tools.backend_services import InMemoryBillingService
from app.tools.support_services import InMemoryKnowledgeBaseService
from app.tools.support_tool_definitions import register_support_tools
from app.tools.tool_definitions import register_billing_tools


@dataclass
class Container:
    """Holds every wired-up service the API layer needs to reach."""
    conversation_store: IConversationStore
    text_normalizer: ITextNormalizer
    intent_recognizer: IIntentRecognizer
    entity_extractor: IEntityExtractor
    context_resolver: IContextResolver
    llm_provider: ILLMProvider
    tool_registry: IToolRegistry
    tool_executor: IToolExecutor
    speech_to_text: ISpeechToText
    orchestrator: AIEngineOrchestrator
    rate_limiter: SlidingWindowRateLimiter


def build_container() -> Container:
    """
    Builds the app-wide wiring. If AI_ENGINE_ANTHROPIC_API_KEY (or the
    standard ANTHROPIC_API_KEY) is set, the REAL Claude tool-calling
    provider is used -- this is what you want for production and for
    anything beyond simple single-slot commands (e.g. parsing a full
    invoice with several line items from one free-text message, or
    answering open-ended support questions). Without a key, it falls back
    to MockLLMProvider so the engine still runs end-to-end for local dev
    and CI without any credentials.

    Both the billing/invoicing tools AND the support tools are registered
    into the SAME tool registry, so one deployment of this engine serves
    both use cases -- the LLM picks whichever tool fits a given message.
    If you only want one use case active, just register one of the two
    tool sets below.

    To point at your real backends instead of the in-memory stand-ins:
      - Replace InMemoryConversationStore with a Redis/Postgres-backed
        IConversationStore implementation.
      - Replace InMemoryBillingService with RealInvoicePlatformService (see
        app/tools/real_backend_services.py) or your own service client.
      - Replace InMemoryKnowledgeBaseService with your real helpdesk/KB API.
      - Replace MockSpeechToText with WhisperSpeechToText() (requires
        `pip install openai-whisper`).
    None of these changes require touching text_normalizer, intent
    recognition, entity extraction, context resolution, confidence
    scoring, or the orchestrator itself.
    """
    conversation_store = InMemoryConversationStore(
        max_messages_per_conversation=settings.conversation_max_messages,
        ttl_seconds=settings.conversation_ttl_seconds,
    )
    text_normalizer = RuleBasedTextNormalizer()
    intent_recognizer = TfidfIntentRecognizer()
    entity_extractor = RegexEntityExtractor()

    billing_service = InMemoryBillingService()
    kb_service = InMemoryKnowledgeBaseService()

    def backend_lookup(slot_name: str, resolved_so_far: dict):
        if slot_name == "invoice_number" and resolved_so_far.get("person_name"):
            return billing_service.find_latest_open_invoice_for_customer(resolved_so_far["person_name"])
        return None

    context_resolver = SlotBasedContextResolver(backend_lookup=backend_lookup)

    anthropic_api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_api_key:
        import anthropic
        llm_provider = AnthropicLLMProvider(
            client=anthropic.Anthropic(api_key=anthropic_api_key),
            model=settings.anthropic_model,
        )
    else:
        llm_provider = MockLLMProvider()

    tool_registry = InMemoryToolRegistry()
    register_billing_tools(tool_registry, billing_service)
    register_support_tools(tool_registry, kb_service)

    audit_logger = StandardAuditLogger()
    tool_executor = RetryingToolExecutor(tool_registry, audit_logger=audit_logger)

    confidence_scorer = WeightedConfidenceScorer(
        low_confidence_threshold=settings.low_confidence_threshold,
        high_risk_confidence_threshold=settings.high_risk_confidence_threshold,
    )

    orchestrator = AIEngineOrchestrator(
        conversation_store=conversation_store,
        text_normalizer=text_normalizer,
        intent_recognizer=intent_recognizer,
        entity_extractor=entity_extractor,
        context_resolver=context_resolver,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        confidence_scorer=confidence_scorer,
        audit_logger=audit_logger,
    )

    speech_to_text = MockSpeechToText(canned_transcript="")

    rate_limiter = SlidingWindowRateLimiter(
        max_requests_per_minute=settings.rate_limit_requests_per_minute
    )

    return Container(
        conversation_store=conversation_store,
        text_normalizer=text_normalizer,
        intent_recognizer=intent_recognizer,
        entity_extractor=entity_extractor,
        context_resolver=context_resolver,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        speech_to_text=speech_to_text,
        orchestrator=orchestrator,
        rate_limiter=rate_limiter,
    )
