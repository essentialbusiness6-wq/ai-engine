"""
AI Engine Orchestrator.

This is the ONLY class that knows about the full pipeline shape. It
composes every service purely through the interfaces in
app.domain.interfaces (constructor-injected), which is what makes each
stage independently replaceable and independently testable, and is the
concrete embodiment of the "swap the LLM without touching business logic"
requirement: this class never imports a concrete LLM/intent/entity class,
only the abstract interfaces.

Pipeline stages, in order:
  1. Text Normalizer     - clean spelling/abbreviations
  2. Intent Recognizer   - classify the normalized text
  3. Entity Extractor    - pull structured values out of the normalized text
  4. Context Resolver    - merge entities with conversation history/backend lookups
  5. LLM Provider         - choose a tool + arguments (language understanding only)
  6. Confidence Scorer    - decide execute / confirm / clarify
  7. Tool Executor        - the ONLY component that runs backend business logic

Every step is wrapped so that failures produce a structured EngineResponse
rather than propagating raw exceptions to the API layer.
"""
from __future__ import annotations

import logging

from app.domain.entities import (
    ConversationContext,
    EngineResponse,
    ExecutionStatus,
    Message,
    NO_TOOL_NAME,
    RESPOND_TOOL_NAME,
    Role,
)
from app.domain.interfaces import (
    Principal,
    IAuditLogger,
    IConversationStore,
    IContextResolver,
    IEntityExtractor,
    IIntentRecognizer,
    ILLMProvider,
    IToolExecutor,
    IToolRegistry,
    ITextNormalizer,
)
from app.services.confidence_scorer import WeightedConfidenceScorer

logger = logging.getLogger("ai_engine.orchestrator")


class AIEngineOrchestrator:
    def __init__(
        self,
        conversation_store: IConversationStore,
        text_normalizer: ITextNormalizer,
        intent_recognizer: IIntentRecognizer,
        entity_extractor: IEntityExtractor,
        context_resolver: IContextResolver,
        llm_provider: ILLMProvider,
        tool_registry: IToolRegistry,
        tool_executor: IToolExecutor,
        confidence_scorer: WeightedConfidenceScorer | None = None,
        audit_logger: IAuditLogger | None = None,
    ):
        self._conversations = conversation_store
        self._normalizer = text_normalizer
        self._intent_recognizer = intent_recognizer
        self._entity_extractor = entity_extractor
        self._context_resolver = context_resolver
        self._llm = llm_provider
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._confidence_scorer = confidence_scorer or WeightedConfidenceScorer()
        self._audit_logger = audit_logger

    def handle_text_request(
        self,
        conversation_id: str,
        text: str,
        principal: Principal,
        confirmed: bool = False,
    ) -> EngineResponse:
        """
        confirmed=True means the caller is re-submitting a previously
        "needs confirmation" turn after the user explicitly agreed
        (e.g. clicked "Yes, pay it") -- this lets write-tools with medium
        confidence be executed on the second pass without re-lowering the
        bar for every request.
        """
        context = self._conversations.get_or_create(conversation_id, principal.user_id, principal.tenant_id)
        context.add_message(Message(role=Role.USER, content=text))

        try:
            normalized = self._normalizer.normalize(text)
            intent = self._intent_recognizer.recognize(normalized)
            entities = self._entity_extractor.extract(normalized)
            resolved_slots = self._context_resolver.resolve(entities, context)

            # NOTE: we deliberately do NOT short-circuit here even if the
            # lightweight intent recognizer reports "unknown" -- a real LLM
            # provider can still correctly handle compound/implicit
            # requests (e.g. multi-item invoice creation) or support/FAQ
            # questions that a TF-IDF classifier wasn't trained to
            # recognize. `intent` is passed to the LLM as a hint only.

            available_tools = self._tool_registry.list_definitions()
            tool_call = self._llm.select_tool(
                user_text=normalized,
                intent=intent,
                resolved_slots=resolved_slots,
                available_tools=available_tools,
                history=context.recent_messages(),
            )

            if tool_call.tool_name == RESPOND_TOOL_NAME:
                # The LLM chose to answer directly (support/FAQ/small talk)
                # rather than call a tool. This never touches the
                # ToolExecutor or any backend service -- it's pure language
                # output, so there is no business-logic risk to gate on.
                answer = tool_call.arguments.get("message", "")
                response = EngineResponse(
                    conversation_id=context.conversation_id,
                    tool_name=None,
                    arguments={},
                    confidence=tool_call.confidence,
                    status=ExecutionStatus.ANSWERED,
                    message=answer,
                )
                self._finish_turn(context, response)
                return response

            if tool_call.tool_name == NO_TOOL_NAME:
                response = self._clarification_response(
                    context.conversation_id, intent.confidence,
                    "I understood your message, but I don't have a tool to act on it yet."
                )
                self._finish_turn(context, response)
                return response

            decision = self._confidence_scorer.score(
                intent_confidence=intent.confidence,
                tool_confidence=tool_call.confidence,
                tool_name=tool_call.tool_name,
            )

            if decision.action == "clarify":
                response = self._clarification_response(
                    context.conversation_id, decision.final_confidence,
                    "I'm not confident enough about what you're asking. Could you provide more detail?"
                )
                self._finish_turn(context, response)
                return response

            if decision.action == "confirm" and not confirmed:
                response = EngineResponse(
                    conversation_id=context.conversation_id,
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    confidence=decision.final_confidence,
                    status=ExecutionStatus.PENDING_CONFIRMATION,
                    message=self._confirmation_message(tool_call.tool_name, tool_call.arguments),
                )
                self._finish_turn(context, response)
                return response

            execution_result = self._tool_executor.execute(tool_call, principal)
            response = self._to_engine_response(
                context.conversation_id, tool_call, decision.final_confidence, execution_result
            )
            self._finish_turn(context, response)
            return response

        except Exception:
            logger.exception("Unhandled error processing conversation %s", conversation_id)
            response = EngineResponse(
                conversation_id=context.conversation_id,
                tool_name=None,
                arguments={},
                confidence=0.0,
                status=ExecutionStatus.FAILED,
                message="Something went wrong while processing your request. Please try again.",
            )
            self._finish_turn(context, response)
            return response

    # -- helpers ---------------------------------------------------------

    def _finish_turn(self, context: ConversationContext, response: EngineResponse) -> None:
        context.add_message(Message(role=Role.ASSISTANT, content=response.message))
        self._conversations.save(context)

    def _clarification_response(self, conversation_id: str, confidence: float, message: str) -> EngineResponse:
        return EngineResponse(
            conversation_id=conversation_id, tool_name=None, arguments={},
            confidence=confidence, status=ExecutionStatus.NEEDS_CLARIFICATION, message=message,
        )

    def _confirmation_message(self, tool_name: str, arguments: dict) -> str:
        pretty_args = ", ".join(f"{k}={v}" for k, v in arguments.items())
        return f"I'd like to run '{tool_name}' with {pretty_args or 'no arguments'}. Should I proceed?"

    def _to_engine_response(self, conversation_id, tool_call, confidence, execution_result) -> EngineResponse:
        status = execution_result.status
        if status == ExecutionStatus.SUCCESS:
            message = f"Done: {tool_call.tool_name} completed successfully."
        elif status == ExecutionStatus.REJECTED:
            message = "You don't have permission to perform that action."
        elif status == ExecutionStatus.NEEDS_CLARIFICATION:
            message = execution_result.error_message or "I need more information to do that."
        else:
            message = execution_result.error_message or "I couldn't complete that request."

        return EngineResponse(
            conversation_id=conversation_id,
            tool_name=tool_call.tool_name,
            arguments=tool_call.arguments,
            confidence=confidence,
            status=status,
            message=message,
            data=execution_result.data,
        )
