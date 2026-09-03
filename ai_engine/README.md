# AI Engine API

A standalone, reusable AI Engine that exposes REST endpoints for interpreting
natural language (text or speech) and executing backend tools through a
hybrid NLP + LLM pipeline. Designed so multiple applications can share one
engine deployment, and so the underlying LLM can be swapped without touching
any business logic.

## Architecture

Clean Architecture / hexagonal layering, dependencies point inward:

```
app/
  domain/            # Pure entities + interfaces (ports). No framework deps.
    entities.py       # Message, ConversationContext, ExtractedEntity, ToolCall, EngineResponse, ...
    interfaces.py      # IConversationStore, ITextNormalizer, IIntentRecognizer,
                        # IEntityExtractor, IContextResolver, ILLMProvider,
                        # IToolRegistry, IToolExecutor, ISpeechToText, IAuditLogger, Principal

  services/           # Concrete implementations of the domain interfaces
    conversation_manager.py   # InMemoryConversationStore (TTL + message window)
    text_normalizer.py        # RuleBasedTextNormalizer (abbreviations + spell-correct)
    intent_recognizer.py      # TfidfIntentRecognizer (TF-IDF cosine similarity)
    entity_extractor.py       # RegexEntityExtractor (dates, amounts, invoice #s, ...)
    context_resolver.py       # SlotBasedContextResolver (merges entities + history + lookups)
    llm_provider.py           # MockLLMProvider + AnthropicLLMProvider stub + injection sanitizer
    tool_registry.py          # InMemoryToolRegistry (name -> definition + handler)
    tool_executor.py          # RetryingToolExecutor (permissions, validation, retries)
    confidence_scorer.py      # WeightedConfidenceScorer (execute/confirm/clarify policy)
    audit_logger.py           # StandardAuditLogger
    speech_to_text.py         # MockSpeechToText + WhisperSpeechToText (lazy-loaded)

  tools/              # "Existing backend services" + their tool adapters
    backend_services.py       # InMemoryBillingService (stand-in for a real microservice)
    tool_definitions.py       # register_billing_tools(): wires service methods as tools

  orchestrator/
    pipeline.py         # AIEngineOrchestrator: composes every stage end-to-end

  core/
    config.py           # Settings (env-var driven)
    security.py          # JWT auth (issue_token / authenticate)
    rate_limiter.py       # SlidingWindowRateLimiter
    exceptions.py          # AIEngineError hierarchy -> consistent HTTP error shape

  api/
    schemas.py            # Pydantic request/response models
    dependencies.py         # FastAPI Depends() providers (auth, rate limit, container)
    middleware.py            # Request logging + error->JSON conversion
    routes/
      interpret.py            # POST /v1/interpret  (text)
      speech.py                # POST /v1/speech     (audio -> same pipeline)
      tools.py                  # GET  /v1/tools      (introspection)
      health.py                  # GET  /health

  container.py         # Composition root: the ONE place concrete classes are chosen
  main.py               # FastAPI app assembly (no business logic)
```

### Pipeline (executed per request, in `AIEngineOrchestrator.handle_text_request`)

1. **Text Normalizer** — expands abbreviations/shorthand, corrects common misspellings.
2. **Intent Recognizer** — TF-IDF + cosine similarity against labeled example utterances (a lightweight embedding-equivalent; swap in a transformer model later by implementing `IIntentRecognizer`).
3. **Entity Extractor** — regex/dateutil based: names, dates, relative dates, amounts, currencies, invoice numbers, payment references, emails.
4. **Context Resolver** — merges this turn's entities with prior-turn slots stored on the conversation, plus an optional backend-lookup hook (e.g. resolve a person's name to their latest open invoice).
5. **LLM Provider** — used **only** for language understanding and tool selection. Input is passed through `sanitize_for_llm()` first (prompt-injection defense). The LLM returns a tool name + arguments; it never touches SQL or a database.
6. **Confidence Scorer** — combines intent + tool-selection confidence into `execute` / `confirm` / `clarify`. Write-style tools (`create_*`, `cancel_*`, `delete_*`, ...) require confirmation unless confidence is high.
7. **Tool Executor** — the **only** component that calls backend service functions. Enforces permissions, validates arguments, retries transient failures, and converts exceptions into structured results. This is what guarantees the LLM never executes business logic directly.

### Swapping the LLM (or any other stage)

Edit only `app/container.py`:

```python
# from:
llm_provider = MockLLMProvider()
# to:
llm_provider = AnthropicLLMProvider(client=anthropic.Anthropic())
```

Nothing in `orchestrator/pipeline.py`, the services, or the API layer needs to
change, because the orchestrator depends only on `ILLMProvider`.

The same pattern applies to `IConversationStore` (swap in Redis/Postgres),
`IIntentRecognizer` (swap in a transformer embedding model), `ISpeechToText`
(swap `MockSpeechToText` for `WhisperSpeechToText`), etc.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Mint a demo JWT (in a real deployment, your consuming application's own auth
service issues these using the same shared secret):

```bash
python -c "
from app.core.security import issue_token
print(issue_token('demo-user', 'tenant-a', ['invoices:read','payments:write'], []))
"
```

Call the API:

```bash
curl -X POST http://localhost:8000/v1/interpret \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-1", "text": "whats the status of inv 12345"}'
```

Speech endpoint (multipart form, audio file + conversation_id):

```bash
curl -X POST http://localhost:8000/v1/speech \
  -H "Authorization: Bearer <token>" \
  -F "audio=@recording.wav" \
  -F "conversation_id=demo-1"
```

## Running tests

```bash
pytest -v
```

82 tests cover every module in isolation plus full end-to-end pipeline and
HTTP-layer integration tests (auth, permissions, rate limiting, confirmation
flow, prompt-injection defense).

## Docker

```bash
docker build -t ai-engine .
docker run -p 8000:8000 -e AI_ENGINE_JWT_SECRET=change-me ai-engine
```

## Response shape

Every `/v1/interpret` and `/v1/speech` call returns:

```json
{
  "conversation_id": "demo-1",
  "tool_name": "get_invoice_status",
  "arguments": {"invoice_number": "12345"},
  "confidence": 0.83,
  "status": "success",
  "message": "Done: get_invoice_status completed successfully.",
  "data": {"invoice_number": "12345", "status": "unpaid", "amount": "500.00", "currency": "USD", "due_date": "2026-08-01"},
  "request_id": "b4295c9e-...",
  "timestamp": "2026-07-27T18:58:00Z"
}
```

`status` is one of: `success`, `failed`, `needs_clarification`, `rejected`,
`pending_confirmation`.

## Security notes

- Auth is JWT bearer-token based (`app/core/security.py`); the AI Engine
  itself doesn't own user accounts, it trusts tokens issued by whichever
  application/auth-service it's embedded behind.
- Authorization is permission-string based, enforced in `RetryingToolExecutor`
  at the moment of execution — independent of whatever the LLM "decided."
- Prompt injection defense (`sanitize_for_llm`) runs before every LLM call
  and forces low confidence (→ confirmation required, or clarification) on
  suspected injection attempts.
- All security-relevant events (permission rejections, tool executions,
  injection detection) go through `StandardAuditLogger`.
- Rate limiting is per-user, sliding-window, enforced before the pipeline runs.
