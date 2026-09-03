# Step-by-Step: Support Assistant + Business Assistant Setup

This engine can run BOTH use cases from one deployment: an AI support-care
agent and an AI business assistant (e.g. invoice creation) that understands
natural, typo-laden language. Both go through the exact same `/v1/interpret`
endpoint — what differs is which **tools** are registered and what
**permissions** a given user's token carries.

## 0. The one thing that makes complex language work: use the real LLM

`MockLLMProvider` (no API key needed) only does simple, single-slot intents.
It **cannot** parse "3 guns 1000 for each one, due 13 august 2026" into
structured line items — nothing rule-based reliably can. For that you need
`AnthropicLLMProvider`, which is now fully implemented (not a stub).

```bash
export AI_ENGINE_ANTHROPIC_API_KEY=sk-ant-...   # or plain ANTHROPIC_API_KEY
```

`app/container.py` automatically switches to the real provider the moment
this env var is present — no code changes needed. Without it, the engine
still runs (useful for CI/tests) but only handles simple commands.

## 1. Business assistant (invoice creation, order-style language)

**Already wired up** in `app/tools/tool_definitions.py` → `create_invoice`.
It accepts free-form line items via a JSON schema (`parameters_schema` on
the `ToolDefinition`), so Claude extracts:

```
"create an invoice for jpohn, he but 3 guns 1000 for each one,
 due by 13 august 2026"
→ create_invoice(
    customer_name="John",
    line_items=[{product:"guns", quantity:3, unit_price:1000}],
    due_date="2026-08-13"
  )
```

Typos ("jpohn", "he but") are absorbed by the LLM itself — this is exactly
what real language models are good at and regex isn't. The Text
Normalizer/Entity Extractor still run first and feed hints into
`resolved_slots`, but the LLM does the heavy lifting for compound requests.

**To connect it to your real invoice platform:** edit `app/container.py`,
swap `InMemoryBillingService()` for `RealInvoicePlatformService(...)` (see
`app/tools/real_backend_services.py` for the adapter pattern — implement
`create_invoice`, `get_invoice`, etc. against your actual API/DB).

**Permissions:** a user's JWT needs `invoices:write` to create invoices and
`invoices:read` to check status/list them. Because `create_invoice` starts
with `create_`, the confidence scorer treats it as a **write action** and
requires confirmation unless confidence is very high — this is your safety
net against the LLM mis-parsing an invoice and it silently going out. Always
show the `pending_confirmation` message to the user and only re-send with
`confirmed: true` after they approve (see the frontend section below).

## 2. Support-care assistant (answers users, opens tickets)

Wired up in `app/tools/support_tool_definitions.py`:

- `search_knowledge_base` — read-only, no permission required, lets the LLM
  ground its answer in real KB content instead of improvising.
- `create_support_ticket` — requires `support:write`.
- `escalate_to_human` — requires `support:write`.
- Plus the built-in `respond_directly` path: when no tool fits (greetings,
  thanks, or a question fully answerable from the KB search results already
  in context), the LLM just replies in plain language. This returns
  `status: "answered"` with `tool_name: null` — **no backend service is ever
  called**, so there's no business-logic risk from a "just chatting" turn.

**To connect it to your real helpdesk:** swap
`InMemoryKnowledgeBaseService` in `app/container.py` for a real client
against Zendesk/Intercom/your internal docs API — implement `search`,
`create_ticket`, `escalate` against it.

**Permissions:** give support users `support:write`; KB search needs
nothing since it's read-only and marked `required_permission=None`.

## 3. Running both from one engine vs. splitting them

Both tool sets are registered into the same `InMemoryToolRegistry` in
`app/container.py` by default — one deployment serves both. This is usually
what you want (shared conversation/auth/rate-limiting infrastructure). If
you truly want separate deployments (e.g. different scaling profiles),
just run two instances of the app with different `register_*_tools` calls
commented out — nothing else changes.

## 4. Issuing tokens for each surface

Your own backend decides who gets which permissions when it mints the JWT
(see `app/core/security.py::issue_token`):

```python
# A logged-in customer using the support widget:
support_token = issue_token(user_id, tenant_id, permissions=["support:write"], roles=[])

# An internal ops user using the invoicing assistant:
business_token = issue_token(user_id, tenant_id,
                              permissions=["invoices:read", "invoices:write"], roles=[])

# Someone who should get both surfaces in one app:
combined_token = issue_token(user_id, tenant_id,
                              permissions=["invoices:read", "invoices:write", "support:write"], roles=[])
```

The AI Engine doesn't know or care which "product" is calling it — only
what permissions the token carries and which tools are registered.

## 5. Calling it from the frontend (JavaScript or TypeScript)

Use `frontend-examples/ai-engine-client.ts` (TypeScript, fully typed, tested
to compile in strict mode) or `frontend-examples/ai-engine-client.js` (same
client, plain JS, no build step required). Both expose the identical API:

```ts
import { AIEngineClient } from "./ai-engine-client"; // or .js — same usage

const client = new AIEngineClient({
  baseUrl: "https://ai.yourdomain.com",
  getAuthToken: async () => {
    const r = await fetch("/api/ai-token", { credentials: "include" }); // your backend mints it
    return (await r.json()).token;
  },
});

const conversationId = crypto.randomUUID(); // one per chat session; reused across turns

const response = await client.interpret(conversationId, userText);

switch (response.status) {
  case "success":              /* action completed — use response.data */ break;
  case "answered":             /* plain reply — support/FAQ, show response.message */ break;
  case "pending_confirmation": /* show Yes/No, then client.confirm(conversationId, userText) */ break;
  case "needs_clarification":  /* show response.message, let user rephrase */ break;
  case "rejected":             /* permission denied */ break;
  case "failed":                /* show error */ break;
}
```

Full working React examples (identical logic, one per use case):
`frontend-examples/BusinessAssistant.tsx` and `frontend-examples/SupportAssistant.tsx`.

## 6. Voice input for either assistant

Same idea, just call `client.speak(conversationId, audioBlob)` instead of
`interpret` — it transcribes then runs through the identical pipeline, so
everything above (confirmation flow, tools, permissions) applies unchanged.
Enable real transcription by installing `openai-whisper` and swapping
`MockSpeechToText` for `WhisperSpeechToText()` in `app/container.py`.

## 7. Sanity-checklist before going live

- [ ] `AI_ENGINE_ANTHROPIC_API_KEY` set in your deployment environment.
- [ ] `AI_ENGINE_JWT_SECRET` set to a strong, random 32+ byte value (matching whatever your backend uses to sign tokens).
- [ ] `InMemoryConversationStore` swapped for a Redis/Postgres-backed store if you run more than one process (in-memory state won't be shared across instances).
- [ ] `InMemoryBillingService` / `InMemoryKnowledgeBaseService` swapped for real backend clients.
- [ ] Confirm write-tool confirmation flow is actually shown in your UI — never auto-send `confirmed: true` without the user seeing the pending action.
- [ ] Rate limit (`AI_ENGINE_RATE_LIMIT_REQUESTS_PER_MINUTE`) tuned for your expected traffic.
- [ ] Run `pytest -v` after any backend-service swap to make sure the tool contracts still match what `tool_definitions.py` expects.
