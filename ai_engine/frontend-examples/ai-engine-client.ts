/**
 * ai-engine-client.ts
 *
 * A single, fully-typed client for BOTH the invoice/business-assistant and
 * the support-assistant use cases -- they're the same API, the only
 * difference is which tools your backend registers and what permissions
 * the caller's JWT carries.
 *
 * Usable from:
 *   - TypeScript projects: import directly, get full type checking.
 *   - Plain JavaScript projects: either compile this file once with `tsc`
 *     (see the one-liner below) and import the resulting .js, or just
 *     delete the type annotations -- the runtime code has zero
 *     TS-only syntax beyond types, so it degrades to valid JS trivially.
 *
 *   npx tsc ai-engine-client.ts --module esnext --target es2020 --outDir dist
 */

export type EngineStatus =
  | "success"
  | "failed"
  | "needs_clarification"
  | "rejected"
  | "pending_confirmation"
  | "answered";

export interface EngineResponse<TData = unknown> {
  conversation_id: string;
  tool_name: string | null;
  arguments: Record<string, unknown>;
  confidence: number;
  status: EngineStatus;
  message: string;
  data: TData;
  request_id: string;
  timestamp: string;
}

export interface ToolInfo {
  name: string;
  description: string;
  required_permission: string | null;
  parameters: Array<{ name: string; type: string; required: boolean }>;
}

export interface AIEngineClientOptions {
  /** e.g. "https://ai.yourdomain.com" or "http://localhost:8000" */
  baseUrl: string;
  /** Fetch/refresh a JWT from YOUR backend. Never hardcode a token here. */
  getAuthToken: () => Promise<string>;
}

export class AIEngineError extends Error {
  errorCode: string;
  status: number;
  requestId?: string;

  constructor(errorCode: string, message: string, status: number, requestId?: string) {
    super(message);
    this.errorCode = errorCode;
    this.status = status;
    this.requestId = requestId;
  }
}

export class AIEngineClient {
  private baseUrl: string;
  private getAuthToken: () => Promise<string>;

  constructor(options: AIEngineClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.getAuthToken = options.getAuthToken;
  }

  /**
   * Send one turn of natural language text. `conversationId` should stay
   * the same across a user's chat session so context (e.g. a previously
   * mentioned invoice or customer) carries across turns.
   */
  async interpret<TData = unknown>(
    conversationId: string,
    text: string,
    opts: { confirmed?: boolean } = {}
  ): Promise<EngineResponse<TData>> {
    return this.postJson<EngineResponse<TData>>("/v1/interpret", {
      conversation_id: conversationId,
      text,
      confirmed: opts.confirmed ?? false,
    });
  }

  /** Re-send the same text with confirmed=true once the user approves a `pending_confirmation` action. */
  async confirm<TData = unknown>(conversationId: string, text: string): Promise<EngineResponse<TData>> {
    return this.interpret<TData>(conversationId, text, { confirmed: true });
  }

  /** Upload recorded audio for speech-to-text + interpretation, same pipeline as interpret(). */
  async speak<TData = unknown>(
    conversationId: string,
    audio: Blob | File,
    opts: { confirmed?: boolean } = {}
  ): Promise<EngineResponse<TData>> {
    const token = await this.getAuthToken();
    const form = new FormData();
    form.append("audio", audio, "recording.wav");
    form.append("conversation_id", conversationId);
    form.append("confirmed", String(opts.confirmed ?? false));

    const res = await fetch(`${this.baseUrl}/v1/speech`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` }, // no Content-Type: browser sets multipart boundary
      body: form,
    });
    return this.parseOrThrow<EngineResponse<TData>>(res);
  }

  /** Introspect which tools this deployment currently exposes. Handy for admin/debug UIs. */
  async listTools(): Promise<ToolInfo[]> {
    const token = await this.getAuthToken();
    const res = await fetch(`${this.baseUrl}/v1/tools`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return this.parseOrThrow<ToolInfo[]>(res);
  }

  // -- internals -----------------------------------------------------

  private async postJson<T>(path: string, body: unknown): Promise<T> {
    const token = await this.getAuthToken();
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    return this.parseOrThrow<T>(res);
  }

  private async parseOrThrow<T>(res: Response): Promise<T> {
    const data = await res.json();
    if (!res.ok) {
      throw new AIEngineError(
        data.error_code ?? "unknown_error",
        data.message ?? `Request failed (${res.status})`,
        res.status,
        data.request_id
      );
    }
    return data as T;
  }
}

/* ------------------------------------------------------------------ *
 * Example usage (TypeScript or JavaScript, identical):
 *
 *   const client = new AIEngineClient({
 *     baseUrl: "https://ai.yourdomain.com",
 *     getAuthToken: async () => {
 *       const r = await fetch("/api/ai-token", { credentials: "include" }); // YOUR backend
 *       const { token } = await r.json();
 *       return token;
 *     },
 *   });
 *
 *   const conversationId = crypto.randomUUID(); // one per open chat session
 *
 *   async function handleUserMessage(text: string) {
 *     const response = await client.interpret(conversationId, text);
 *
 *     switch (response.status) {
 *       case "success":
 *         showAssistantMessage(response.message);
 *         renderResult(response.data);   // e.g. the created invoice, ticket, invoice status
 *         break;
 *       case "answered":
 *         showAssistantMessage(response.message); // support/FAQ answer, no action taken
 *         break;
 *       case "pending_confirmation":
 *         showConfirmButton(response.message, () => client.confirm(conversationId, text).then(render));
 *         break;
 *       case "needs_clarification":
 *         showAssistantMessage(response.message);
 *         break;
 *       case "rejected":
 *         showAssistantMessage("You don't have permission to do that.");
 *         break;
 *       case "failed":
 *         showAssistantMessage(response.message ?? "Something went wrong.");
 *         break;
 *     }
 *   }
 * ------------------------------------------------------------------ */
