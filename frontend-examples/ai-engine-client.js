/**
 * ai-engine-client.js
 *
 * Same client as ai-engine-client.ts, written directly in plain JS for
 * projects with no TypeScript build step at all (e.g. a plain
 * Create-React-App/Vite JS project, or vanilla HTML+JS). If your project
 * DOES use TypeScript, prefer ai-engine-client.ts instead -- it's the
 * fully-typed, verified-compiling source of truth; this file is a
 * hand-kept runtime-equivalent copy for JS-only consumers.
 */

export class AIEngineError extends Error {
  constructor(errorCode, message, status, requestId) {
    super(message);
    this.errorCode = errorCode;
    this.status = status;
    this.requestId = requestId;
  }
}

export class AIEngineClient {
  /**
   * @param {{ baseUrl: string, getAuthToken: () => Promise<string> }} options
   */
  constructor({ baseUrl, getAuthToken }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.getAuthToken = getAuthToken;
  }

  async interpret(conversationId, text, { confirmed = false } = {}) {
    return this._postJson("/v1/interpret", { conversation_id: conversationId, text, confirmed });
  }

  async confirm(conversationId, text) {
    return this.interpret(conversationId, text, { confirmed: true });
  }

  async speak(conversationId, audio, { confirmed = false } = {}) {
    const token = await this.getAuthToken();
    const form = new FormData();
    form.append("audio", audio, "recording.wav");
    form.append("conversation_id", conversationId);
    form.append("confirmed", String(confirmed));

    const res = await fetch(`${this.baseUrl}/v1/speech`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
    return this._parseOrThrow(res);
  }

  async listTools() {
    const token = await this.getAuthToken();
    const res = await fetch(`${this.baseUrl}/v1/tools`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return this._parseOrThrow(res);
  }

  async _postJson(path, body) {
    const token = await this.getAuthToken();
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return this._parseOrThrow(res);
  }

  async _parseOrThrow(res) {
    const data = await res.json();
    if (!res.ok) {
      throw new AIEngineError(
        data.error_code ?? "unknown_error",
        data.message ?? `Request failed (${res.status})`,
        res.status,
        data.request_id
      );
    }
    return data;
  }
}

export default AIEngineClient;
