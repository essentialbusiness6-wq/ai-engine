/**
 * SupportAssistant.tsx
 *
 * Support-assistant example: general questions get answered directly
 * (status "answered", no backend action taken); issues the KB can't
 * resolve get turned into a support ticket via the create_support_ticket
 * tool -- same /v1/interpret endpoint, same client, different tools/JWT
 * permissions than the business assistant.
 */
import { useRef, useState } from "react";
import { AIEngineClient, EngineResponse } from "./ai-engine-client";

const client = new AIEngineClient({
  baseUrl: import.meta.env.VITE_AI_ENGINE_URL ?? "http://localhost:8000",
  getAuthToken: async () => {
    const r = await fetch("/api/ai-token", { credentials: "include" });
    const { token } = await r.json();
    return token;
  },
});

export default function SupportAssistant() {
  const [log, setLog] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [pendingText, setPendingText] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const conversationId = useRef(crypto.randomUUID()).current;

  async function send(text: string, confirmed = false) {
    setLog((l) => [...l, { role: "user", text }]);
    try {
      const response: EngineResponse = await client.interpret(conversationId, text, { confirmed });
      setLog((l) => [...l, { role: "assistant", text: response.message }]);

      if (response.status === "pending_confirmation") {
        setPendingText(text);
      } else {
        setPendingText(null);
      }
      // response.status === "answered"  -> plain FAQ reply, nothing else to do
      // response.status === "success" with tool_name === "create_support_ticket" -> ticket was opened
    } catch (err: any) {
      setLog((l) => [...l, { role: "assistant", text: `Error: ${err.message}` }]);
    }
  }

  return (
    <div className="max-w-lg mx-auto">
      <div className="border rounded-lg p-4 space-y-2 h-80 overflow-y-auto">
        {log.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
            <span className="inline-block px-3 py-2 rounded-lg bg-gray-100">{m.text}</span>
          </div>
        ))}
        {pendingText && (
          <div className="flex gap-2 justify-center">
            <button className="px-3 py-1 rounded bg-black text-white" onClick={() => send(pendingText, true)}>
              Yes, open a ticket
            </button>
            <button className="px-3 py-1 rounded border" onClick={() => setPendingText(null)}>
              Never mind
            </button>
          </div>
        )}
      </div>

      <form
        className="flex gap-2 mt-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (!input.trim()) return;
          send(input);
          setInput("");
        }}
      >
        <input
          className="flex-1 border rounded px-2 py-1"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="How can we help? e.g. how do I reset my password"
        />
        <button type="submit" className="px-3 py-1 rounded bg-black text-white">
          Send
        </button>
      </form>
    </div>
  );
}
