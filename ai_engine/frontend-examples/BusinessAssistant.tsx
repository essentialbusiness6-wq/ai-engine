/**
 * BusinessAssistant.tsx
 *
 * Business-assistant example: a user types/says something like
 * "create an invoice for John, 3 guns at 1000 each, due 13 august 2026"
 * and this renders the confirm step, then the created invoice.
 *
 * Uses the create_invoice tool registered in tool_definitions.py.
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

interface LineItem {
  product: string;
  quantity: string;
  unit_price: string;
  line_total: string;
}
interface InvoiceData {
  invoice_number: string;
  customer_name: string;
  amount: string;
  currency: string;
  due_date: string;
  line_items: LineItem[];
}

export default function BusinessAssistant() {
  const [log, setLog] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [invoice, setInvoice] = useState<InvoiceData | null>(null);
  const [pendingText, setPendingText] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const conversationId = useRef(crypto.randomUUID()).current;

  async function send(text: string, confirmed = false) {
    setLog((l) => [...l, { role: "user", text }]);
    try {
      const response: EngineResponse<InvoiceData> = await client.interpret(conversationId, text, { confirmed });
      setLog((l) => [...l, { role: "assistant", text: response.message }]);

      if (response.status === "pending_confirmation") {
        setPendingText(text);
      } else {
        setPendingText(null);
        if (response.status === "success" && response.tool_name === "create_invoice") {
          setInvoice(response.data);
        }
      }
    } catch (err: any) {
      setLog((l) => [...l, { role: "assistant", text: `Error: ${err.message}` }]);
    }
  }

  return (
    <div className="max-w-lg mx-auto space-y-4">
      <div className="border rounded-lg p-4 space-y-2 h-64 overflow-y-auto">
        {log.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
            <span className="inline-block px-3 py-2 rounded-lg bg-gray-100">{m.text}</span>
          </div>
        ))}
        {pendingText && (
          <div className="flex gap-2 justify-center">
            <button className="px-3 py-1 rounded bg-black text-white" onClick={() => send(pendingText, true)}>
              Confirm & Create
            </button>
            <button className="px-3 py-1 rounded border" onClick={() => setPendingText(null)}>
              Cancel
            </button>
          </div>
        )}
      </div>

      <form
        className="flex gap-2"
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
          placeholder="e.g. invoice John for 3 guns at 1000 each, due 13 aug 2026"
        />
        <button type="submit" className="px-3 py-1 rounded bg-black text-white">
          Send
        </button>
      </form>

      {invoice && (
        <div className="border rounded-lg p-4">
          <h3 className="font-semibold">Invoice {invoice.invoice_number}</h3>
          <p>Customer: {invoice.customer_name}</p>
          <p>Due: {invoice.due_date}</p>
          <table className="w-full text-sm mt-2">
            <thead>
              <tr className="text-left border-b">
                <th>Product</th><th>Qty</th><th>Unit Price</th><th>Total</th>
              </tr>
            </thead>
            <tbody>
              {invoice.line_items.map((li, i) => (
                <tr key={i}>
                  <td>{li.product}</td><td>{li.quantity}</td><td>{li.unit_price}</td><td>{li.line_total}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="font-semibold mt-2">Total: {invoice.amount} {invoice.currency}</p>
        </div>
      )}
    </div>
  );
}
