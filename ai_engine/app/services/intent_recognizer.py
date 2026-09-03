"""
Intent Recognition layer.

Classifies normalized user text into one of a fixed set of intents using
sentence-similarity: each intent has a handful of example utterances, we
vectorize everything with TF-IDF and pick the intent whose examples are
closest (cosine similarity) to the input.

This is a deliberately lightweight stand-in for a full sentence-embedding
model (e.g. sentence-transformers). It requires no GPU and no multi-hundred
MB model download, which matters for a "lightweight NLP model" requirement
and for fast cold starts. Because it implements IIntentRecognizer, it can be
swapped for a transformer-embedding-based implementation later without any
change to the orchestrator, context resolver, or API layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.domain.entities import IntentResult
from app.domain.interfaces import IIntentRecognizer


@dataclass
class IntentExample:
    intent: str
    utterance: str


# Default training examples for a billing/invoicing-style domain assistant.
# In production these would come from a config file or a labeled dataset
# curated by the product team.
DEFAULT_EXAMPLES: list[IntentExample] = [
    IntentExample("get_invoice_status", "what is the status of invoice 123"),
    IntentExample("get_invoice_status", "check invoice status"),
    IntentExample("get_invoice_status", "has invoice been paid"),
    IntentExample("get_invoice_status", "show me the invoice details"),
    IntentExample("get_invoice_status", "is my invoice overdue"),

    IntentExample("create_payment", "pay this invoice"),
    IntentExample("create_payment", "make a payment of 500 dollars"),
    IntentExample("create_payment", "send payment for invoice 123"),
    IntentExample("create_payment", "process payment now"),
    IntentExample("create_payment", "pay it"),

    IntentExample("list_invoices", "show all my invoices"),
    IntentExample("list_invoices", "list unpaid invoices"),
    IntentExample("list_invoices", "give me overdue invoices"),
    IntentExample("list_invoices", "what invoices do i have"),

    IntentExample("get_account_balance", "what is my account balance"),
    IntentExample("get_account_balance", "how much do i owe"),
    IntentExample("get_account_balance", "check my balance"),

    IntentExample("cancel_payment", "cancel the payment"),
    IntentExample("cancel_payment", "undo that payment"),
    IntentExample("cancel_payment", "stop the transaction"),

    IntentExample("greeting", "hello"),
    IntentExample("greeting", "hi there"),
    IntentExample("greeting", "good morning"),

    IntentExample("help", "what can you do"),
    IntentExample("help", "help me"),
    IntentExample("help", "i need assistance"),
]


class TfidfIntentRecognizer(IIntentRecognizer):
    def __init__(
        self,
        examples: list[IntentExample] | None = None,
        low_confidence_threshold: float = 0.25,
    ):
        self._examples = examples if examples is not None else DEFAULT_EXAMPLES
        self._low_confidence_threshold = low_confidence_threshold
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        corpus = [ex.utterance for ex in self._examples]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def recognize(self, normalized_text: str) -> IntentResult:
        if not normalized_text.strip():
            return IntentResult(intent="unknown", confidence=0.0)

        query_vec = self._vectorizer.transform([normalized_text])
        sims = cosine_similarity(query_vec, self._matrix)[0]

        # Aggregate similarity per intent by taking the max score among that
        # intent's example utterances (nearest-neighbor style).
        scores: dict[str, float] = {}
        for score, example in zip(sims, self._examples):
            if score > scores.get(example.intent, -1.0):
                scores[example.intent] = float(score)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked or ranked[0][1] < self._low_confidence_threshold:
            top_intent, top_score = "unknown", ranked[0][1] if ranked else 0.0
        else:
            top_intent, top_score = ranked[0]

        alternatives = [(intent, score) for intent, score in ranked[1:4]]
        return IntentResult(intent=top_intent, confidence=round(top_score, 4), alternatives=alternatives)
