"""
Text Normalizer.

Runs BEFORE intent recognition and entity extraction. Its job is purely
textual cleanup:
  1. Expand common business abbreviations / shorthand ("inv" -> "invoice",
     "pls" -> "please", "asap" -> "as soon as possible").
  2. Correct common misspellings via a lightweight edit-distance corrector
     against a domain vocabulary (no heavy ML dependency required).
  3. Normalize whitespace/punctuation/casing artifacts from speech-to-text
     or fat-fingered typing.

Design note: this is intentionally dictionary + edit-distance based rather
than a transformer spell-checker, to keep the engine lightweight and fast.
The ITextNormalizer interface means a more powerful implementation (e.g. a
neural spell-corrector) can be swapped in later with zero changes elsewhere.
"""
from __future__ import annotations

import re

from app.domain.interfaces import ITextNormalizer

# Common shorthand / abbreviations seen in business chat & speech-to-text output.
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    "inv": "invoice",
    "invs": "invoices",
    "pymt": "payment",
    "pymts": "payments",
    "amt": "amount",
    "amts": "amounts",
    "acct": "account",
    "accts": "accounts",
    "bal": "balance",
    "ref": "reference",
    "refs": "references",
    "qty": "quantity",
    "pls": "please",
    "plz": "please",
    "asap": "as soon as possible",
    "u": "you",
    "ur": "your",
    "thx": "thanks",
    "tmrw": "tomorrow",
    "yday": "yesterday",
    "info": "information",
    "cust": "customer",
    "custs": "customers",
    "usd": "USD",
    "eur": "EUR",
    "gbp": "GBP",
    "ngn": "NGN",
    "w/": "with",
    "w/o": "without",
    "b/w": "between",
    "no.": "number",
    "num": "number",
}

# Small domain vocabulary used for edit-distance spell correction. In a real
# deployment this would be loaded from the product's own terminology.
DOMAIN_VOCABULARY: list[str] = [
    "invoice", "invoices", "payment", "payments", "amount", "amounts",
    "account", "accounts", "balance", "reference", "customer", "customers",
    "refund", "receipt", "transaction", "transactions", "currency",
    "product", "products", "order", "orders", "quantity", "cancel",
    "confirm", "schedule", "reminder", "overdue", "pending", "paid",
    "unpaid", "send", "create", "update", "delete", "show", "list",
]


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance, O(len(a)*len(b)); fine for short tokens."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


class RuleBasedTextNormalizer(ITextNormalizer):
    def __init__(
        self,
        abbreviations: dict[str, str] | None = None,
        vocabulary: list[str] | None = None,
        max_edit_distance: int = 2,
    ):
        self._abbreviations = {**DEFAULT_ABBREVIATIONS, **(abbreviations or {})}
        self._vocabulary = vocabulary if vocabulary is not None else DOMAIN_VOCABULARY
        self._max_edit_distance = max_edit_distance

    def normalize(self, text: str) -> str:
        text = self._clean_whitespace(text)
        tokens = text.split(" ")
        normalized_tokens = [self._normalize_token(tok) for tok in tokens]
        result = " ".join(t for t in normalized_tokens if t != "")
        return self._clean_whitespace(result)

    # -- internal helpers ---------------------------------------------

    def _clean_whitespace(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _normalize_token(self, token: str) -> str:
        if token == "":
            return token
        # Preserve trailing punctuation (e.g. "asap!" -> "as soon as possible!")
        match = re.match(r"^([\w/\.\-]+)([!?.,]*)$", token)
        core, trailing_punct = (match.group(1), match.group(2)) if match else (token, "")

        lowered = core.lower()

        # 1. Abbreviation expansion (exact match on lowercase core).
        if lowered in self._abbreviations:
            expanded = self._abbreviations[lowered]
            return f"{expanded}{trailing_punct}"

        # 2. Spell correction against domain vocabulary, only for
        #    alphabetic tokens of reasonable length (avoid mangling
        #    numbers, invoice codes like INV-2024-001, emails, etc).
        if lowered.isalpha() and len(lowered) > 3:
            corrected = self._spell_correct(lowered)
            if corrected is not None:
                return f"{corrected}{trailing_punct}"

        return f"{core}{trailing_punct}"

    def _spell_correct(self, word: str) -> str | None:
        if word in self._vocabulary:
            return None  # already correct, leave original casing untouched
        best_match, best_distance = None, self._max_edit_distance + 1
        for vocab_word in self._vocabulary:
            # Cheap length-based pruning before computing full edit distance.
            if abs(len(vocab_word) - len(word)) > self._max_edit_distance:
                continue
            dist = _levenshtein(word, vocab_word)
            if dist < best_distance:
                best_distance, best_match = dist, vocab_word
        if best_match is not None and best_distance <= self._max_edit_distance:
            return best_match
        return None
