"""
Entity Extraction layer.

Pulls structured values out of normalized text: person names, absolute and
relative dates, monetary amounts, currencies, invoice numbers, payment
references, and product mentions.

Implementation choice: regex + dateutil rather than a full NER model. This
keeps the engine lightweight and deterministic for the highly-structured
entity types this domain cares about (invoice numbers, currency amounts),
which regex handles very reliably. Free-text entities like person names use
a conservative capitalized-word heuristic. Swapping in spaCy/NER later only
requires a new IEntityExtractor implementation.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from dateutil import parser as dateutil_parser

from app.domain.entities import EntityType, ExtractedEntity
from app.domain.interfaces import IEntityExtractor

CURRENCY_SYMBOLS = {
    "$": "USD", "€": "EUR", "£": "GBP", "₦": "NGN", "¥": "JPY",
}
CURRENCY_CODES = {"usd", "eur", "gbp", "ngn", "jpy", "cad", "aud", "zar", "kes", "ghs"}

RELATIVE_DATE_PATTERNS: list[tuple[str, callable]] = [
    (r"\btoday\b", lambda now: now),
    (r"\btomorrow\b", lambda now: now + timedelta(days=1)),
    (r"\byesterday\b", lambda now: now - timedelta(days=1)),
    (r"\bnext week\b", lambda now: now + timedelta(weeks=1)),
    (r"\blast week\b", lambda now: now - timedelta(weeks=1)),
    (r"\bnext month\b", lambda now: _add_months(now, 1)),
    (r"\blast month\b", lambda now: _add_months(now, -1)),
    (r"\bin (\d+) days?\b", None),   # handled specially (needs captured group)
    (r"\b(\d+) days? ago\b", None),  # handled specially
]

INVOICE_NUMBER_RE = re.compile(r"\b(?:inv(?:oice)?[-\s#]*)([A-Za-z]?-?\d{2,}(?:-\d+)*)\b", re.IGNORECASE)
PAYMENT_REF_RE = re.compile(r"\b(?:ref(?:erence)?|txn|transaction)[-\s#:]?([A-Za-z0-9]{4,})\b", re.IGNORECASE)
AMOUNT_RE = re.compile(
    r"(?P<symbol>[$€£₦¥])?\s?(?P<amount>\d{1,3}(?:[,\d]{0,})(?:\.\d+)?)\s?(?P<code>usd|eur|gbp|ngn|jpy|cad|aud|zar|kes|ghs)?",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CAPITALIZED_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b")

# Words that are capitalized but are NOT names (avoid false positives at
# sentence starts, days, months, etc).
NAME_STOPWORDS = {
    "I", "The", "Please", "Show", "What", "Pay", "Send", "Check", "Cancel",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Invoice", "USD", "EUR", "GBP", "NGN",
}


def _add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)
    return dt.replace(year=year, month=month, day=day)


class RegexEntityExtractor(IEntityExtractor):
    def __init__(self, reference_time: datetime | None = None):
        # Injected for testability; defaults to real "now".
        self._reference_time = reference_time

    def extract(self, normalized_text: str) -> list[ExtractedEntity]:
        now = self._reference_time or datetime.now(timezone.utc)
        entities: list[ExtractedEntity] = []

        entities.extend(self._extract_invoice_numbers(normalized_text))
        entities.extend(self._extract_payment_references(normalized_text))
        entities.extend(self._extract_relative_dates(normalized_text, now))
        entities.extend(self._extract_absolute_dates(normalized_text))
        entities.extend(self._extract_amounts_and_currencies(normalized_text))
        entities.extend(self._extract_emails(normalized_text))
        entities.extend(self._extract_names(normalized_text))

        return entities

    # -- extractors -----------------------------------------------------

    def _extract_invoice_numbers(self, text: str) -> list[ExtractedEntity]:
        out = []
        for m in INVOICE_NUMBER_RE.finditer(text):
            out.append(ExtractedEntity(
                type=EntityType.INVOICE_NUMBER,
                value=m.group(1).upper(),
                raw_text=m.group(0),
                confidence=0.9,
                start=m.start(), end=m.end(),
            ))
        return out

    def _extract_payment_references(self, text: str) -> list[ExtractedEntity]:
        out = []
        for m in PAYMENT_REF_RE.finditer(text):
            out.append(ExtractedEntity(
                type=EntityType.PAYMENT_REFERENCE,
                value=m.group(1).upper(),
                raw_text=m.group(0),
                confidence=0.85,
                start=m.start(), end=m.end(),
            ))
        return out

    def _extract_relative_dates(self, text: str, now: datetime) -> list[ExtractedEntity]:
        out = []
        lowered = text.lower()

        for pattern, resolver in RELATIVE_DATE_PATTERNS:
            if resolver is None:
                continue
            m = re.search(pattern, lowered)
            if m:
                out.append(ExtractedEntity(
                    type=EntityType.RELATIVE_DATE,
                    value=resolver(now),
                    raw_text=m.group(0),
                    confidence=0.9,
                    start=m.start(), end=m.end(),
                ))

        m = re.search(r"\bin (\d+) days?\b", lowered)
        if m:
            days = int(m.group(1))
            out.append(ExtractedEntity(
                type=EntityType.RELATIVE_DATE,
                value=now + timedelta(days=days),
                raw_text=m.group(0),
                confidence=0.9,
                start=m.start(), end=m.end(),
            ))

        m = re.search(r"\b(\d+) days? ago\b", lowered)
        if m:
            days = int(m.group(1))
            out.append(ExtractedEntity(
                type=EntityType.RELATIVE_DATE,
                value=now - timedelta(days=days),
                raw_text=m.group(0),
                confidence=0.9,
                start=m.start(), end=m.end(),
            ))
        return out

    def _extract_absolute_dates(self, text: str) -> list[ExtractedEntity]:
        out = []
        # Look for date-like substrings: "12 March 2024", "2024-03-12", "03/12/2024"
        candidates = re.findall(
            r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
            r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{2,4}|"
            r"[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4})\b",
            text,
        )
        for cand in candidates:
            try:
                parsed = dateutil_parser.parse(cand, fuzzy=False)
            except (ValueError, OverflowError):
                continue
            start = text.find(cand)
            out.append(ExtractedEntity(
                type=EntityType.DATE,
                value=parsed,
                raw_text=cand,
                confidence=0.85,
                start=start, end=start + len(cand),
            ))
        return out

    def _extract_amounts_and_currencies(self, text: str) -> list[ExtractedEntity]:
        out = []
        for m in AMOUNT_RE.finditer(text):
            amount_str = m.group("amount")
            if not amount_str:
                continue
            symbol, code = m.group("symbol"), m.group("code")
            if symbol is None and code is None:
                # A bare number with no currency marker is too ambiguous to
                # call a monetary "amount" (could be an invoice/date/etc,
                # which are handled by their own extractors elsewhere).
                continue
            try:
                value = Decimal(amount_str.replace(",", ""))
            except InvalidOperation:
                continue
            currency = CURRENCY_SYMBOLS.get(symbol) if symbol else (code.upper() if code else None)
            out.append(ExtractedEntity(
                type=EntityType.AMOUNT,
                value=value,
                raw_text=m.group(0).strip(),
                confidence=0.9,
                start=m.start(), end=m.end(),
            ))
            if currency:
                out.append(ExtractedEntity(
                    type=EntityType.CURRENCY,
                    value=currency,
                    raw_text=symbol or code,
                    confidence=0.9,
                    start=m.start(), end=m.end(),
                ))
        return out

    def _extract_emails(self, text: str) -> list[ExtractedEntity]:
        out = []
        for m in EMAIL_RE.finditer(text):
            out.append(ExtractedEntity(
                type=EntityType.EMAIL, value=m.group(0), raw_text=m.group(0),
                confidence=0.95, start=m.start(), end=m.end(),
            ))
        return out

    def _extract_names(self, text: str) -> list[ExtractedEntity]:
        out = []
        for m in CAPITALIZED_NAME_RE.finditer(text):
            candidate = m.group(1)
            first_word = candidate.split()[0]
            if first_word in NAME_STOPWORDS:
                continue
            out.append(ExtractedEntity(
                type=EntityType.PERSON_NAME,
                value=candidate,
                raw_text=candidate,
                confidence=0.6,   # heuristic-based, deliberately modest
                start=m.start(), end=m.end(),
            ))
        return out
