"""
Context Resolver.

Takes the entities extracted from the CURRENT turn plus the ConversationContext
(which holds prior turns' resolved slots) and produces a single merged
"resolved slots" dict that the LLM tool-selection step and tool executor can
use. This is what lets a user say "pay it" after previously mentioning
"invoice 123" -- the current turn has no invoice entity, but the resolver
pulls it from context.slots.

It can optionally consult a backend lookup port (e.g. "if the user mentions
just a customer name, look up their most recent open invoice") -- this is
injected as a callable so the resolver has no direct knowledge of any
specific backend service (Dependency Inversion again).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from app.domain.entities import ConversationContext, EntityType, ExtractedEntity
from app.domain.interfaces import IContextResolver

# Maps entity types to the slot name they populate. Kept as a plain module
# constant since it's simple configuration, not behavior.
ENTITY_TO_SLOT: dict[EntityType, str] = {
    EntityType.INVOICE_NUMBER: "invoice_number",
    EntityType.PAYMENT_REFERENCE: "payment_reference",
    EntityType.AMOUNT: "amount",
    EntityType.CURRENCY: "currency",
    EntityType.DATE: "date",
    EntityType.RELATIVE_DATE: "date",
    EntityType.PERSON_NAME: "person_name",
    EntityType.EMAIL: "email",
    EntityType.PRODUCT: "product",
}

# Backend lookup signature: (slot_name, resolved_slots_so_far) -> value or None
BackendLookup = Callable[[str, dict[str, Any]], Optional[Any]]


class SlotBasedContextResolver(IContextResolver):
    def __init__(self, backend_lookup: Optional[BackendLookup] = None):
        self._backend_lookup = backend_lookup

    def resolve(
        self,
        entities: list[ExtractedEntity],
        context: ConversationContext,
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(context.slots)  # start from carried-over slots

        # 1. Overlay entities extracted THIS turn (most recent info wins).
        current_turn_slots = self._entities_to_slots(entities)
        resolved.update(current_turn_slots)

        # 2. For any well-known slot still missing, try a backend lookup
        #    (e.g. infer invoice_number from a person_name via a customer
        #    lookup service). This never invents business logic here --
        #    it delegates to whatever backend function was injected.
        if self._backend_lookup is not None:
            for slot_name in ("invoice_number", "amount", "currency"):
                if resolved.get(slot_name) is None:
                    looked_up = self._backend_lookup(slot_name, resolved)
                    if looked_up is not None:
                        resolved[slot_name] = looked_up

        # 3. Persist back into context so future turns can reuse it
        #    ("pay it" two turns from now should still work).
        context.slots.update(current_turn_slots)

        return resolved

    def _entities_to_slots(self, entities: list[ExtractedEntity]) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        best_confidence: dict[str, float] = {}
        for entity in entities:
            slot_name = ENTITY_TO_SLOT.get(entity.type)
            if slot_name is None:
                continue
            # If multiple entities map to the same slot, keep the
            # highest-confidence one.
            if slot_name not in slots or entity.confidence > best_confidence[slot_name]:
                slots[slot_name] = entity.value
                best_confidence[slot_name] = entity.confidence
        return slots
