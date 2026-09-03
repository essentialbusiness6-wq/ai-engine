from decimal import Decimal

from app.domain.entities import ConversationContext, EntityType, ExtractedEntity
from app.services.context_resolver import SlotBasedContextResolver


def make_context(slots=None):
    return ConversationContext(conversation_id="c1", user_id="u1", slots=slots or {})


def test_resolves_entities_from_current_turn():
    resolver = SlotBasedContextResolver()
    ctx = make_context()
    entities = [ExtractedEntity(type=EntityType.INVOICE_NUMBER, value="123", raw_text="123")]
    resolved = resolver.resolve(entities, ctx)
    assert resolved["invoice_number"] == "123"


def test_falls_back_to_prior_turn_slot_when_missing_this_turn():
    resolver = SlotBasedContextResolver()
    ctx = make_context(slots={"invoice_number": "999"})
    entities: list[ExtractedEntity] = []  # e.g. user just said "pay it"
    resolved = resolver.resolve(entities, ctx)
    assert resolved["invoice_number"] == "999"


def test_current_turn_overrides_prior_slot():
    resolver = SlotBasedContextResolver()
    ctx = make_context(slots={"invoice_number": "999"})
    entities = [ExtractedEntity(type=EntityType.INVOICE_NUMBER, value="123", raw_text="123")]
    resolved = resolver.resolve(entities, ctx)
    assert resolved["invoice_number"] == "123"


def test_highest_confidence_entity_wins_for_same_slot():
    resolver = SlotBasedContextResolver()
    ctx = make_context()
    entities = [
        ExtractedEntity(type=EntityType.INVOICE_NUMBER, value="AAA", raw_text="AAA", confidence=0.5),
        ExtractedEntity(type=EntityType.INVOICE_NUMBER, value="BBB", raw_text="BBB", confidence=0.9),
    ]
    resolved = resolver.resolve(entities, ctx)
    assert resolved["invoice_number"] == "BBB"


def test_context_slots_are_persisted_after_resolve():
    resolver = SlotBasedContextResolver()
    ctx = make_context()
    entities = [ExtractedEntity(type=EntityType.AMOUNT, value=Decimal("500"), raw_text="500")]
    resolver.resolve(entities, ctx)
    assert ctx.slots["amount"] == Decimal("500")


def test_backend_lookup_fills_missing_slot():
    def fake_lookup(slot_name, resolved_so_far):
        if slot_name == "invoice_number" and resolved_so_far.get("person_name") == "John Smith":
            return "INV-777"
        return None

    resolver = SlotBasedContextResolver(backend_lookup=fake_lookup)
    ctx = make_context()
    entities = [ExtractedEntity(type=EntityType.PERSON_NAME, value="John Smith", raw_text="John Smith")]
    resolved = resolver.resolve(entities, ctx)
    assert resolved["invoice_number"] == "INV-777"
