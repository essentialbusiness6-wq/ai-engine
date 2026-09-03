import time

from app.domain.entities import Message, Role
from app.services.conversation_manager import InMemoryConversationStore


def test_get_or_create_creates_new_conversation():
    store = InMemoryConversationStore()
    ctx = store.get_or_create("c1", "user-1")
    assert ctx.conversation_id == "c1"
    assert ctx.user_id == "user-1"
    assert ctx.messages == []


def test_get_or_create_returns_same_conversation():
    store = InMemoryConversationStore()
    ctx1 = store.get_or_create("c1", "user-1")
    ctx1.slots["invoice_number"] = "INV-001"
    ctx2 = store.get_or_create("c1", "user-1")
    assert ctx2.slots["invoice_number"] == "INV-001"


def test_append_message_updates_history():
    store = InMemoryConversationStore()
    store.get_or_create("c1", "user-1")
    store.append_message("c1", Message(role=Role.USER, content="hello"))
    ctx = store.get_or_create("c1", "user-1")
    assert len(ctx.messages) == 1
    assert ctx.messages[0].content == "hello"


def test_append_message_raises_for_unknown_conversation():
    store = InMemoryConversationStore()
    try:
        store.append_message("does-not-exist", Message(role=Role.USER, content="hi"))
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_message_window_is_capped():
    store = InMemoryConversationStore(max_messages_per_conversation=3)
    store.get_or_create("c1", "user-1")
    for i in range(5):
        store.append_message("c1", Message(role=Role.USER, content=f"msg-{i}"))
    ctx = store.get_or_create("c1", "user-1")
    assert len(ctx.messages) == 3
    assert [m.content for m in ctx.messages] == ["msg-2", "msg-3", "msg-4"]


def test_ttl_eviction():
    store = InMemoryConversationStore(ttl_seconds=0)
    store.get_or_create("c1", "user-1")
    time.sleep(0.01)
    ctx = store.get_or_create("c1", "user-1")
    # Since TTL is 0, the old context should have been evicted and recreated (empty).
    assert ctx.messages == []
