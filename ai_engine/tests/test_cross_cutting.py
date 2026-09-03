import pytest

from app.core.exceptions import AuthenticationError
from app.core.rate_limiter import SlidingWindowRateLimiter
from app.core.security import authenticate, issue_token
from app.services.confidence_scorer import WeightedConfidenceScorer
from app.services.speech_to_text import MockSpeechToText


# -- confidence scorer -------------------------------------------------

def test_low_confidence_leads_to_clarify():
    scorer = WeightedConfidenceScorer()
    decision = scorer.score(intent_confidence=0.1, tool_confidence=0.1, tool_name="get_invoice_status")
    assert decision.action == "clarify"


def test_high_confidence_read_tool_executes():
    scorer = WeightedConfidenceScorer()
    decision = scorer.score(intent_confidence=0.9, tool_confidence=0.9, tool_name="get_invoice_status")
    assert decision.action == "execute"


def test_medium_confidence_write_tool_requires_confirmation():
    scorer = WeightedConfidenceScorer()
    decision = scorer.score(intent_confidence=0.6, tool_confidence=0.6, tool_name="create_payment")
    assert decision.action == "confirm"


def test_high_confidence_write_tool_executes():
    scorer = WeightedConfidenceScorer()
    decision = scorer.score(intent_confidence=0.95, tool_confidence=0.95, tool_name="create_payment")
    assert decision.action == "execute"


# -- rate limiter --------------------------------------------------------

def test_rate_limiter_allows_up_to_limit():
    limiter = SlidingWindowRateLimiter(max_requests_per_minute=3)
    results = [limiter.allow("user-1") for _ in range(3)]
    assert all(results)


def test_rate_limiter_blocks_after_limit():
    limiter = SlidingWindowRateLimiter(max_requests_per_minute=2)
    limiter.allow("user-1")
    limiter.allow("user-1")
    assert limiter.allow("user-1") is False


def test_rate_limiter_is_per_key():
    limiter = SlidingWindowRateLimiter(max_requests_per_minute=1)
    assert limiter.allow("user-1") is True
    assert limiter.allow("user-2") is True


# -- security (JWT) -------------------------------------------------------

def test_issue_and_authenticate_round_trip():
    token = issue_token("user-1", "tenant-a", ["invoices:read"], ["member"])
    principal = authenticate(token)
    assert principal.user_id == "user-1"
    assert principal.tenant_id == "tenant-a"
    assert principal.has_permission("invoices:read")
    assert not principal.has_permission("payments:write")


def test_authenticate_rejects_garbage_token():
    with pytest.raises(AuthenticationError):
        authenticate("not-a-real-jwt")


def test_authenticate_rejects_empty_token():
    with pytest.raises(AuthenticationError):
        authenticate("")


def test_admin_role_has_all_permissions():
    token = issue_token("admin-1", None, [], ["admin"])
    principal = authenticate(token)
    assert principal.has_permission("anything:whatsoever")


# -- speech to text --------------------------------------------------------

def test_mock_speech_to_text_returns_canned_transcript():
    stt = MockSpeechToText(canned_transcript="pay invoice 123")
    result = stt.transcribe(b"fake-audio-bytes", "audio/wav")
    assert result == "pay invoice 123"


def test_mock_speech_to_text_rejects_empty_audio():
    stt = MockSpeechToText(canned_transcript="hello")
    with pytest.raises(ValueError):
        stt.transcribe(b"", "audio/wav")
