import pytest
from fastapi.testclient import TestClient

from app.core.security import issue_token
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def auth_header(permissions=None, roles=None):
    token = issue_token(
        "user-1", "tenant-a",
        permissions or ["invoices:read", "payments:write"],
        roles or [],
    )
    return {"Authorization": f"Bearer {token}"}


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_interpret_requires_auth(client):
    resp = client.post("/v1/interpret", json={"conversation_id": "c1", "text": "hello"})
    assert resp.status_code == 401


def test_interpret_rejects_invalid_token(client):
    resp = client.post(
        "/v1/interpret",
        json={"conversation_id": "c1", "text": "hello"},
        headers={"Authorization": "Bearer garbage"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["error_code"] == "authentication_failed"


def test_interpret_happy_path(client):
    resp = client.post(
        "/v1/interpret",
        json={"conversation_id": "c-api-1", "text": "what is the status of invoice 12345"},
        headers=auth_header(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_name"] == "get_invoice_status"
    assert body["status"] == "success"
    assert body["data"]["invoice_number"] == "12345"


def test_interpret_validation_error_on_empty_text(client):
    resp = client.post(
        "/v1/interpret",
        json={"conversation_id": "c1", "text": ""},
        headers=auth_header(),
    )
    assert resp.status_code == 422


def test_interpret_write_tool_needs_confirmation_then_executes(client):
    payload = {"conversation_id": "c-api-2", "text": "pay $500 usd for invoice 12345"}
    first = client.post("/v1/interpret", json=payload, headers=auth_header())
    assert first.json()["status"] == "pending_confirmation"

    payload["confirmed"] = True
    second = client.post("/v1/interpret", json=payload, headers=auth_header())
    assert second.json()["status"] == "success"


def test_list_tools_endpoint(client):
    resp = client.get("/v1/tools", headers=auth_header())
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    assert "get_invoice_status" in names
    assert "create_payment" in names


def test_rate_limiting_returns_429_after_threshold(client, monkeypatch):
    from app.core.rate_limiter import SlidingWindowRateLimiter
    app.state.container.rate_limiter = SlidingWindowRateLimiter(max_requests_per_minute=2)

    headers = auth_header()
    payload = {"conversation_id": "c-rate-1", "text": "check invoice 12345"}
    r1 = client.post("/v1/interpret", json=payload, headers=headers)
    r2 = client.post("/v1/interpret", json=payload, headers=headers)
    r3 = client.post("/v1/interpret", json=payload, headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def test_response_has_request_id_header(client):
    resp = client.get("/health")
    assert "X-Request-ID" in resp.headers
