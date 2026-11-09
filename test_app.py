import json

import httpx
import pytest

from app import create_app
from limiter import TokenRateLimiter
from router import ModelRouter, ProviderHTTPError

pytestmark = pytest.mark.anyio
REQ = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50}


class P:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    async def complete(self, req):
        return await self.fn()


async def good():
    return {"choices": [], "usage": {"total_tokens": 5}}


async def bad():
    raise ProviderHTTPError(500, body="Traceback secret")


def client(tmp_path, primary=good, secondary=good, limit=50_000):
    lim = TokenRateLimiter(str(tmp_path / "a.db"), limit=limit)
    app = create_app(ModelRouter(lim, P("p", primary), P("s", secondary), primary_timeout=0.2))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


def auth(k="tenant-1"):
    return {"Authorization": f"Bearer {k}"}


async def test_success_has_request_id_header(tmp_path):
    r = await client(tmp_path).post("/v1/completions", json=REQ, headers=auth())
    assert r.status_code == 200 and r.headers["x-request-id"] == r.json()["gateway"]["request_id"]


async def test_failover_is_invisible_except_for_the_flag(tmp_path):
    r = await client(tmp_path, primary=bad).post("/v1/completions", json=REQ, headers=auth())
    assert r.status_code == 200 and r.json()["gateway"]["fallback_used"] is True


async def test_request_larger_than_the_whole_budget_is_413(tmp_path):
    r = await client(tmp_path, limit=2_000).post("/v1/completions", json={**REQ, "max_tokens": 2_000}, headers=auth())
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"


async def test_429_shape_and_retry_after(tmp_path):
    lim = TokenRateLimiter(str(tmp_path / "b.db"), limit=1_000)
    await lim.reserve("tenant-1", 1_000)                      # budget fully held by in-flight work
    app = create_app(ModelRouter(lim, P("p", good), P("s", good)))
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    r = await c.post("/v1/completions", json={**REQ, "max_tokens": 10}, headers=auth())
    assert r.status_code == 429
    assert 1 <= int(r.headers["retry-after"]) <= 60
    e = r.json()["error"]
    assert e["code"] == "rate_limit_exceeded" and e["type"] == "gateway_error" and e["retry_after_seconds"] > 0


@pytest.mark.parametrize("headers,body,status,code", [
    ({}, REQ, 401, "invalid_api_key"),
    ({"Authorization": "Basic abc"}, REQ, 401, "invalid_api_key"),
    ({"Authorization": "Bearer "}, REQ, 401, "invalid_api_key"),
    (auth(), b"{not json", 400, "invalid_request"),
    (auth(), {"messages": []}, 400, "invalid_request"),
])
async def test_all_errors_use_the_same_envelope(tmp_path, headers, body, status, code):
    c = client(tmp_path)
    kw = {"content": body} if isinstance(body, bytes) else {"json": body}
    r = await c.post("/v1/completions", headers=headers, **kw)
    assert r.status_code == status
    err = r.json()["error"]
    assert err["code"] == code and err["type"] == "gateway_error"
    assert err["request_id"].startswith("req_")                     # EVERY error is correlatable
    assert r.headers["x-request-id"] == err["request_id"]
    assert set(err) <= {"type", "code", "message", "request_id", "retry_after_seconds"}


async def test_total_outage_returns_sanitised_502(tmp_path):
    r = await client(tmp_path, primary=bad, secondary=bad).post("/v1/completions", json=REQ, headers=auth())
    assert r.status_code == 502 and "Traceback" not in r.text and "secret" not in r.text


async def test_oversized_body_413(tmp_path):
    r = await client(tmp_path).post("/v1/completions", content=b"x" * 1_100_000, headers=auth())
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"
