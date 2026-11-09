import asyncio
import json
import random
import time

import httpx
import pytest

from limiter import TokenRateLimiter
from providers import HttpProvider
from router import (GatewayError, ModelRouter, ProviderConnectionError, ProviderHTTPError, ProviderTimeout,
                    classify, Failure)

pytestmark = pytest.mark.anyio

REQ = {"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 100}
SECRETS = ["postgres://admin:hunter2@10.0.0.5", "Traceback (most recent call last)", "/srv/app/secret.py",
           "primary-provider-internal", "backup-provider-internal", "sk-live-UPSTREAM-KEY"]


FIXED_MESSAGES = {
    "upstream_overloaded": "The model service is overloaded. Retry later.",
    "upstream_timeout": "The model provider timed out.",
    "upstream_unavailable": "The model service is temporarily unavailable.",
    "request_rejected": "The model provider rejected the request.",
    "internal_error": "An internal error occurred.",
}


class Fake:
    def __init__(self, name, fn):
        self.name, self.fn, self.calls, self.cancelled = name, fn, 0, 0

    async def complete(self, req):
        self.calls += 1
        try:
            return await self.fn(req)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


def ok(total=100, text="hi"):
    async def f(req):
        return {"choices": [{"message": {"role": "assistant", "content": text}}], "usage": {"total_tokens": total}}
    return f


def raises(exc):
    async def f(req):
        raise exc
    return f


def sleeps(sec, then=None):
    async def f(req):
        await asyncio.sleep(sec)
        return await (then or ok())(req)
    return f


@pytest.fixture
def lim(tmp_path):
    return TokenRateLimiter(str(tmp_path / "r.db"), limit=50_000)


def mk(lim, primary, secondary, **kw):
    kw.setdefault("primary_timeout", 0.1)
    kw.setdefault("secondary_timeout", 1.0)
    return ModelRouter(lim, primary, secondary, **kw)


async def fail(router, payload=REQ, key="k"):
    with pytest.raises(GatewayError) as ei:
        await router.complete(key, payload)
    return ei.value


# ---------------------------------------------------------------- happy path / billing
async def test_primary_success_does_not_touch_secondary_and_bills_actual_usage(lim):
    p, s = Fake("p", ok(123, "from-primary")), Fake("s", ok())
    out = await mk(lim, p, s).complete("k", REQ)
    assert out["choices"][0]["message"]["content"] == "from-primary"
    assert out["gateway"]["fallback_used"] is False and out["gateway"]["request_id"].startswith("req_")
    assert (p.calls, s.calls) == (1, 0)
    assert await lim.usage("k") == 123                  # estimate was ~1000+, billed what was actually used


async def test_missing_usage_falls_back_to_the_estimate(lim):
    async def no_usage(req):
        return {"choices": []}
    r = mk(lim, Fake("p", no_usage), Fake("s", ok()), estimator=lambda _: 777)
    await r.complete("k", REQ)
    assert await lim.usage("k") == 777


# -------------------------------------------------------------------------- failover
async def test_429_fails_over_and_bills_once(lim):
    p, s = Fake("p", raises(ProviderHTTPError(429, body="slow down"))), Fake("s", ok(50, "from-secondary"))
    out = await mk(lim, p, s).complete("k", REQ)
    assert out["choices"][0]["message"]["content"] == "from-secondary" and out["gateway"]["fallback_used"] is True
    assert (p.calls, s.calls) == (1, 1) and await lim.usage("k") == 50


async def test_timeout_fails_over_and_cancels_the_hung_primary(lim):
    p, s = Fake("p", sleeps(30)), Fake("s", ok(60, "backup"))
    t0 = time.monotonic()
    out = await mk(lim, p, s, primary_timeout=0.1).complete("k", REQ)
    assert out["gateway"]["fallback_used"] and time.monotonic() - t0 < 1.0
    await asyncio.sleep(0.05)
    assert p.cancelled == 1                             # we stopped paying for the hung call
    assert await lim.usage("k") == 60


async def test_spec_value_3000ms_default_timeout(lim):
    p, s = Fake("p", sleeps(60)), Fake("s", ok())
    r = ModelRouter(lim, p, s)                          # defaults
    assert r.primary_timeout == 3.0
    t0 = time.monotonic()
    out = await r.complete("k", REQ)
    elapsed = time.monotonic() - t0
    assert out["gateway"]["fallback_used"] and 3.0 <= elapsed < 3.6, elapsed


async def test_late_primary_result_after_timeout_is_discarded_and_not_billed(lim):
    """Provider swallows the cancellation and 'succeeds' late with a huge usage number."""
    async def stubborn(req):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            return {"choices": [{"message": {"content": "LATE"}}], "usage": {"total_tokens": 9_999}}
    p, s = Fake("p", stubborn), Fake("s", ok(40, "on-time"))
    out = await mk(lim, p, s).complete("k", REQ)
    await asyncio.sleep(0.1)
    assert out["choices"][0]["message"]["content"] == "on-time"
    assert await lim.usage("k") == 40                   # NOT 9,999


@pytest.mark.parametrize("exc", [ProviderHTTPError(500, body="x"), ProviderHTTPError(503), ProviderConnectionError("refused"),
                                 ConnectionResetError("rst"), RuntimeError("adapter bug"), ProviderHTTPError(401, body="bad key"),
                                 ProviderHTTPError(404, body="model not found"), ProviderHTTPError(408), ProviderTimeout("t")])
async def test_other_upstream_failures_also_fail_over(lim, exc):
    p, s = Fake("p", raises(exc)), Fake("s", ok())
    out = await mk(lim, p, s).complete("k", REQ)
    assert out["gateway"]["fallback_used"] and s.calls == 1


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_tenant_errors_are_not_failed_over_and_not_billed(lim, status):
    p, s = Fake("p", raises(ProviderHTTPError(status, body="context too long"))), Fake("s", ok())
    e = await fail(mk(lim, p, s))
    assert (e.status, e.code) == (400, "request_rejected") and s.calls == 0
    assert e.message == FIXED_MESSAGES["request_rejected"]
    assert await lim.usage("k") == 0


async def test_race_at_the_deadline_never_produces_two_outcomes(lim):
    """Primary finishes within +-20ms of the timeout. Invariant: exactly one winner, billed exactly once."""
    rng = random.Random(7)

    async def one(i):
        d = rng.uniform(0.03, 0.07)
        p, s = Fake("p", sleeps(d, ok(11, "P"))), Fake("s", ok(22, "S"))
        out = await mk(lim, p, s, primary_timeout=0.05).complete(f"key-{i}", REQ)
        used_fallback = out["gateway"]["fallback_used"]
        assert p.calls == 1 and s.calls == (1 if used_fallback else 0)
        assert out["choices"][0]["message"]["content"] == ("S" if used_fallback else "P")
        assert await lim.usage(f"key-{i}") == (22 if used_fallback else 11)
        return used_fallback

    outcomes = await asyncio.gather(*[one(i) for i in range(60)])
    assert any(outcomes) and not all(outcomes)          # the sweep really straddled the deadline


# ----------------------------------------------------- both providers failing
@pytest.mark.parametrize("pe,se,status,code", [
    (ProviderHTTPError(429), ProviderHTTPError(429), 503, "upstream_overloaded"),
    (ProviderTimeout("a"), ProviderTimeout("b"), 504, "upstream_timeout"),
    (ProviderHTTPError(500), ProviderHTTPError(429), 502, "upstream_unavailable"),
    (ProviderConnectionError("x"), ProviderHTTPError(503), 502, "upstream_unavailable"),
    (ProviderHTTPError(429), ProviderTimeout("t"), 503, "upstream_overloaded"),
])
async def test_total_failure_maps_to_standard_error_and_releases_tokens(lim, pe, se, status, code):
    e = await fail(mk(lim, Fake("p", raises(pe)), Fake("s", raises(se))))
    assert (e.status, e.code) == (status, code) and e.request_id.startswith("req_")
    assert e.message == FIXED_MESSAGES[code]            # fixed literal: no interpolation of internal state
    assert await lim.usage("k") == 0                    # tenant not charged for a failed request


async def test_error_payload_is_standardised_and_leaks_nothing(lim):
    boom1 = ProviderHTTPError(500, body=f"{SECRETS[1]} File '{SECRETS[2]}' db={SECRETS[0]} {SECRETS[3]}")
    boom2 = RuntimeError(f"{SECRETS[4]} key={SECRETS[5]}")
    e = await fail(mk(lim, Fake("primary-provider-internal", raises(boom1)), Fake("backup-provider-internal", raises(boom2))))
    payload = e.to_payload()
    text = json.dumps(payload)
    for secret in SECRETS:
        assert secret not in text, secret
    assert set(payload) == {"error"} and set(payload["error"]) == {"type", "code", "message", "request_id"}
    assert payload["error"]["type"] == "gateway_error"
    assert e.__cause__ is None and "Traceback" not in str(e)


async def test_unexpected_internal_failure_becomes_generic_500(lim):
    async def broken(*a, **k):
        raise RuntimeError("sqlite exploded: /var/lib/secret.db")
    lim.reserve = broken
    e = await fail(mk(lim, Fake("p", ok()), Fake("s", ok())))
    assert (e.status, e.code) == (500, "internal_error") and e.message == FIXED_MESSAGES["internal_error"]
    assert "secret.db" not in json.dumps(e.to_payload()) and "sqlite" not in json.dumps(e.to_payload())


# ------------------------------------------------------------- rate limiting via router
async def test_rate_limited_request_is_429_with_retry_after_and_never_reaches_a_provider(lim):
    p, s = Fake("p", ok(10)), Fake("s", ok())
    r = mk(lim, p, s, estimator=lambda _: 50_000)
    await r.complete("k", REQ)                          # settles to 10 tokens...
    # ...so reserve again with a pending (unsettled) 50k reservation to model concurrent in-flight work:
    await lim.reserve("k2", 50_000)
    e = await fail(r, key="k2")
    assert (e.status, e.code) == (429, "rate_limit_exceeded") and 0 < e.retry_after <= 60
    assert p.calls == 1                                 # only the first request ever hit the provider


async def test_request_larger_than_whole_budget_is_413(lim):
    e = await fail(mk(lim, Fake("p", ok()), Fake("s", ok()), estimator=lambda _: 50_001))
    assert (e.status, e.code) == (413, "request_too_large")


async def test_inflight_requests_hold_their_full_estimate(lim):
    """60 concurrent requests, 1000-token estimate each, 50k budget -> exactly 50 admitted."""
    p, s = Fake("p", sleeps(0.3, ok(100))), Fake("s", ok())
    r = mk(lim, p, s, estimator=lambda _: 1_000, primary_timeout=2.0)
    results = await asyncio.gather(*[r.complete("k", REQ) for _ in range(60)], return_exceptions=True)
    good = [x for x in results if isinstance(x, dict)]
    limited = [x for x in results if isinstance(x, GatewayError) and x.code == "rate_limit_exceeded"]
    assert (len(good), len(limited)) == (50, 10)
    assert p.calls == 50 and await lim.usage("k") == 50 * 100     # settled to actual usage


# ------------------------------------------------------------------------ validation
@pytest.mark.parametrize("payload", [
    None, [], "x", {}, {"messages": []}, {"messages": "hi"}, {"messages": [None]}, {"messages": [{"role": "user"}]},
    {"messages": [{"role": "user", "content": 5}]}, {"messages": [{"role": "user", "content": "a"}], "max_tokens": 0},
    {"messages": [{"role": "user", "content": "a"}], "max_tokens": -5}, {"messages": [{"role": "user", "content": "a"}], "max_tokens": "9"},
    {"messages": [{"role": "user", "content": "a"}], "max_tokens": True}, {"messages": [{"role": "user", "content": "a"}], "max_tokens": 10**9},
    {"messages": [{"role": "user", "content": "a"}], "model": 5},
])
async def test_invalid_payloads_are_400_before_any_work(lim, payload):
    p = Fake("p", ok())
    e = await fail(mk(lim, p, Fake("s", ok())), payload=payload)
    assert (e.status, e.code) == (400, "invalid_request") and p.calls == 0 and await lim.usage("k") == 0


@pytest.mark.parametrize("key", [None, "", "   ", 123])
async def test_missing_api_key_is_401(lim, key):
    e = await fail(mk(lim, Fake("p", ok()), Fake("s", ok())), key=key)
    assert (e.status, e.code) == (401, "invalid_api_key")


# ----------------------------------------------------------------------- cancellation
async def test_cancelled_request_releases_reservation_and_cancels_provider(lim):
    p = Fake("p", sleeps(30))
    r = mk(lim, p, Fake("s", ok()), primary_timeout=10.0, estimator=lambda _: 5_000)
    task = asyncio.ensure_future(r.complete("k", REQ))
    await asyncio.sleep(0.1)
    assert await lim.usage("k") == 5_000                # reserved while in flight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)
    assert p.cancelled == 1 and await lim.usage("k") == 0


# ----------------------------------------------------------------------- classification
def test_classify_table():
    assert classify(ProviderHTTPError(429)) is Failure.RATE_LIMITED
    assert classify(asyncio.TimeoutError()) is Failure.TIMEOUT
    assert classify(ProviderHTTPError(502)) is Failure.SERVER_ERROR
    assert classify(ProviderHTTPError(422)) is Failure.CLIENT_ERROR
    assert classify(ValueError("x")) is Failure.UNKNOWN


# ------------------------------------------------------------------- HttpProvider
def http_provider(handler):
    return HttpProvider("http-p", "http://upstream.invalid/v1/chat/completions", "sk-up",
                        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_http_provider_success_and_auth_header():
    seen = {}

    def h(request):
        seen["auth"], seen["body"] = request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"usage": {"total_tokens": 5}, "choices": []})
    p = http_provider(h)
    from router import parse_request
    assert (await p.complete(parse_request(REQ)))["usage"]["total_tokens"] == 5
    assert seen["auth"] == "Bearer sk-up" and seen["body"]["stream"] is False


@pytest.mark.parametrize("handler,exc", [
    (lambda r: httpx.Response(429, headers={"retry-after": "7"}, text="slow"), ProviderHTTPError),
    (lambda r: httpx.Response(500, text="Traceback..."), ProviderHTTPError),
    (lambda r: httpx.Response(200, text="<html>not json</html>"), ProviderHTTPError),
    (lambda r: httpx.Response(200, json=[1, 2]), ProviderHTTPError),
])
async def test_http_provider_maps_errors(handler, exc):
    from router import parse_request
    with pytest.raises(exc):
        await http_provider(handler).complete(parse_request(REQ))


async def test_http_provider_maps_transport_errors():
    from router import parse_request

    def timeout(r):
        raise httpx.ReadTimeout("t")

    def refused(r):
        raise httpx.ConnectError("refused")
    with pytest.raises(ProviderTimeout):
        await http_provider(timeout).complete(parse_request(REQ))
    with pytest.raises(ProviderConnectionError):
        await http_provider(refused).complete(parse_request(REQ))


async def test_end_to_end_with_http_providers_failing_over_on_real_429(lim):
    calls = {"p": 0, "s": 0}

    def primary(r):
        calls["p"] += 1
        return httpx.Response(429, text="rate limited by vendor: org-secret-123")

    def secondary(r):
        calls["s"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 9}})
    out = await mk(lim, http_provider(primary), http_provider(secondary)).complete("k", REQ)
    assert out["gateway"]["fallback_used"] and calls == {"p": 1, "s": 1} and await lim.usage("k") == 9
