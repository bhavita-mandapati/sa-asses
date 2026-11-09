"""Resilient model router: tenant rate limiting + primary->secondary failover + sanitized errors.

    request -> validate -> reserve tokens (SQLite) -> primary (3000ms hard deadline)
                                                        |  on 429 / timeout / 5xx / conn error
                                                        v
                                                     secondary
    success -> settle actual usage      any failure or cancellation -> release reservation

Design decisions worth knowing
------------------------------
* HARD deadline: the provider call runs as its own task and we `asyncio.wait(..., timeout)`.
  On expiry the task is cancelled and its result is *never* looked at - even if the provider swallows
  the cancellation and returns later - so a late primary response can't be used or double-charged.
  If the task finished before we got to look (finished-at-the-deadline race) we USE that result
  rather than throw away a completed response and pay for a second call.
* Billing: tokens are reserved up front, then settled to the winning provider's actual usage.
  Failed / cancelled / fully-failed-over requests release the reservation (tenant isn't charged).
* Failover triggers: 429 and timeout (as specified), plus 5xx / connection errors / upstream
  auth-config errors (a "resilient" gateway shouldn't 502 while a healthy backup exists).
  Tenant-caused errors (400/413/422) are NOT failed over - the backup would reject them too.
* Errors: every failure leaves this module as GatewayError with a fixed, generic message and a
  request_id. Upstream bodies, exception text, URLs and provider names go to the server log only.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from limiter import RateLimitExceeded, RequestExceedsLimit, TokenRateLimiter

log = logging.getLogger("llm-router")

DEFAULT_MAX_TOKENS = 1024
MAX_ALLOWED_MAX_TOKENS = 128_000
MAX_MESSAGES = 1000


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:16]


# ------------------------------------------------------------------------ errors
class GatewayError(Exception):
    """The ONLY error type that leaves the router. Message is always a fixed generic string."""

    def __init__(self, status: int, code: str, message: str, *, retry_after: float | None = None,
                 request_id: str | None = None) -> None:
        super().__init__(code)
        self.status, self.code, self.message = status, code, message
        self.retry_after, self.request_id = retry_after, request_id

    def to_payload(self) -> dict[str, Any]:
        err: dict[str, Any] = {"type": "gateway_error", "code": self.code, "message": self.message,
                               "request_id": self.request_id}
        if self.retry_after is not None:
            err["retry_after_seconds"] = round(self.retry_after, 3)
        return {"error": err}


class ProviderError(Exception):
    """Base for errors raised by provider adapters. str() may contain upstream details: log-only."""


class ProviderHTTPError(ProviderError):
    def __init__(self, status: int, *, body: str = "", retry_after: float | None = None) -> None:
        super().__init__(f"upstream HTTP {status}: {body[:200]}")
        self.status, self.retry_after = status, retry_after


class ProviderConnectionError(ProviderError):
    pass


class ProviderTimeout(ProviderError):
    pass


class Failure(enum.Enum):
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    CONNECTION = "connection"
    CLIENT_ERROR = "client_error"   # tenant's fault: never fail over
    UNKNOWN = "unknown"


def classify(exc: BaseException) -> Failure:
    if isinstance(exc, (ProviderTimeout, TimeoutError)):  # asyncio.TimeoutError is TimeoutError on 3.11+
        return Failure.TIMEOUT
    if isinstance(exc, ProviderHTTPError):
        if exc.status == 429:
            return Failure.RATE_LIMITED
        if exc.status == 408:
            return Failure.TIMEOUT
        if exc.status in (400, 413, 422):
            return Failure.CLIENT_ERROR
        return Failure.SERVER_ERROR  # 5xx, and 401/403/404 = OUR upstream config is broken -> fail over, log loudly
    if isinstance(exc, (ProviderConnectionError, ConnectionError, OSError)):
        return Failure.CONNECTION
    return Failure.UNKNOWN


# --------------------------------------------------------------- request model
@dataclass(frozen=True)
class CompletionRequest:
    messages: list[dict[str, str]]
    max_tokens: int | None
    model: str | None
    raw: dict[str, Any]


def parse_request(payload: Any) -> CompletionRequest:
    def bad(msg: str) -> GatewayError:
        return GatewayError(400, "invalid_request", msg)

    if not isinstance(payload, dict):
        raise bad("Request body must be a JSON object.")
    msgs = payload.get("messages")
    if not isinstance(msgs, list) or not msgs or len(msgs) > MAX_MESSAGES:
        raise bad("'messages' must be a non-empty list.")
    for m in msgs:
        if not (isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)):
            raise bad("Each message needs string 'role' and 'content'.")
    mt = payload.get("max_tokens")
    if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or not 1 <= mt <= MAX_ALLOWED_MAX_TOKENS):
        raise bad("'max_tokens' must be a positive integer.")
    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or len(model) > 200):
        raise bad("'model' must be a string.")
    return CompletionRequest(msgs, mt, model, payload)


def estimate_tokens(req: CompletionRequest) -> int:
    """Cheap, dependency-free upper-ish estimate: ~4 chars/token + per-message overhead + max_tokens."""
    prompt = 3 + sum(4 + math.ceil(len(m["content"]) / 4) for m in req.messages)
    return prompt + (req.max_tokens or DEFAULT_MAX_TOKENS)


class Provider(Protocol):
    name: str

    async def complete(self, request: CompletionRequest) -> dict[str, Any]: ...


def _drain(task: asyncio.Future) -> None:
    if not task.cancelled():
        task.exception()  # mark retrieved: abandoned provider tasks must not log "never retrieved"


# ----------------------------------------------------------------------- router
class ModelRouter:
    def __init__(self, limiter: TokenRateLimiter, primary: Provider, secondary: Provider, *,
                 primary_timeout: float = 3.0, secondary_timeout: float = 10.0,
                 estimator: Callable[[CompletionRequest], int] = estimate_tokens) -> None:
        self.limiter, self.primary, self.secondary = limiter, primary, secondary
        self.primary_timeout, self.secondary_timeout = primary_timeout, secondary_timeout
        self.estimator = estimator

    async def complete(self, api_key: str | None, payload: Any) -> dict[str, Any]:
        rid = new_request_id()
        try:
            return await self._complete(api_key, payload, rid)
        except GatewayError as e:
            e.request_id = rid
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("unexpected router failure rid=%s", rid)
            raise GatewayError(500, "internal_error", "An internal error occurred.", request_id=rid) from None

    async def _complete(self, api_key: str | None, payload: Any, rid: str) -> dict[str, Any]:
        if not isinstance(api_key, str) or not api_key.strip():
            raise GatewayError(401, "invalid_api_key", "A valid API key is required.")
        req = parse_request(payload)
        estimate = self.estimator(req)

        try:
            reservation = await self.limiter.reserve(api_key, estimate)
        except RateLimitExceeded as e:
            raise GatewayError(429, "rate_limit_exceeded", "Token rate limit exceeded. Retry later.",
                               retry_after=e.retry_after) from None
        except RequestExceedsLimit:
            raise GatewayError(413, "request_too_large", "Request exceeds the per-minute token budget.") from None

        settled = False
        try:
            resp, fallback_used = await self._dispatch(req, rid)
            await asyncio.shield(self.limiter.settle(reservation, self._actual_tokens(resp, estimate)))
            settled = True
            out = dict(resp)
            out["gateway"] = {"request_id": rid, "fallback_used": fallback_used}
            return out
        finally:
            if not settled:  # failure OR cancellation: never bill for work that didn't produce a response
                await asyncio.shield(self.limiter.release(reservation))

    @staticmethod
    def _actual_tokens(resp: dict[str, Any], estimate: int) -> int:
        usage = resp.get("usage")
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        return total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else estimate

    async def _call(self, provider: Provider, req: CompletionRequest, timeout: float) -> dict[str, Any]:
        task = asyncio.ensure_future(provider.complete(req))
        try:
            await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:  # the caller went away: stop paying for the upstream call
            task.cancel()
            task.add_done_callback(_drain)
            raise
        if task.done():  # includes "finished in the instant the deadline fired": use it, don't redo it
            if task.cancelled():
                raise ProviderError("provider task cancelled itself")
            return task.result()
        task.cancel()
        task.add_done_callback(_drain)  # whatever it does after this point is ignored
        raise ProviderTimeout(f"no response within {timeout}s")

    async def _dispatch(self, req: CompletionRequest, rid: str) -> tuple[dict[str, Any], bool]:
        failures: list[Failure] = []
        for attempt, (provider, timeout) in enumerate(((self.primary, self.primary_timeout),
                                                       (self.secondary, self.secondary_timeout))):
            try:
                result = await self._call(provider, req, timeout)
                if not isinstance(result, dict):
                    raise ProviderError("provider returned a non-object response")
                return result, attempt == 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                kind = classify(exc)
                log.warning("provider failure rid=%s provider=%s kind=%s detail=%.200r",
                            rid, getattr(provider, "name", "?"), kind.name, exc)  # server log only
                if kind is Failure.CLIENT_ERROR:
                    raise GatewayError(400, "request_rejected", "The model provider rejected the request.") from None
                failures.append(kind)
                if attempt == 0:
                    log.info("failing over rid=%s", rid)
        raise self._exhausted(failures)

    @staticmethod
    def _exhausted(failures: list[Failure]) -> GatewayError:
        if all(f is Failure.TIMEOUT for f in failures):
            return GatewayError(504, "upstream_timeout", "The model provider timed out.")
        if all(f in (Failure.RATE_LIMITED, Failure.TIMEOUT) for f in failures):
            return GatewayError(503, "upstream_overloaded", "The model service is overloaded. Retry later.", retry_after=1.0)
        return GatewayError(502, "upstream_unavailable", "The model service is temporarily unavailable.")
