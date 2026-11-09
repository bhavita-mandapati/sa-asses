"""Token-aware sliding-window rate limiter backed by on-disk SQLite.

Algorithm: sliding window *log*. Every reservation is a row (tenant, ts_ms, tokens); a tenant's
usage is the SUM of rows with ts_ms > now - window. Exact (no fixed-window burst at the edges).

Atomicity: check-and-consume runs inside `BEGIN IMMEDIATE`, which takes SQLite's write lock before
reading. So the "SUM then INSERT" cannot interleave with another coroutine, thread, OR gateway
process using the same database file (WAL + busy_timeout handle contention).

Reservation lifecycle (this is what makes it *token-aware* rather than request-counting):
    reserve(estimate)  -> row inserted up front, so concurrent requests can't all overshoot
    settle(actual)     -> row rewritten to real usage (refund if under, charge if over)
    release()          -> row deleted when the request failed / was cancelled (tenant not billed)

Window semantics: a row counts while  ts_ms > now_ms - window_ms  (window is (now-60s, now]),
so a row stops counting at exactly 60.000s old.

API keys are never stored: tenants are identified by a truncated SHA-256 of the key.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("llm-router.limiter")

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant TEXT    NOT NULL,
    ts_ms  INTEGER NOT NULL,
    tokens INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS idx_token_events_tenant_ts ON token_events (tenant, ts_ms);
"""


def tenant_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float, used: int, limit: int):
        super().__init__("rate limit exceeded")
        self.retry_after, self.used, self.limit = retry_after, used, limit


class RequestExceedsLimit(Exception):
    """A single request larger than the whole per-window budget can never succeed."""


@dataclass(frozen=True)
class Reservation:
    id: int
    tenant: str
    tokens: int


class TokenRateLimiter:
    def __init__(self, db_path: str, *, limit: int = 50_000, window_seconds: float = 60.0,
                 clock: Callable[[], float] = time.time, purge_every: int = 200,
                 busy_timeout_seconds: float = 10.0) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window must be positive")
        self.db_path, self.limit = db_path, limit
        self._window_ms = round(window_seconds * 1000)
        self._clock = clock
        self._purge_every = purge_every
        self._busy = busy_timeout_seconds
        self._local = threading.local()
        self._calls = 0
        self._calls_lock = threading.Lock()
        c = self._conn()
        c.executescript(SCHEMA)

    # ------------------------------------------------------------------ plumbing
    def _now_ms(self) -> int:
        return round(self._clock() * 1000)

    def _conn(self) -> sqlite3.Connection:
        """One connection per worker thread (sqlite3 connections are not shareable across threads)."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=self._busy, isolation_level=None)  # autocommit; we BEGIN explicitly
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute(f"PRAGMA busy_timeout={int(self._busy * 1000)}")
            self._local.conn = c
        return c

    # ------------------------------------------------------------- sync operations
    def _reserve_sync(self, key: str, tokens: int) -> Reservation:
        tenant, now_ms = tenant_id(key), self._now_ms()
        if tokens > self.limit:
            raise RequestExceedsLimit()
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            c.execute("DELETE FROM token_events WHERE tenant=? AND ts_ms<=?", (tenant, now_ms - self._window_ms))  # evict
            used = c.execute("SELECT COALESCE(SUM(tokens),0) FROM token_events WHERE tenant=?", (tenant,)).fetchone()[0]
            if used + tokens <= self.limit:
                rid = c.execute("INSERT INTO token_events(tenant, ts_ms, tokens) VALUES (?,?,?)",
                                (tenant, now_ms, tokens)).lastrowid
                c.execute("COMMIT")
                self._maybe_purge()
                return Reservation(rid, tenant, tokens)
            rows = c.execute("SELECT ts_ms, tokens FROM token_events WHERE tenant=? ORDER BY ts_ms, id", (tenant,)).fetchall()
            c.execute("COMMIT")
        except BaseException:
            if c.in_transaction:
                c.execute("ROLLBACK")
            raise
        # When will enough of the oldest rows expire to fit `tokens`?
        need, acc, retry_ms = used + tokens - self.limit, 0, self._window_ms
        for ts, tk in rows:
            acc += tk
            if acc >= need:
                retry_ms = ts + self._window_ms - now_ms
                break
        raise RateLimitExceeded(max(retry_ms, 1) / 1000.0, used, self.limit)

    def _settle_sync(self, r: Reservation, actual: int) -> None:
        # No-op if the row was already evicted (request outlived the window): it no longer counts.
        self._conn().execute("UPDATE token_events SET tokens=? WHERE id=?", (max(0, int(actual)), r.id))

    def _release_sync(self, r: Reservation) -> None:
        self._conn().execute("DELETE FROM token_events WHERE id=?", (r.id,))

    def _usage_sync(self, key: str) -> int:
        cutoff = self._now_ms() - self._window_ms
        return self._conn().execute("SELECT COALESCE(SUM(tokens),0) FROM token_events WHERE tenant=? AND ts_ms>?",
                                    (tenant_id(key), cutoff)).fetchone()[0]

    def _purge_sync(self) -> int:
        return self._conn().execute("DELETE FROM token_events WHERE ts_ms<=?", (self._now_ms() - self._window_ms,)).rowcount

    def _maybe_purge(self) -> None:
        """Opportunistic global eviction so rows of idle tenants don't accumulate forever."""
        with self._calls_lock:
            self._calls += 1
            due = self._calls % self._purge_every == 0
        if due:
            self._purge_sync()

    # ------------------------------------------------------------ async interface
    async def reserve(self, api_key: str, tokens: int) -> Reservation:
        """Atomically check the budget and reserve `tokens`. Raises RateLimitExceeded / RequestExceedsLimit.

        Cancellation safety: the DB work runs in a thread and may finish AFTER the awaiting task was
        cancelled. A bare `await to_thread()` would then leak a reservation nobody will ever release,
        so we shield the call and, if cancelled, release whatever it ends up reserving.
        """
        fut = asyncio.ensure_future(asyncio.to_thread(self._reserve_sync, api_key, tokens))
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            loop = asyncio.get_running_loop()

            def _cleanup(f: asyncio.Future) -> None:
                if f.cancelled() or f.exception() is not None:
                    return  # nothing reserved (denied / failed) - exception retrieved, no warning
                loop.run_in_executor(None, self._release_sync, f.result())

            fut.add_done_callback(_cleanup)
            raise

    async def settle(self, r: Reservation, actual_tokens: int) -> None:
        await asyncio.to_thread(self._settle_sync, r, actual_tokens)

    async def release(self, r: Reservation) -> None:
        await asyncio.to_thread(self._release_sync, r)

    async def usage(self, api_key: str) -> int:
        return await asyncio.to_thread(self._usage_sync, api_key)

    async def purge_expired(self) -> int:
        return await asyncio.to_thread(self._purge_sync)
