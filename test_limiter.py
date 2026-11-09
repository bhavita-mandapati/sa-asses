import asyncio
import multiprocessing as mp
import os
import sqlite3

import pytest

from limiter import RateLimitExceeded, RequestExceedsLimit, TokenRateLimiter, tenant_id

pytestmark = pytest.mark.anyio


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def lim(tmp_path, clock):
    return TokenRateLimiter(str(tmp_path / "rl.db"), limit=50_000, window_seconds=60, clock=clock)


async def denied(lim, key, tokens):
    with pytest.raises(RateLimitExceeded) as ei:
        await lim.reserve(key, tokens)
    return ei.value


# ------------------------------------------------------------ basic accounting
async def test_allows_up_to_limit_then_denies(lim):
    await lim.reserve("k", 30_000)
    await lim.reserve("k", 20_000)                     # exactly 50,000 -> allowed
    assert await lim.usage("k") == 50_000
    e = await denied(lim, "k", 1)
    assert e.used == 50_000 and e.limit == 50_000


async def test_single_request_bigger_than_budget_can_never_succeed(lim):
    with pytest.raises(RequestExceedsLimit):
        await lim.reserve("k", 50_001)
    assert await lim.usage("k") == 0                    # nothing written


async def test_tenants_are_isolated(lim):
    await lim.reserve("tenant-a", 50_000)
    await denied(lim, "tenant-a", 1)
    await lim.reserve("tenant-b", 50_000)               # unaffected


# ------------------------------------------------------- sliding window / eviction
async def test_window_boundary_is_exact(lim, clock):
    await lim.reserve("k", 30_000)                      # t = 0
    clock.advance(30)
    await lim.reserve("k", 20_000)                      # t = 30  (total 50k)
    clock.advance(29.999)                               # t = 59.999: first row is 59.999s old -> still counts
    e = await denied(lim, "k", 1)
    assert e.retry_after == pytest.approx(0.001, abs=1e-6)
    clock.advance(0.001)                                # t = 60.000: first row is exactly 60s old -> evicted
    assert await lim.usage("k") == 20_000
    await lim.reserve("k", 30_000)                      # fits again
    await denied(lim, "k", 1)


async def test_sliding_not_fixed_window(lim, clock):
    """Fixed windows allow 2x burst across a boundary; a sliding window must not."""
    clock.advance(59)                                   # late in "minute 0"
    await lim.reserve("k", 50_000)
    clock.advance(2)                                    # t=61: would be "minute 1" in a fixed window
    await denied(lim, "k", 50_000)                      # but 50k was spent only 2s ago


async def test_retry_after_points_at_the_row_whose_expiry_makes_room(lim, clock):
    await lim.reserve("k", 30_000)                      # t=0
    clock.advance(10)
    await lim.reserve("k", 20_000)                      # t=10
    clock.advance(10)                                   # t=20
    # need 25k: expiring the 30k row (t=0 -> frees at t=60) is enough -> 40s
    assert (await denied(lim, "k", 25_000)).retry_after == pytest.approx(40.0)
    # need 35k: 30k alone isn't enough, need the t=10 row too (frees at t=70) -> 50s
    assert (await denied(lim, "k", 35_000)).retry_after == pytest.approx(50.0)


async def test_following_retry_after_actually_succeeds(lim, clock):
    await lim.reserve("k", 50_000)
    e = await denied(lim, "k", 10_000)
    clock.advance(e.retry_after)
    await lim.reserve("k", 10_000)


async def test_eviction_deletes_rows_and_global_purge_covers_idle_tenants(tmp_path, clock):
    path = str(tmp_path / "rl.db")
    lim = TokenRateLimiter(path, limit=50_000, clock=clock)
    for i in range(20):
        await lim.reserve(f"idle-{i}", 100)
    clock.advance(61)
    assert await lim.purge_expired() == 20
    rows = sqlite3.connect(path).execute("SELECT COUNT(*) FROM token_events").fetchone()[0]
    assert rows == 0


async def test_rows_of_one_tenant_are_evicted_on_next_reserve(tmp_path, clock):
    path = str(tmp_path / "rl.db")
    lim = TokenRateLimiter(path, limit=50_000, clock=clock)
    for _ in range(10):
        await lim.reserve("k", 10)
    clock.advance(61)
    await lim.reserve("k", 10)
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM token_events").fetchone()[0] == 1


# ----------------------------------------------------- token tracking: settle/release
async def test_settle_refunds_unused_and_charges_overage(lim):
    r = await lim.reserve("k", 10_000)
    await lim.settle(r, 2_000)
    assert await lim.usage("k") == 2_000                # refund
    await lim.settle(r, 12_000)
    assert await lim.usage("k") == 12_000               # actual usage exceeded the estimate: charge it


async def test_release_frees_the_reservation(lim):
    r = await lim.reserve("k", 50_000)
    await denied(lim, "k", 1)
    await lim.release(r)
    await lim.reserve("k", 50_000)


async def test_settle_after_eviction_is_a_noop(lim, clock):
    r = await lim.reserve("k", 1_000)
    clock.advance(61)
    await lim.purge_expired()
    await lim.settle(r, 5_000)                          # request outlived the window; row already gone
    assert await lim.usage("k") == 0


# ------------------------------------------------------------------ persistence
async def test_state_is_on_disk_and_survives_a_new_instance(tmp_path, clock):
    path = str(tmp_path / "rl.db")
    await TokenRateLimiter(path, clock=clock).reserve("k", 40_000)
    assert os.path.getsize(path) > 0
    other = TokenRateLimiter(path, clock=clock)         # "restart"
    assert await other.usage("k") == 40_000
    await denied(other, "k", 20_000)


async def test_raw_api_keys_are_never_stored(tmp_path, clock):
    path = str(tmp_path / "rl.db")
    lim = TokenRateLimiter(path, clock=clock)
    await lim.reserve("sk-super-secret-key-123", 10)
    blob = b"".join(open(path + s, "rb").read() for s in ("", "-wal") if os.path.exists(path + s))
    assert b"sk-super-secret-key-123" not in blob
    assert tenant_id("sk-super-secret-key-123").encode() in blob


# ------------------------------------------------------------------ concurrency
async def test_concurrent_reservations_never_overshoot(lim):
    results = await asyncio.gather(*[lim.reserve("k", 1_000) for _ in range(200)], return_exceptions=True)
    ok = [r for r in results if not isinstance(r, Exception)]
    assert len(ok) == 50                                # exactly 50 x 1000 = 50k fit
    assert all(isinstance(r, RateLimitExceeded) for r in results if isinstance(r, Exception))
    assert await lim.usage("k") == 50_000


def test_separate_processes_sharing_one_db_file_never_overshoot(tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    db = str(tmp_path / "shared.db")
    TokenRateLimiter(db)                                # create schema first
    from mp_worker import worker
    procs = [ctx.Process(target=worker, args=(db, 40, 1_000, q)) for _ in range(4)]   # 160 attempts for 50 slots
    for p in procs:
        p.start()
    got = [q.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=30)
    assert sum(got) == 50, got


# ----------------------------------------------------------- cancellation safety
async def test_cancelling_a_reserve_in_flight_does_not_leak_the_reservation(lim):
    orig = lim._reserve_sync

    def slow(key, tokens):
        import time
        time.sleep(0.2)                                 # DB work finishes AFTER the caller is cancelled
        return orig(key, tokens)

    lim._reserve_sync = slow
    task = asyncio.ensure_future(lim.reserve("k", 10_000))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.5)                            # let the thread finish + cleanup callback run
    assert await lim.usage("k") == 0, "reservation leaked after cancellation"


async def test_invalid_config_rejected(tmp_path):
    with pytest.raises(ValueError):
        TokenRateLimiter(str(tmp_path / "x.db"), limit=0)
