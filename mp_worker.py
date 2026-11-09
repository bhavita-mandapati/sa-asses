"""Child-process worker for the cross-process limiter test (must be importable under 'spawn')."""
import asyncio


def worker(db_path, attempts, tokens, q):
    from limiter import RateLimitExceeded, TokenRateLimiter

    lim = TokenRateLimiter(db_path, limit=50_000)

    async def go():
        ok = 0
        for _ in range(attempts):
            try:
                await lim.reserve("shared-tenant-key", tokens)
                ok += 1
            except RateLimitExceeded:
                pass
        return ok

    q.put(asyncio.run(go()))
