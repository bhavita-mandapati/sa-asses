# Solution Architect Assessment: MCP & LLM Gateway Tasks

All four tasks are implemented in Python 3.13, each in its own self-contained directory with tests.

```
pip install -r requirements.txt
./run_tests.sh          # 247 tests: 40 + 50 + 71 + 86
```

| Task | Dir | What it is | Tests |
|---|---|---|---|
| 1 | `task1_mcp_server/` | MCP server (stdio), strict Pydantic validation, JSON-RPC errors | 40 |
| 2 | `task2_mcp_gateway/` | MCP reverse proxy: bearer role extraction, `admin_*` tool authorization | 50 |
| 3 | `task3_stream_guardrail/` | LLM gateway with streaming PII redaction (SSE) | 71 |
| 4 | `task4_router/` | Token-aware SQLite rate limiter + 3000 ms failover router | 86 |

---

## Task 1: MCP server

`server.py` uses the SDK's low-level `Server` so validation and error mapping are explicit.

- **Validation:** strict Pydantic models with `extra="forbid"`. `amount` rejects `0`, negatives, `"10"` (strings), booleans, `NaN` and `Infinity`. `reason` is stripped before the 10-character minimum, so ten spaces fails. `CUST-12345\n` is rejected.
- **Error mapping:** invalid arguments and unknown tools return JSON-RPC `-32602` with field-level detail (`data.errors`) that never echoes the submitted values. A missing customer is a *tool* error (`isError: true`), not a protocol error, so the model can self-correct (per the MCP spec).
- **stdout isolation:** no `print()` anywhere, logging goes to stderr, and the SDK's `stdio_server` also points fd 1 at stderr while serving. A test injects `print()` and `sys.stdout.write()` into a tool handler and asserts stdout is still pure JSON-RPC.
- Also cross-checked with the SDK's own client over real stdio.

## Task 2: MCP gateway

`gateway.py` (Starlette + httpx). Tokens are HS256 JWTs with a `role` claim.

- `tools/list` and every non-`tools/call` method are forwarded unchanged. `tools/call` on an `admin_*` tool by a non-admin returns `-32001 Unauthorized Tool Call` with the original request `id`, and the downstream is never contacted (the tests assert zero downstream calls).
- **Batch smuggling:** JSON-RPC batches are authorized element by element. A naive "inspect the object" check would let a viewer hide `admin_*` calls inside an array.
- **Name normalization:** `ADMIN_x` and ` admin_x` are denied, since a lenient downstream could normalize them.
- **JWT hygiene:** `alg: none`, wrong signature, expiry and malformed tokens are all rejected, with constant-time signature comparison. Role must be exactly `"admin"`; missing or non-string means no privilege.
- The user's `Authorization` header is not forwarded downstream. SSE responses stream through unbuffered. Downstream failures return sanitized 502/504.
- Verified over real TCP with uvicorn as well as in-process.

## Task 3: Streaming PII guardrail

`redactor.py` is the core. The hard part is a secret split across chunks (`123-45-` | `6789`).

- **Approach:** hold back only the trailing text that could still become a match. Everything before that point is final and is emitted immediately. Typical lag is one word (about one token) and the hard cap is about 320 characters, so memory is constant. `ctx` remembers the last emitted character so lookbehind boundaries survive a cut.
- **Correctness:** a randomized property test checks that output does not depend on chunking, against an independent one-shot oracle. It ran 4,000 cases in the suite and 40,000 extra cases as a one-off, with zero mismatches.
- **Bugs this process found and fixed:**
  1. A complete card followed by `.` was split by the email hold-back. The cut now moves left to the start of any match that straddles it.
  2. A Luhn-failing "card" swallowed a real email inside it, so output depended on chunk alignment. Failed candidates no longer consume their span.
  3. Emails with local parts over 64 characters left an email-shaped tail. The email pattern is no longer left-anchored, so it over-redacts, which is the safe direction.
- `gateway.py` handles SSE parsing (byte-split safe), a redactor per choice index, a flush before `finish_reason` and `[DONE]`, and fail-closed handling of non-JSON `data:` payloads.
- **Streaming behavior:** one test proves chunks reach the client while the provider is still blocked mid-stream. It uses real sockets because `httpx.ASGITransport` buffers responses and would hide this. Client disconnect closes the upstream connection.
- **Numbers (`bench.py`):** about 1.7 µs per chunk on average, about 2 KiB peak heap over 2.9 MB. The single worst call was about 2 ms, which I did not investigate (likely GC or scheduler noise).

## Task 4: Rate limiter and fallback router

`limiter.py`, `router.py`, `providers.py`, `app.py`.

- **Limiter:** a sliding window log in on-disk SQLite (WAL). Check-and-consume runs inside `BEGIN IMMEDIATE`, so it is atomic across coroutines, threads **and separate processes**. One test has 4 processes race 160 reservations against one file and exactly 50 succeed.
- **Token-aware lifecycle:** reserve the estimate up front, settle to actual usage on success (refund or overage), and release on failure or cancellation, so tenants aren't billed for failed calls. In-flight work counts at its full estimate.
- **Window semantics:** a row counts while `ts > now - 60s`, so it expires at exactly 60.000 s. `Retry-After` is computed from the row whose expiry actually makes room, and a test confirms following it succeeds. API keys are never stored (truncated SHA-256).
- **Timeout race:** the provider call runs as its own task with a hard `asyncio.wait` deadline. On expiry it is cancelled and its result is never read, even if the provider swallows the cancellation and returns late (tested: a late 9,999-token result is not billed). If the call finished in the instant the deadline fired, its result is used rather than paying for a second call.
- **Cancellation safety:** a `to_thread` reservation can complete after the caller is cancelled, which would leak tokens. A shield plus a cleanup callback handles it. I mutation-checked this: removing the callback fails the test.
- **Errors:** every failure leaves as `GatewayError` with a fixed generic message and a `request_id`. Upstream bodies, exception text and provider names go only to the server log. Tests pin each message to a literal.

---

## Assumptions and judgment calls (please check these)

1. **MCP SDK version.** The environment has `mcp==2.3.0`, whose low-level API (`Server(..., on_call_tool=...)`) differs from 1.x's decorator style. Task 1 targets 2.x; on 1.x the handler registration would need adapting.
2. **`CUST-XXXXX`** is read as exactly five digits.
3. **Token format (Task 2)** isn't specified. I used HS256 JWT with a `role` claim and a secret from `GATEWAY_JWT_SECRET` (no default). Gateway-defined codes `-32002` (auth required) and `-32003` (downstream error) are my additions alongside the required `-32001`.
4. **`tools/list` is forwarded unfiltered** as specified, so a viewer can still *see* admin tool names; they just can't call them. Filtering the list is a one-line hardening if wanted.
5. **Card detection requires a Luhn pass** (to avoid redacting order IDs). SSNs need separators; bare 9-digit numbers are not matched because of false positives.
6. **Failover triggers (Task 4)** go beyond the spec's 429 and timeout to also cover 5xx, connection errors and upstream auth/config errors. Tenant errors (400/413/422) are deliberately not failed over. The secondary timeout (10 s) is my choice.
7. **Token estimate** is a ~4 chars/token heuristic plus `max_tokens`. It is pluggable, and settlement corrects it afterward.

## Known limitations

- **Task 3** scans only `delta.content` and `text`. Tool-call arguments and reasoning fields are not scanned, and the gateway has no client auth.
- **Task 4:** there is no circuit breaker, so during a primary outage every request pays the 3 s timeout before failing over. The SQLite limiter is correct across processes on one host, but not across machines (use Redis or similar there). I have not benchmarked its throughput.
- The assessment overview mentions "troubleshooting zero-trust network deployments", but none of the four tasks covers it, so it is not addressed here.
