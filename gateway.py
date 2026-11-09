"""Task 3 - LLM gateway endpoint with a streaming PII guardrail.

POST /v1/chat/completions   (OpenAI-compatible; `"stream": true` -> SSE)

  client <--SSE-- gateway(guardrail) <--SSE-- LLM provider

Pipeline (all incremental, constant memory):
    upstream bytes -> SSEParser -> StreamGuard (per-choice StreamRedactor) -> SSE frames -> client

* TTFT: response headers are sent as soon as the provider answers; every chunk is forwarded the
  moment it is parsed. The only added delay is the redactor's hold-back (at most the current
  word, ~1 token) because a secret may straddle two chunks.
* Memory: nothing accumulates. SSE parser buffer is capped; each redactor holds <= ~320 chars.
* Fail-closed: non-JSON `data:` payloads are redacted as raw text instead of passed through.
* Errors are sanitised: upstream bodies / stack traces / URLs are never returned to the client.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import anyio
import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from redactor import StreamRedactor, redact_text

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s guardrail: %(message)s")
log = logging.getLogger("llm-gateway")


# ----------------------------------------------------------------------------- SSE
class SSEError(Exception):
    pass


@dataclass
class SSEEvent:
    lines: list[bytes]  # raw lines (no terminator) in arrival order

    @property
    def data(self) -> str | None:
        parts = []
        for ln in self.lines:
            if ln.startswith(b"data:"):
                v = ln[5:]
                parts.append((v[1:] if v.startswith(b" ") else v).decode("utf-8", "replace"))
        return "\n".join(parts) if parts else None

    def other_lines(self) -> list[bytes]:
        return [ln for ln in self.lines if not ln.startswith(b"data:")]


class SSEParser:
    """Incremental SSE parser. Robust to arbitrary byte-level splits (incl. mid-UTF-8 and \\r\\n)."""

    def __init__(self, max_event_bytes: int = 1 << 20) -> None:
        self._buf = bytearray()
        self._scan = 0
        self._lines: list[bytes] = []
        self._event_bytes = 0
        self._max = max_event_bytes

    def feed(self, data: bytes) -> list[SSEEvent]:
        self._buf += data
        events: list[SSEEvent] = []
        while True:
            nl = self._buf.find(b"\n", self._scan)
            if nl < 0:
                self._scan = len(self._buf)
                if len(self._buf) + self._event_bytes > self._max:
                    raise SSEError("event too large")
                return events
            line = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            self._scan = 0
            if line.endswith(b"\r"):
                line = line[:-1]
            if line == b"":
                if self._lines:
                    events.append(SSEEvent(self._lines))
                    self._lines, self._event_bytes = [], 0
                continue
            self._event_bytes += len(line)
            if self._event_bytes > self._max:
                raise SSEError("event too large")
            self._lines.append(line)

    def finish(self) -> list[SSEEvent]:
        """Upstream closed: dispatch a trailing, unterminated event if any."""
        if self._buf:
            self._lines.append(bytes(self._buf).rstrip(b"\r"))
            self._buf.clear()
        if self._lines:
            ev, self._lines, self._event_bytes = SSEEvent(self._lines), [], 0
            return [ev]
        return []


def frame(data: str, other: list[bytes] | None = None) -> bytes:
    head = b"".join(ln + b"\n" for ln in (other or []))
    return head + b"".join(b"data: " + part.encode() + b"\n" for part in data.split("\n")) + b"\n"


def raw_frame(ev: SSEEvent) -> bytes:
    return b"".join(ln + b"\n" for ln in ev.lines) + b"\n"


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------------------ the guard
class StreamGuard:
    """Turns upstream SSE events into redacted downstream SSE frames."""

    def __init__(self) -> None:
        self._red: dict[int, StreamRedactor] = {}
        self._template: dict[str, Any] = {}  # id/model/created reused for synthetic flush chunks

    def _r(self, idx: int) -> StreamRedactor:
        r = self._red.get(idx)
        if r is None:
            r = self._red[idx] = StreamRedactor()
        return r

    def process(self, ev: SSEEvent) -> list[bytes]:
        data = ev.data
        if data is None:  # comment / keep-alive / event-name only
            return [raw_frame(ev)]
        if data.strip() == "[DONE]":
            return self.flush_frames() + [raw_frame(ev)]
        try:
            obj = json.loads(data)
        except ValueError:
            return [frame(redact_text(data), ev.other_lines())]  # fail closed: scrub as plain text
        if not isinstance(obj, dict):
            return [frame(redact_text(data), ev.other_lines())]

        for k in ("id", "object", "created", "model"):
            if k in obj:
                self._template[k] = obj[k]
        changed = False
        choices = obj.get("choices")
        for ch in choices if isinstance(choices, list) else []:
            if not isinstance(ch, dict):
                continue
            idx = ch.get("index", 0) if isinstance(ch.get("index", 0), int) else 0
            holder, key = None, None
            delta = ch.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                holder, key = delta, "content"
            elif isinstance(ch.get("text"), str):  # legacy /completions shape
                holder, key = ch, "text"
            finishing = ch.get("finish_reason") is not None
            if holder is None and not finishing:
                continue
            text = holder[key] if holder is not None else ""
            red = self._r(idx)
            out = red.feed(text)
            if finishing:
                out += red.flush()  # nothing may stay held back past the final chunk
            if holder is None:
                if out:
                    ch.setdefault("delta", {})["content"] = out
                    changed = True
            elif out != text:
                holder[key] = out
                changed = True
        if not changed:
            return [raw_frame(ev)]
        return [frame(dumps(obj), ev.other_lines())]

    def flush_frames(self) -> list[bytes]:
        frames = []
        for idx, red in sorted(self._red.items()):
            tail = red.flush()
            if tail:
                chunk = {**self._template, "choices": [{"index": idx, "delta": {"content": tail}, "finish_reason": None}]}
                chunk.setdefault("object", "chat.completion.chunk")
                frames.append(frame(dumps(chunk)))
        return frames


# --------------------------------------------------------------------------- app
@dataclass
class Config:
    upstream_url: str
    upstream_api_key: str | None = None
    max_body_bytes: int = 2_000_000
    connect_timeout: float = 5.0
    read_timeout: float = 60.0  # per-chunk idle timeout (not total), appropriate for streams
    client: httpx.AsyncClient | None = field(default=None, repr=False)


def error_json(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


STREAM_ERROR_FRAME = frame(dumps({"error": {"type": "upstream_error", "message": "Upstream stream interrupted."}}))


def create_app(cfg: Config) -> Starlette:
    client = cfg.client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=cfg.connect_timeout, read=cfg.read_timeout, write=10.0, pool=5.0))

    async def completions(request: Request) -> Response:
        body = await request.body()
        if len(body) > cfg.max_body_bytes:
            return error_json(413, "invalid_request", "Request too large.")
        try:
            payload = json.loads(body)
        except ValueError:
            return error_json(400, "invalid_request", "Body must be valid JSON.")
        if not isinstance(payload, dict):
            return error_json(400, "invalid_request", "Body must be a JSON object.")
        streaming = payload.get("stream") is True

        headers = {"content-type": "application/json", "accept": "text/event-stream" if streaming else "application/json",
                   "accept-encoding": "identity"}
        if cfg.upstream_api_key:
            headers["authorization"] = f"Bearer {cfg.upstream_api_key}"

        try:
            req = client.build_request("POST", cfg.upstream_url, content=body, headers=headers)
            resp = await client.send(req, stream=True)
        except httpx.TimeoutException:
            log.error("upstream timeout")
            return error_json(504, "upstream_timeout", "The upstream provider timed out.")
        except httpx.HTTPError as e:
            log.error("upstream connect error: %r", e)
            return error_json(502, "upstream_error", "The upstream provider is unavailable.")

        if resp.status_code >= 400:
            await resp.aclose()
            log.warning("upstream status %s", resp.status_code)
            status = 429 if resp.status_code == 429 else 504 if resp.status_code in (408, 504) else 502
            return error_json(status, "upstream_error", "The upstream provider returned an error.")

        if not streaming:
            try:
                data = json.loads(await resp.aread())
            except (ValueError, httpx.HTTPError):
                return error_json(502, "upstream_error", "The upstream provider returned an invalid response.")
            finally:
                await resp.aclose()
            for ch in data.get("choices", []) if isinstance(data, dict) else []:
                if isinstance(ch, dict):
                    if isinstance(ch.get("message"), dict) and isinstance(ch["message"].get("content"), str):
                        ch["message"]["content"] = redact_text(ch["message"]["content"])
                    if isinstance(ch.get("text"), str):
                        ch["text"] = redact_text(ch["text"])
            return JSONResponse(data)

        async def body_iter() -> AsyncIterator[bytes]:
            guard, parser = StreamGuard(), SSEParser()
            clean_end = False
            try:
                async for chunk in resp.aiter_bytes():
                    for ev in parser.feed(chunk):
                        for out in guard.process(ev):
                            yield out
                for ev in parser.finish():
                    for out in guard.process(ev):
                        yield out
                for out in guard.flush_frames():  # provider closed without [DONE]
                    yield out
                clean_end = True
            except (httpx.HTTPError, SSEError) as e:
                log.error("stream interrupted: %r", e)
                for out in guard.flush_frames():
                    yield out
                yield STREAM_ERROR_FRAME
            finally:
                with anyio.CancelScope(shield=True):  # runs even when the client disconnects
                    await resp.aclose()
                if not clean_end:
                    log.info("stream ended early")

        return StreamingResponse(body_iter(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    app = Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"]),
                            Route("/healthz", health, methods=["GET"])])
    app.state.client = client
    return app


if __name__ == "__main__":  # pragma: no cover
    import os

    import uvicorn

    app = create_app(Config(os.environ["UPSTREAM_URL"], os.environ.get("UPSTREAM_API_KEY")))
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8081")))
