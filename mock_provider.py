"""Mock OpenAI-compatible provider that streams scripted tokens (for tests / demo)."""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Sequence

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


def create_provider(tokens: Sequence[str], *, gate: threading.Event | None = None, gate_after: int = 1,
                    fail_after: int | None = None, status: int = 200, n_choices: int = 1) -> Starlette:
    state = {"closed": threading.Event(), "tokens_sent": 0}

    def chunk(idx: int, delta: dict, finish=None) -> bytes:
        obj = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "mock-1",
               "choices": [{"index": idx, "delta": delta, "finish_reason": finish}]}
        return b"data: " + json.dumps(obj).encode() + b"\n\n"

    async def handler(request: Request):
        if status != 200:
            return JSONResponse({"error": "Traceback (most recent call last): File '/srv/secret.py' ... db=postgres://u:p@h"},
                                status_code=status)
        body = await request.json()
        if not body.get("stream"):
            text = "".join(tokens)
            return JSONResponse({"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                         "finish_reason": "stop"}]})

        async def gen():
            try:
                for i in range(n_choices):
                    yield chunk(i, {"role": "assistant", "content": ""})
                for k, tok in enumerate(tokens):
                    if fail_after is not None and k >= fail_after:
                        raise RuntimeError("provider exploded")
                    for i in range(n_choices):
                        yield chunk(i, {"content": tok})
                    state["tokens_sent"] += 1
                    if gate is not None and k + 1 == gate_after:
                        while not gate.is_set():
                            await asyncio.sleep(0.01)
                    await asyncio.sleep(0)
                for i in range(n_choices):
                    yield chunk(i, {}, "stop")
                yield b"data: [DONE]\n\n"
            finally:
                state["closed"].set()

        return StreamingResponse(gen(), media_type="text/event-stream")

    app = Starlette(routes=[Route("/v1/chat/completions", handler, methods=["POST"])])
    app.state.info = state
    return app
