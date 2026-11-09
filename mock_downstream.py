"""Minimal mock downstream MCP server (HTTP JSON-RPC). Counts calls so tests can prove
a blocked request never reached it."""
from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

TOOLS = [
    {"name": "get_status", "description": "Public status", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Rotate API key", "inputSchema": {"type": "object"}},
]


def create_mock() -> Starlette:
    calls: list[dict[str, Any]] = []
    seen_headers: list[dict[str, str]] = []

    def one(msg: dict[str, Any]) -> dict[str, Any] | None:
        calls.append(msg)
        if "id" not in msg:
            return None
        m = msg.get("method")
        if m == "tools/list":
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": TOOLS}}
        if m == "tools/call":
            name = msg["params"]["name"]
            if name not in {t["name"] for t in TOOLS}:
                return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
            return {"jsonrpc": "2.0", "id": msg["id"],
                    "result": {"content": [{"type": "text", "text": f"ran {name}"}], "isError": False}}
        return {"jsonrpc": "2.0", "id": msg["id"], "result": {}}

    async def mcp(request: Request) -> Response:
        seen_headers.append(dict(request.headers))
        data = await request.json()
        if isinstance(data, list):
            out = [r for r in (one(m) for m in data) if r is not None]
            return JSONResponse(out) if out else Response(status_code=202)
        r = one(data)
        return JSONResponse(r) if r is not None else Response(status_code=202)

    app = Starlette(routes=[Route("/mcp", mcp, methods=["POST"])])
    app.state.calls = calls
    app.state.seen_headers = seen_headers
    return app
