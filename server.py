"""Task 1 - Custom MCP server with strict validation and clean stdio transport.

Tools
-----
get_customer_record(customer_id: "CUST-XXXXX")
trigger_refund(customer_id, amount > 0, reason >= 10 chars)

Design notes
------------
* Uses the low-level ``mcp.server.Server`` so validation and error mapping are
  explicit and visible, rather than hidden in a decorator framework.
* Validation is Pydantic (strict, ``extra=forbid``). A validation failure is a
  *protocol* error -> JSON-RPC ``-32602 Invalid params``; an unknown tool is also
  ``-32602`` (per the MCP spec). Business failures (customer not found) are
  returned as ``CallToolResult(is_error=True)`` so the LLM can self-correct.
* stdout is reserved for JSON-RPC. All logging goes to stderr, and nothing in this
  file calls ``print()``. ``stdio_server`` additionally points fd 1 at stderr for
  the lifetime of the server, so stray writes from libraries cannot corrupt framing.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Annotated, Any

import anyio
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

# --- logging: stderr only -------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("customer-mcp")

# --- input schemas --------------------------------------------------------
# NOTE: "XXXXX" is interpreted as exactly five ASCII digits. Anchors are safe here:
# pydantic's regex engine treats `$` as end-of-text (no trailing-newline allowance).
CustomerId = Annotated[str, StringConstraints(pattern=r"^CUST-[0-9]{5}$")]


class GetCustomerRecordInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customer_id: CustomerId


class TriggerRefundInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customer_id: CustomerId
    # strict: no "12.5" strings, no bools; allow_inf_nan=False: rejects Infinity/NaN
    amount: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    # strip first so 10 spaces is not a valid reason
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=500)]


TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "get_customer_record": (
        GetCustomerRecordInput,
        "Look up a customer record by ID (format CUST-XXXXX, five digits).",
    ),
    "trigger_refund": (
        TriggerRefundInput,
        "Issue a refund to a customer. amount must be > 0; reason must be at least 10 characters.",
    ),
}

# --- mock backing store ---------------------------------------------------
CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"customer_id": "CUST-00001", "name": "Ada Lovelace", "tier": "gold", "balance_due": 0.0},
    "CUST-00002": {"customer_id": "CUST-00002", "name": "Alan Turing", "tier": "silver", "balance_due": 42.5},
}


def _validate(tool_name: str, arguments: dict[str, Any] | None) -> BaseModel:
    if tool_name not in TOOLS:
        raise MCPError(types.INVALID_PARAMS, f"Unknown tool: {tool_name}")
    model_cls, _ = TOOLS[tool_name]
    try:
        return model_cls.model_validate(arguments or {})
    except ValidationError as exc:
        # Report location + reason only; never echo raw input values back.
        problems = [
            {"field": ".".join(str(p) for p in e["loc"]) or "<root>", "reason": e["msg"], "type": e["type"]}
            for e in exc.errors(include_input=False, include_url=False, include_context=False)
        ]
        raise MCPError(types.INVALID_PARAMS, f"Invalid arguments for {tool_name}", data={"errors": problems}) from None


def _text(payload: str, *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=payload)], is_error=is_error)


# --- handlers -------------------------------------------------------------
async def on_list_tools(ctx: ServerRequestContext, params: Any) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(name=name, description=desc, input_schema=model.model_json_schema())
            for name, (model, desc) in TOOLS.items()
        ]
    )


async def on_call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> types.CallToolResult:
    args = _validate(params.name, params.arguments)
    log.info("tools/call name=%s", params.name)  # stderr; never log amounts/reasons (PII-adjacent)

    if isinstance(args, GetCustomerRecordInput):
        record = CUSTOMERS.get(args.customer_id)
        if record is None:
            return _text(f"Customer {args.customer_id} not found.", is_error=True)
        return _text(json.dumps(record))

    assert isinstance(args, TriggerRefundInput)
    if args.customer_id not in CUSTOMERS:
        return _text(f"Customer {args.customer_id} not found.", is_error=True)
    refund_id = f"RF-{uuid.uuid4().hex[:10]}"
    log.info("refund issued id=%s customer=%s", refund_id, args.customer_id)
    return _text(f"Refund {refund_id} of {args.amount:.2f} issued to {args.customer_id}.")


def build_server() -> Server:
    return Server("customer-mcp", version="1.0.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def main() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
