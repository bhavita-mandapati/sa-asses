"""Cross-check against the SDK's own client over real stdio."""
import os
import sys

import anyio
import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError

SERVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server.py")
PARAMS = StdioServerParameters(command=sys.executable, args=[SERVER])


def test_official_client_roundtrip():
    async def go():
        async with Client(PARAMS) as c:
            tools = await c.list_tools()
            assert {t.name for t in tools.tools} == {"get_customer_record", "trigger_refund"}
            ok = await c.call_tool("get_customer_record", {"customer_id": "CUST-00001"})
            assert "Ada Lovelace" in ok.content[0].text
            with pytest.raises(MCPError) as ei:
                await c.call_tool("trigger_refund", {"customer_id": "CUST-00001", "amount": -5, "reason": "too short"})
            assert ei.value.code == -32602
    anyio.run(go)
