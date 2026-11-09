"""Spawns the server as a real subprocess and speaks raw JSON-RPC over stdio."""
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(HERE, "server.py")

INIT = {"jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
INITED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def run(requests, script=SERVER, timeout=20):
    """Behave like a real MCP client: keep stdin open until every response arrives, then close."""
    msgs = [INIT, INITED, *requests]
    want = {m["id"] for m in msgs if "id" in m}
    p = subprocess.Popen([sys.executable, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    out_lines, got = [], set()
    try:
        p.stdin.write("\n".join(json.dumps(m) for m in msgs) + "\n")
        p.stdin.flush()
        deadline = time.monotonic() + timeout
        while got != want and time.monotonic() < deadline:
            line = p.stdout.readline()
            if not line:
                break
            out_lines.append(line)
            if line.strip():
                got.add(json.loads(line).get("id"))  # raises if stdout is polluted
        p.stdin.close()
        rest_out, err = p.communicate(timeout=10)
    finally:
        if p.poll() is None:
            p.kill()
    out = "".join(out_lines) + rest_out
    parsed = [json.loads(l) for l in out.splitlines() if l.strip()]
    return parsed, out, err


def call(name, args, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": {"name": name, "arguments": args}}


def by_id(msgs, id_):
    return next(m for m in msgs if m.get("id") == id_)


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr():
    msgs, out, err = run([call("get_customer_record", {"customer_id": "CUST-00001"})])
    assert all(m.get("jsonrpc") == "2.0" for m in msgs)
    assert "Ada Lovelace" in json.dumps(by_id(msgs, 1))
    assert "tools/call name=get_customer_record" in err      # logs exist, on stderr
    assert "tools/call" not in out                            # ...and never on stdout


def test_stray_print_inside_handler_cannot_corrupt_stdout(tmp_path):
    """Simulate the classic bug: a print() in tool code. stdout must stay pure."""
    wrapper = tmp_path / "wrapped.py"
    wrapper.write_text(textwrap.dedent(f"""
        import sys, anyio
        sys.path.insert(0, {HERE!r})
        import server
        orig = server.on_call_tool
        async def noisy(ctx, params):
            print("DEBUG: this would break the protocol")
            sys.stdout.write("more junk\\n"); sys.stdout.flush()
            return await orig(ctx, params)
        server.on_call_tool = noisy
        anyio.run(server.main)
    """))
    msgs, out, err = run([call("get_customer_record", {"customer_id": "CUST-00001"})], script=str(wrapper))
    assert "DEBUG" not in out and "junk" not in out
    assert "Ada Lovelace" in json.dumps(by_id(msgs, 1))


def test_list_tools_exposes_both_tools():
    msgs, *_ = run([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
    names = {t["name"] for t in by_id(msgs, 1)["result"]["tools"]}
    assert names == {"get_customer_record", "trigger_refund"}


@pytest.mark.parametrize("bad_id", [
    "cust-00001", "CUST-0001", "CUST-000001", "CUST-ABCDE", "CUST-12345\n", " CUST-12345", "CUST-12345 ",
    "", 12345, None, ["CUST-00001"], {"a": 1}, True,
])
def test_get_customer_rejects_malformed_ids(bad_id):
    msgs, *_ = run([call("get_customer_record", {"customer_id": bad_id})])
    err = by_id(msgs, 1)["error"]
    assert err["code"] == -32602


def test_missing_and_extra_fields_rejected():
    msgs, *_ = run([
        call("get_customer_record", {}, 1),
        call("get_customer_record", {"customer_id": "CUST-00001", "admin": True}, 2),
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_customer_record"}},
    ])
    for i in (1, 2, 3):
        assert by_id(msgs, i)["error"]["code"] == -32602


@pytest.mark.parametrize("amount", [0, -1, -0.01, "10", "10.5", None, True, False, [1], {"v": 1}, 1e999])
def test_refund_rejects_bad_amounts(amount):
    # 1e999 serialises to `Infinity` in Python json; either parse failure or validation failure must be an error
    msgs, *_ = run([call("trigger_refund", {"customer_id": "CUST-00001", "amount": amount, "reason": "duplicate charge"})])
    m = by_id(msgs, 1)
    assert "error" in m and m["error"]["code"] in (-32602, -32700)


@pytest.mark.parametrize("reason", ["", "short", "123456789", "          ", "   a   b   ", 12345678901, None])
def test_refund_rejects_bad_reasons(reason):
    msgs, *_ = run([call("trigger_refund", {"customer_id": "CUST-00001", "amount": 10.0, "reason": reason})])
    assert by_id(msgs, 1)["error"]["code"] == -32602


def test_error_does_not_echo_input_values():
    secret = "SECRET-DO-NOT-ECHO"
    msgs, out, _ = run([call("trigger_refund", {"customer_id": secret, "amount": 1, "reason": "valid reason text"})])
    assert by_id(msgs, 1)["error"]["code"] == -32602
    assert secret not in out


def test_unknown_tool_is_invalid_params():
    msgs, *_ = run([call("drop_database", {})])
    assert by_id(msgs, 1)["error"]["code"] == -32602


def test_valid_refund_succeeds_with_int_or_float_amount():
    msgs, *_ = run([
        call("trigger_refund", {"customer_id": "CUST-00001", "amount": 25.5, "reason": "duplicate charge"}, 1),
        call("trigger_refund", {"customer_id": "CUST-00002", "amount": 5, "reason": "  item arrived damaged  "}, 2),
    ])
    for i in (1, 2):
        res = by_id(msgs, i)["result"]
        assert not res.get("isError")
        assert "Refund RF-" in res["content"][0]["text"]


def test_business_error_unknown_customer_is_tool_error_not_protocol_error():
    msgs, *_ = run([call("get_customer_record", {"customer_id": "CUST-99999"})])
    res = by_id(msgs, 1)["result"]
    assert res["isError"] is True and "not found" in res["content"][0]["text"]
