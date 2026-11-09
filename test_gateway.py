import json
import random
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from gateway import Config, SSEError, SSEEvent, SSEParser, StreamGuard, create_app
from mock_provider import create_provider

PII_TOKENS = ["Email ", "john", ".doe", "@exam", "ple.com", " SSN ", "123", "-45-", "6789", ". Card ",
              "4111", " 1111 ", "1111 1111", ".", " Bye"]
CLEAN = "Email [REDACTED] SSN [REDACTED]. Card [REDACTED]. Bye"
LEAKS = ["john.doe", "example.com", "123-45-6789", "4111 1111 1111 1111"]


# --------------------------------------------------------------- test servers
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Srv:
    def __init__(self, app, port=None):
        self.port = port or free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="critical"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return self
            time.sleep(0.02)
        raise RuntimeError("server did not start")

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


def stack(provider_app):
    """provider + gateway, returns (context-manager-ish tuple)"""
    prov = Srv(provider_app)
    return prov


class Stack:
    def __init__(self, provider_app, **cfg):
        self.prov = Srv(provider_app)
        self.cfg = cfg

    def __enter__(self):
        self.prov.__enter__()
        self.gw = Srv(create_app(Config(f"{self.prov.url}/v1/chat/completions", **self.cfg)))
        self.gw.__enter__()
        return self

    def __exit__(self, *a):
        self.gw.__exit__()
        self.prov.__exit__()

    def post(self, **body):
        return httpx.post(f"{self.gw.url}/v1/chat/completions", json=body, timeout=10)

    def stream(self, **body):
        return httpx.stream("POST", f"{self.gw.url}/v1/chat/completions", json={"stream": True, **body}, timeout=10)


def sse_data(raw: bytes):
    out = []
    for block in raw.decode().split("\n\n"):
        for ln in block.split("\n"):
            if ln.startswith("data: "):
                out.append(ln[6:])
    return out


def content_of(raw: bytes):
    text = ""
    for d in sse_data(raw):
        if d.strip() != "[DONE]":
            for ch in json.loads(d).get("choices", []):
                text += (ch.get("delta") or {}).get("content") or ""
    return text


# ------------------------------------------------------------------ SSE parser
def make_stream():
    events = [b'data: {"a":"caf\xc3\xa9 \xe2\x9c\x93"}\n\n', b": keepalive\n\n", b"event: x\ndata: one\ndata: two\n\n",
              b'data: {"b":1}\r\n\r\n', b"data: [DONE]\n\n"]
    return b"".join(events)


def test_parser_is_split_invariant_including_mid_utf8_and_crlf():
    raw = make_stream()
    want = [e.lines for e in SSEParser().feed(raw)]
    assert len(want) == 5
    rng = random.Random(3)
    for _ in range(500):
        cuts = sorted(rng.sample(range(1, len(raw)), rng.randint(1, 12)))
        p, got, prev = SSEParser(), [], 0
        for c in cuts + [len(raw)]:
            got += [e.lines for e in p.feed(raw[prev:c])]
            prev = c
        assert got == want
    assert [e.lines for e in SSEParser().feed(raw[:1])] == []  # one byte at a time too
    p, got = SSEParser(), []
    for i in range(len(raw)):
        got += [e.lines for e in p.feed(raw[i:i + 1])]
    assert got == want


def test_parser_multiline_data_and_oversize():
    ev = SSEParser().feed(b"data: one\ndata: two\n\n")[0]
    assert ev.data == "one\ntwo"
    with pytest.raises(SSEError):
        SSEParser(max_event_bytes=100).feed(b"data: " + b"x" * 500)  # no newline ever -> bounded
    with pytest.raises(SSEError):
        p = SSEParser(max_event_bytes=100)
        for _ in range(20):
            p.feed(b"data: xxxxxxxxxx\n")  # many lines, never terminated by blank line


def test_parser_finish_dispatches_unterminated_tail():
    p = SSEParser()
    assert p.feed(b"data: tail") == []
    assert p.finish()[0].data == "tail"


# ---------------------------------------------------------------- StreamGuard
def ev(obj_or_str):
    s = obj_or_str if isinstance(obj_or_str, str) else json.dumps(obj_or_str)
    return SSEParser().feed(f"data: {s}\n\n".encode())[0]


def delta(text, idx=0, finish=None):
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": idx, "delta": {"content": text}, "finish_reason": finish}]}


def run_guard(events):
    g, out = StreamGuard(), []
    for e in events:
        out += g.process(e)
    return b"".join(out)


def test_guard_flushes_held_back_text_before_finish_and_done():
    raw = run_guard([ev(delta("hello wor")), ev(delta("ld", finish="stop")), ev("[DONE]")])
    assert content_of(raw) == "hello world"
    assert raw.rstrip().endswith(b"data: [DONE]")
    # flushed inside the finish chunk, not after it
    datas = sse_data(raw)
    assert json.loads(datas[-2])["choices"][0]["finish_reason"] == "stop"


def test_guard_flushes_at_done_when_no_finish_chunk():
    raw = run_guard([ev(delta("trailing john")), ev("[DONE]")])
    assert content_of(raw) == "trailing john"
    assert raw.rstrip().endswith(b"data: [DONE]")


def test_guard_independent_state_per_choice():
    evs = [ev(delta("a@b", 0)), ev(delta("1", 1)), ev(delta(".com ok", 0)), ev(delta("23-45-6789 x", 1))]
    g = StreamGuard()
    texts = {0: "", 1: ""}
    for e in evs + [ev("[DONE]")]:
        for f in g.process(e):
            for d in sse_data(f):
                if d != "[DONE]":
                    for ch in json.loads(d)["choices"]:
                        texts[ch["index"]] += ch["delta"].get("content", "")
    assert texts[0] == "[REDACTED] ok" and texts[1] == "[REDACTED] x"


def test_guard_untouched_events_pass_through_byte_identical():
    role = {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
    e = ev(role)
    assert StreamGuard().process(e) == [b"data: " + json.dumps(role).encode() + b"\n\n"]
    ka = SSEParser().feed(b": keepalive\n\n")[0]
    assert StreamGuard().process(ka) == [b": keepalive\n\n"]


def test_guard_fails_closed_on_non_json_data():
    raw = run_guard([ev("plain text with a@b.com inside")])
    assert b"a@b.com" not in raw and b"[REDACTED]" in raw


def test_guard_legacy_text_completions_shape():
    raw = run_guard([ev({"choices": [{"index": 0, "text": "mail a@b.", "finish_reason": None}]}),
                     ev({"choices": [{"index": 0, "text": "com now", "finish_reason": "stop"}]})])
    got = "".join(ch["text"] for d in sse_data(raw) for ch in json.loads(d)["choices"])
    assert got == "mail [REDACTED] now"


@pytest.mark.parametrize("junk", [{"choices": None}, {"choices": "x"}, {"choices": [None, 5]}, [1, 2], 7,
                                  {"choices": [{"index": "zero", "delta": {"content": "a@b.com"}}]}])
def test_guard_tolerates_malformed_shapes(junk):
    out = run_guard([ev(junk), ev("[DONE]")])
    assert b"a@b.com" not in out


# ------------------------------------------------------------------------ e2e
def test_e2e_pii_split_across_events_is_fully_redacted():
    with Stack(create_provider(PII_TOKENS)) as s:
        with s.stream(messages=[]) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            raw = b"".join(r.iter_bytes())
    assert content_of(raw) == CLEAN
    for leak in LEAKS:
        assert leak.encode() not in raw
    assert raw.rstrip().endswith(b"data: [DONE]")


def test_e2e_chunks_are_forwarded_before_upstream_finishes():
    """TTFT / no-buffering: the client must get early chunks while the provider is still mid-stream."""
    gate = threading.Event()
    toks = ["Hello ", "there ", "friend ", "this ", "is ", "streaming ", "right ", "now. "]
    with Stack(create_provider(toks, gate=gate, gate_after=3)) as s:
        seen = []
        t0 = time.monotonic()
        with s.stream(messages=[]) as r:
            lines = r.iter_lines()                  # ONE iterator for the whole response
            for line in lines:
                if line.startswith("data: ") and line != "data: [DONE]":
                    c = json.loads(line[6:])["choices"][0]["delta"].get("content")
                    if c:
                        seen.append(c)
                if len("".join(seen)) >= len("Hello there "):
                    break
            first_text_latency = time.monotonic() - t0
            assert not gate.is_set()               # provider is STILL blocked -> we did not wait for the end
            assert "".join(seen).startswith("Hello there")
            gate.set()
            rest = "\n".join(lines)
        assert first_text_latency < 3
        assert "[DONE]" in rest


def test_e2e_upstream_http_errors_are_sanitised():
    with Stack(create_provider([], status=500)) as s:
        r = s.post(stream=True, messages=[])
    assert r.status_code == 502
    assert "Traceback" not in r.text and "postgres" not in r.text and "secret.py" not in r.text
    assert r.json()["error"]["type"] == "upstream_error"
    with Stack(create_provider([], status=429)) as s:
        assert s.post(stream=True).status_code == 429


def test_e2e_mid_stream_failure_emits_generic_error_and_no_leak():
    toks = ["partial john", "@ex", "ample.com is ", "x", "y", "z"]
    with Stack(create_provider(toks, fail_after=3)) as s:
        with s.stream(messages=[]) as r:
            raw = b"".join(r.iter_bytes())
    assert b"provider exploded" not in raw and b"RuntimeError" not in raw
    assert b"john@example.com" not in raw
    assert b"Upstream stream interrupted" in raw


def test_e2e_non_streaming_response_is_redacted_too():
    with Stack(create_provider(PII_TOKENS)) as s:
        r = s.post(messages=[], stream=False)
    assert r.json()["choices"][0]["message"]["content"] == CLEAN


def test_e2e_client_disconnect_closes_upstream():
    gate = threading.Event()                       # never released: provider would hang forever
    prov = create_provider(["one ", "two ", "three "], gate=gate, gate_after=1)
    with Stack(prov) as s:
        with s.stream(messages=[]) as r:
            for line in r.iter_lines():
                if "one" in line:
                    break
        # leaving the context manager closes the client connection
        assert prov.state.info["closed"].wait(timeout=5), "upstream stream was not closed after client disconnect"
        gate.set()


def test_e2e_bad_requests():
    with Stack(create_provider(["x"]), max_body_bytes=1000) as s:
        url = f"{s.gw.url}/v1/chat/completions"
        assert httpx.post(url, content=b"{nope").status_code == 400
        assert httpx.post(url, content=b"[1,2]").status_code == 400
        assert httpx.post(url, content=b'{"p":"' + b"x" * 2000 + b'"}').status_code == 413


def test_e2e_upstream_unreachable_is_sanitised():
    port = free_port()  # nothing listening
    with Srv(create_app(Config(f"http://127.0.0.1:{port}/v1/chat/completions"))) as gw:
        r = httpx.post(f"{gw.url}/v1/chat/completions", json={"stream": True}, timeout=10)
    assert r.status_code == 502 and str(port) not in r.text and "127.0.0.1" not in r.text
