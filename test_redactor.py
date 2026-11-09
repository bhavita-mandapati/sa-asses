import random
import re
import tracemalloc

import pytest

from redactor import (MAX_EMAIL_RUN, REDACTED, StreamRedactor, _WITH_EMAIL, luhn_ok, redact_text)

R = REDACTED


# ------------------------------------------------------------------ detection
@pytest.mark.parametrize("pii", [
    "john.doe@example.com", "a+b@sub.domain.co.uk", "USER_1@EX.IO", "x@y.museum",
    "123-45-6789", "123 45 6789",
    "4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111",
    "3782 822463 10005", "378282246310005", "5500 0000 0000 0004", "4111 1111 1111 1",  # 13 digits grouped? see below
])
def test_pii_is_redacted(pii):
    if pii == "4111 1111 1111 1":  # 13 digits, Luhn-invalid -> must NOT be redacted
        assert redact_text(f"x {pii} y") == f"x {pii} y"
        return
    assert redact_text(f"before {pii} after") == f"before {R} after"


@pytest.mark.parametrize("text", [
    "version 1.2.3", "date 2024-11-06", "order 1234567890123456", "000-12-3456", "666-12-3456", "900-12-3456",
    "123-00-4567", "123-45-0000", "call 555-123-4567", "foo@bar", "@handle", "a@b.c", "x@.com", "just text, no pii.",
    "1699999999999 ms", "12345678901234567890",  # 20 digits: too long for a card
    "4111 1111 1111 1112",                        # fails Luhn
])
def test_no_false_positives(text):
    assert redact_text(text) == text


@pytest.mark.parametrize("text,expected", [
    ("mail me at a@b.com.", f"mail me at {R}."),
    ("<a@b.com>", f"<{R}>"),
    ('"a@b.com",', f'"{R}",'),
    ("a@b.com,c@d.org", f"{R},{R}"),
    ("SSN:123-45-6789;", f"SSN:{R};"),
    ("card=4111111111111111\n", f"card={R}\n"),
    ("é123-45-6789é", f"é{R}é"),
])
def test_boundaries_and_punctuation(text, expected):
    assert redact_text(text) == expected


def test_luhn():
    assert luhn_ok("4111111111111111") and luhn_ok("378282246310005")
    assert not luhn_ok("4111111111111112") and not luhn_ok("123")


# -------------------------------------------------- chunk-boundary handling
TEXT = "Contact john.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111. Bye a@b.org"
EXPECTED = f"Contact {R}, SSN {R}, card {R}. Bye {R}"


def test_every_two_way_split_point_matches():
    for i in range(len(TEXT) + 1):
        r = StreamRedactor()
        assert r.feed(TEXT[:i]) + r.feed(TEXT[i:]) + r.flush() == EXPECTED, i


def test_char_by_char():
    r = StreamRedactor()
    assert "".join(r.feed(c) for c in TEXT) + r.flush() == EXPECTED


def test_partial_pii_at_end_of_stream_is_flushed_not_lost():
    r = StreamRedactor()
    assert r.feed("ends with john@exa") + r.flush() == "ends with john@exa"
    assert redact_text("tail 123-45-678") == "tail 123-45-678"


def test_failed_luhn_candidate_does_not_hide_an_email_inside_it():
    t = "4111 1111 1111 1112byzo@dap.com"          # card-shaped but Luhn-invalid, email starts inside it
    expected = f"4111 1111 1111 {R}"
    assert redact_text(t) == expected
    r = StreamRedactor()
    assert "".join(r.feed(ch) for ch in t) + r.flush() == expected


# ------------------------------------------- randomized split-invariance
def luhn_card(rng, n=16):
    body = [rng.randrange(10) for _ in range(n - 1)]
    body[0] = rng.choice([3, 4, 5, 6])
    for check in range(10):
        if luhn_ok("".join(map(str, body + [check]))):
            return "".join(map(str, body + [check]))


def gen_segment(rng):
    k = rng.random()
    if k < 0.30:
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(1, 9)))
    if k < 0.42:
        return rng.choice([" ", " ", "  ", "\n", "\t"])
    if k < 0.52:
        return rng.choice([".", ",", ";", ":", "!", "?", "(", ")", '"', "'", "-", "_", "@", "<", ">", "é", "/"])
    if k < 0.62:
        local = "".join(rng.choice("abcxyz.+_-0123456789") for _ in range(rng.randint(1, 12))).strip(".") or "u"
        dom = ".".join("".join(rng.choice("abcdefmnop") for _ in range(rng.randint(1, 7))) for _ in range(rng.randint(1, 3)))
        return f"{local}@{dom}.{rng.choice(['com', 'org', 'io', 'co'])}"
    if k < 0.70:
        sep = rng.choice("- ")
        return f"{rng.randint(1, 665):03d}{sep}{rng.randint(1, 99):02d}{sep}{rng.randint(1, 9999):04d}"
    if k < 0.80:
        c = luhn_card(rng, rng.choice([13, 15, 16, 16, 16, 19]))
        sep = rng.choice(["", " ", "-"])
        if sep and len(c) == 16:
            return sep.join(c[i:i + 4] for i in range(0, 16, 4))
        return c
    if k < 0.90:
        return "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 18)))  # numbers that are mostly not PII
    return rng.choice(["foo@bar", "@x", "a@b.c", "000-00-0000", "4111 1111 1111 1112"])


def oracle(text):
    """Independent one-shot implementation: scan the whole string left to right in one go.
    Failed-Luhn candidates are skipped one character at a time (they must not hide inner PII)."""
    out, pos, last = [], 0, 0
    while (m := _WITH_EMAIL.search(text, pos)):
        if m.group("card") is not None and not luhn_ok(re.sub(r"\D", "", m.group())):
            pos = m.start() + 1
            continue
        out += [text[last:m.start()], R]
        last = pos = m.end()
    out.append(text[last:])
    return "".join(out)


def random_splits(rng, text):
    mode = rng.random()
    if mode < 0.2:
        return list(text)  # token-by-character
    pieces, i = [], 0
    while i < len(text):
        step = rng.randint(1, 6) if mode < 0.7 else rng.randint(1, 40)
        pieces.append(text[i:i + step])
        i += step
    return pieces


def test_split_invariance_vs_oracle():
    rng = random.Random(1234)
    for case in range(4000):
        text = "".join(gen_segment(rng) for _ in range(rng.randint(1, 40)))
        want = oracle(text)
        r = StreamRedactor()
        got = []
        for piece in random_splits(rng, text):
            got.append(r.feed(piece))
            assert r.pending <= 2 * MAX_EMAIL_RUN
        got.append(r.flush())
        assert "".join(got) == want, f"case {case}: {text!r}\n got  {''.join(got)!r}\n want {want!r}"


def test_no_pii_ever_emitted_unredacted_in_any_chunk():
    rng = random.Random(99)
    for _ in range(1000):
        text = "".join(gen_segment(rng) for _ in range(30))
        r = StreamRedactor()
        emitted = "".join(r.feed(p) for p in random_splits(rng, text)) + r.flush()
        assert oracle(emitted) == emitted  # re-scanning the output finds nothing left to redact


# ---------------------------------------------------------- latency / memory
def test_no_added_latency_after_whitespace():
    r = StreamRedactor()
    assert r.feed("Hello ") == "Hello "            # delimiter seen -> nothing held
    assert r.feed("world, ") == "world, "
    assert r.pending == 0


def test_holds_back_at_most_the_current_word():
    r = StreamRedactor()
    assert r.feed("Hello wor") == "Hello "         # 'wor' could still become an email local-part
    assert r.pending == 3
    assert r.feed("ld ") == "world "


def test_memory_is_bounded_on_a_large_stream():
    rng = random.Random(5)
    words = ["hello", "world", "token", "stream", "a@b.com", "123-45-6789", "4111111111111111", "x" * 50]
    tracemalloc.start()
    r = StreamRedactor()
    total = max_pending = 0
    for _ in range(300_000):
        piece = rng.choice(words) + " "
        total += len(r.feed(piece))
        max_pending = max(max_pending, r.pending)
    total += len(r.flush())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert total > 3_000_000                      # >3MB of text went through
    assert max_pending <= MAX_EMAIL_RUN
    assert peak < 1_000_000                       # while peak heap stayed under 1MB


def test_pathological_long_run_does_not_blow_up_or_leak_memory():
    r = StreamRedactor()
    for _ in range(5000):
        r.feed("A" * 50)                          # 250KB with no delimiter at all
        assert r.pending <= MAX_EMAIL_RUN
    r.flush()
