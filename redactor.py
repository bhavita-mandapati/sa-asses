"""Incremental, bounded-memory PII redactor for token streams.

The hard part of redacting a *stream* is that a secret can straddle chunk
boundaries ("123-45-" | "6789", or "john" | "@example.com"). Two naive options are
both bad: (a) buffer the whole response -> unbounded memory and TTFT = total latency;
(b) regex each chunk independently -> leaks anything that is split.

Approach: "hold back only what could still become a match".
  feed(chunk):
    buf = carry + chunk
    cut = start of the trailing text that is still a *possible prefix* of a match
          (pulled further left if a complete match straddles it)
    emit  redact(buf[:cut])     <- final: nothing arriving later can change it
    carry = buf[cut:]           <- bounded (~MAX_EMAIL_RUN chars)
  flush(): redact + emit the carry at end of stream.

Why `buf[:cut]` is final: every PII shape we detect is a contiguous run of a known
character class, and the cut is placed at the start of the trailing run of that class.
A match ending before the cut is followed by an already-seen non-class character, so
its lookahead/boundary is decided. A match that touches the end of the buffer starts
inside the held-back run by construction.

Typical hold-back is one word (until whitespace/punctuation arrives), so added latency
is about one token and it is never more than MAX_EMAIL_RUN characters.

`ctx` remembers the last already-emitted character so lookbehind boundaries like
(?<!\\d) stay correct across a cut (otherwise the carry would look like it starts a
fresh number and could be falsely matched or the boundary logic could be skipped).
"""
from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# --- patterns ---------------------------------------------------------------
# No left-anchor on purpose: if the local-part run is longer than 64 chars (not a valid address),
# the regex still redacts the last 64 chars + domain, so an email-shaped tail can never survive
# next to junk or a glued-on number. Over-redacting here is the safe direction.
_EMAIL = (
    r"(?P<email>[A-Za-z0-9._%+\-]{1,64}@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24})"
)
# US SSN: 3-2-4 with consistent '-' or ' ' separator; excludes invalid areas/groups/serials.
_SSN = (
    r"(?P<ssn>(?<!\d)(?!000|666|9\d\d)\d{3}(?P<ssep>[- ])(?!00)\d{2}(?P=ssep)(?!0000)\d{4}(?!\d))"
)
# Cards: 13-19 digits unspaced, 4-4-4-(1..7) grouped, or Amex-style 4-6-(4|5). Luhn-checked.
_CARD = (
    r"(?P<card>(?<!\d)(?:\d{13,19}"
    r"|\d{4}(?P<c1>[ -])\d{4}(?P=c1)\d{4}(?P=c1)\d{1,7}"
    r"|\d{4}(?P<c2>[ -])\d{6}(?P=c2)\d{4,5})(?!\d))"
)
_WITH_EMAIL = re.compile("|".join((_EMAIL, _SSN, _CARD)))
_NO_EMAIL = re.compile("|".join((_SSN, _CARD)))  # fast path when no '@' in the text

# --- hold-back geometry -----------------------------------------------------
# Longest email we match: 64 (local) + 1 + 253 (domain) = 318 -> 320 with slack.
MAX_EMAIL_RUN = 320
# Longest card/SSN span: 4-4-4-7 digits with 3 separators = 22 chars -> 24 with slack.
MAX_NUM_RUN = 24
_EMAIL_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-@")
_NUM_CHARS = frozenset("0123456789 -")


def luhn_ok(digits: str) -> bool:
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i & 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _hold_start(buf: str) -> int:
    """Index where the trailing, possibly-incomplete PII candidate begins (len(buf) if none)."""
    n = len(buf)
    i, lim = n, max(0, n - MAX_EMAIL_RUN)
    while i > lim and buf[i - 1] in _EMAIL_CHARS:
        i -= 1
    cut = i
    j, lim = n, max(0, n - MAX_NUM_RUN)
    while j > lim and buf[j - 1] in _NUM_CHARS:
        j -= 1
    while j < n and buf[j] in " -":  # a number candidate starts at a digit
        j += 1
    if j < n and j < cut:
        cut = j
    return cut


def _redact_upto(ctx: str, buf: str, cut: int, replacement: str) -> tuple[str, int]:
    """Redact buf[:cut]; return (emitted_text, final_cut).

    Matches are searched over the WHOLE buffer so lookaheads see real following text.
    A *complete* match that straddles `cut` (e.g. a card followed by '.', where the '.' makes
    the email hold-back start mid-number) pulls the cut left to the match start, so a secret
    is never split between "emitted" and "held back". `ctx` is visible to lookbehinds only.
    """
    text = ctx + buf
    off = len(ctx)
    cut_abs = off + cut
    pat = _WITH_EMAIL if "@" in buf else _NO_EMAIL
    out: list[str] = []
    last = pos = off
    while True:
        m = pat.search(text, pos)  # lookbehinds can see text before `pos`
        if m is None or m.start() >= cut_abs:
            break
        if m.group("card") is not None and not luhn_ok("".join(c for c in m.group() if c.isdigit())):
            # Looks like a card but isn't (order id, timestamp...). It must NOT consume its span:
            # a real email/SSN can start inside it, and swallowing it would make the result
            # depend on where chunk boundaries fall. Resume one char after the candidate start.
            pos = m.start() + 1
            continue
        if m.end() > cut_abs:
            cut_abs = m.start()
            break
        out.append(text[last:m.start()])
        out.append(replacement)
        last = pos = m.end()
    out.append(text[last:cut_abs])
    return "".join(out), cut_abs - off


class StreamRedactor:
    """One instance per independent text stream (e.g. per choice index)."""

    __slots__ = ("_carry", "_ctx", "_replacement")

    def __init__(self, replacement: str = REDACTED) -> None:
        self._carry = ""
        self._ctx = ""
        self._replacement = replacement

    @property
    def pending(self) -> int:
        """Characters currently held back (always <= MAX_EMAIL_RUN)."""
        return len(self._carry)

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        buf = self._carry + chunk
        return self._process(buf, _hold_start(buf))

    def flush(self) -> str:
        buf, self._carry = self._carry, ""
        return self._process(buf, len(buf)) if buf else ""

    def _process(self, buf: str, cut: int) -> str:
        out, cut = _redact_upto(self._ctx, buf, cut, self._replacement)
        if cut:
            self._ctx = buf[cut - 1]  # ORIGINAL char: lookbehinds must see real text, not the placeholder
        self._carry = buf[cut:]
        return out


def redact_text(text: str, replacement: str = REDACTED) -> str:
    """One-shot redaction of a complete string."""
    r = StreamRedactor(replacement)
    return r.feed(text) + r.flush()
