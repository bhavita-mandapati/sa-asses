"""Throughput + memory benchmark for the streaming redactor. Run: python3 bench.py"""
import random
import time
import tracemalloc

from redactor import StreamRedactor

rng = random.Random(1)
words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog,", "and", "emails",
         "john.doe@example.com", "or", "123-45-6789", "paid", "with", "4111 1111 1111 1111.", "Thanks!"]
text = " ".join(rng.choice(words) for _ in range(400_000))
tokens = []
i = 0
while i < len(text):                      # LLM-like tokens: 2-6 chars
    step = rng.randint(2, 6)
    tokens.append(text[i:i + step])
    i += step

def run(tokens):
    r = StreamRedactor()
    worst = max_pending = 0
    for tok in tokens:
        t0 = time.perf_counter()
        r.feed(tok)
        worst = max(worst, time.perf_counter() - t0)
        max_pending = max(max_pending, r.pending)
    r.flush()
    return worst, max_pending


# pass 1: speed (tracemalloc OFF - it slows execution several-fold and would distort timings)
t = time.perf_counter()
worst, max_pending = run(tokens)
dt = time.perf_counter() - t
print(f"input          : {len(text)/1e6:.1f} MB in {len(tokens):,} chunks")
print(f"throughput     : {len(text)/1e6/dt:.1f} MB/s   ({dt/len(tokens)*1e6:.1f} us/chunk avg, worst single chunk {worst*1e6:.0f} us)")
print(f"max held back  : {max_pending} chars (hard cap ~{320})")

# pass 2: memory (tracemalloc ON; token list was built before tracing, so this is the redactor's own heap use)
tracemalloc.start()
run(tokens)
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"peak heap      : {peak/1024:.0f} KiB while streaming {len(text)/1e6:.1f} MB through (redactor only)")
