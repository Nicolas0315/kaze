"""Measure kaze server aggregate throughput over HTTP at N concurrent requests.

Mirrors bench/compare_llama_server.py so the two servers can be compared honestly.
"""
import concurrent.futures as cf
import json
import sys
import time

import requests

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
PROMPT_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
GEN_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 64
PROMPT = ("Write a short paragraph about GPUs. " * (PROMPT_TOKENS // 8 + 2))[:PROMPT_TOKENS * 4]

BASE = "http://127.0.0.1:%d" % PORT
# ignore_eos matches bench/compare_llama_server.py, which passes ignore_eos=True: without
# it a model that emits EOS early generates fewer tokens (7B averaged 24 of 64) and the
# aggregate tok/s numbers are not comparable at all.
payload = {"model": "kaze", "prompt": PROMPT, "max_tokens": GEN_TOKENS,
           "temperature": 0.0, "ignore_eos": True}

for _ in range(90):
    try:
        requests.get(BASE + "/health", timeout=2)
        break
    except requests.RequestException:
        time.sleep(1)
else:
    raise SystemExit("server did not become healthy")

warm = requests.post(BASE + "/v1/completions", json=payload, timeout=600).json()
print("warmup completion_tokens:", warm["usage"]["completion_tokens"])

t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    futures = [ex.submit(lambda: requests.post(BASE + "/v1/completions", json=payload,
                                              timeout=900).json()) for _ in range(N)]
    outs = [f.result() for f in futures]
dt = time.perf_counter() - t0
tokens = sum(o["usage"]["completion_tokens"] for o in outs)
print("RESULT " + json.dumps({"engine": "kaze-server", "concurrency": N,
                              "agg_tok_s": tokens / dt, "wall_s": dt, "tokens": tokens}))
