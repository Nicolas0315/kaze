"""Speculative decode through the HTTP server (single stream, repetitive prompt)."""
import json, sys, time
import requests
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 128
BASE = "http://127.0.0.1:%d" % PORT
KIND = sys.argv[3] if len(sys.argv) > 3 else "code"
REPEAT = """def process(item):\n    name = item["name"]\n    value = item["value"]\n    if value > 0:\n        return {"name": name, "value": value}\n    return None\n""" * 8
prompt = (REPEAT + "\n# continue the same pattern\ndef process2(item):\n"
          if KIND == "code" else "The history of the printing press begins in the fifteenth century, when ")
payload = {"model": "kaze", "prompt": prompt, "max_tokens": GEN,
           "temperature": 0.0, "ignore_eos": True}
requests.post(BASE + "/v1/completions", json=payload, timeout=600)  # warmup
best = 0.0
for _ in range(3):
    t0 = time.perf_counter()
    r = requests.post(BASE + "/v1/completions", json=payload, timeout=600).json()
    dt = time.perf_counter() - t0
    n = r["usage"]["completion_tokens"]
    best = max(best, n / dt)
print("server single-stream: %.1f tok/s (%d tokens)" % (best, GEN))