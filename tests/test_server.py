"""End-to-end server smoke test: start the OpenAI-compatible API and ask a question.

Covers the layer the engine tests cannot: config defaults, uvicorn wiring, JSON
formats, streaming, and shutdown.  Run: python3 tests/test_server.py
"""
from __future__ import annotations

import json
import subprocess
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, ".")

PORT = 8147
MODELS = os.environ.get("KAZE_MODELS", os.path.expanduser("~/models"))
MODEL = os.path.join(MODELS, "Qwen2.5-0.5B-Instruct")


def wait_health(timeout: float = 180.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                json.dumps(body).encode(),
                                {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "kaze", "serve", "--model", MODEL, "--kind", "int4",
         "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    fails = []
    try:
        if not wait_health():
            print("server did not become healthy")
            return 1
        chat = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 8, "temperature": 0.0})
        text = chat["choices"][0]["message"]["content"]
        ok = "Paris" in text
        print(f"chat          {text!r} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append("chat")
        comp = post("/v1/completions", {"model": "kaze", "prompt": "2 + 2 =",
                                          "max_tokens": 4, "temperature": 0.0})
        print(f"completions   {comp['choices'][0]['text']!r} ok")
        body = json.dumps({"model": "kaze", "prompt": "Count to three.",
                           "max_tokens": 8, "temperature": 0.0, "stream": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", body,
                                     {"Content-Type": "application/json"})
        chunks = 0
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                if line.strip().startswith(b"data:"):
                    chunks += 1
        ok = chunks > 1
        print(f"streaming     {chunks} chunks {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append("stream")
        # 異常系: 壊れた JSON は 400 で返り、サーバは落ちない
        bad = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/completions", b"{not json",
            {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(bad, timeout=15)
            code = 200
        except urllib.error.HTTPError as exc:
            code = exc.code
        ok = 400 <= code < 500     # FastAPI は壊れた JSON に 422 を返す
        print(f"bad json      {code} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append("bad json")

        # 異常系: 存在しないパスは 404
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/nope", timeout=15)
            code = 200
        except urllib.error.HTTPError as exc:
            code = exc.code
        ok = code == 404
        print(f"unknown path  {code} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append("unknown path")

        # 異常系: 過大な max_tokens でも固まらず、上限に丸めて返す
        try:
            out = post("/v1/completions", {"model": "kaze", "prompt": "hi",
                                           "max_tokens": 10**6})
            code = 200
            ok_body = isinstance(out.get("choices"), list)
        except urllib.error.HTTPError as exc:
            code = exc.code
            ok_body = True
        ok = (400 <= code < 500) or (code == 200 and ok_body)
        print(f"huge tokens   {code} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append("huge tokens")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    print()
    print("all ok" if not fails else "failed: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())