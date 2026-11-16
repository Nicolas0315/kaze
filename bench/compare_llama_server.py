"""Measure llama.cpp server aggregate throughput with N concurrent requests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import requests

BIN = "./bench/llamacpp/llama.cpp/build/bin/llama-server"


def run(model: str, n: int = 32, gen_len: int = 64, prompt_len: int = 64) -> dict:
    proc = subprocess.Popen(
        [BIN, "-m", model, "-ngl", "99", "-c", str(n * (prompt_len + gen_len + 128)), "-np", str(n),
         "--host", "127.0.0.1", "--port", "8199", "-t", "8"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                requests.get("http://127.0.0.1:8199/health", timeout=2).raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2)
        else:
            raise TimeoutError("llama.cpp server did not become healthy")
        prompt = ("Write a short paragraph about GPUs. " * (prompt_len // 8 + 2))[:prompt_len * 4]
        payload = {"prompt": prompt, "n_predict": gen_len, "temperature": 0.0,
                   "cache_prompt": os.environ.get("LLAMA_CACHE_PROMPT", "0") == "1", "ignore_eos": True}

        def one() -> int:
            r = requests.post("http://127.0.0.1:8199/completion", json=payload, timeout=600)
            r.raise_for_status()
            count = r.json().get("tokens_predicted")
            if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= gen_len:
                raise ValueError("missing or invalid tokens_predicted")
            return count

        one()                                  # warmup
        import concurrent.futures as cf
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            counts = list(ex.map(lambda _: one(), range(n)))
        dt = time.perf_counter() - t0
        return {"engine": "llama.cpp-server", "model": model.split("/")[-1].split("-Q4")[0],
                "concurrency": n, "agg_tok_s": sum(counts) / dt, "wall_s": dt}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    print("RESULT " + json.dumps(run(sys.argv[1], int(sys.argv[2]),                                        int(sys.argv[4]) if len(sys.argv) > 4 else 64,                                        int(sys.argv[3]) if len(sys.argv) > 3 else 64)))
