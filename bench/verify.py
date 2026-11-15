"""One-shot verification: tests, stress, and the headline benchmarks with thresholds.

Run: python3 bench/verify.py        # ~8 minutes on a 3070
Exit code is non-zero when any threshold fails, so it can gate a change.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("KAZE_MODELS", os.path.expanduser("~/models"))

# (label, model, kind, threshold decode, threshold prefill)
DECODE = [
    ("0.5B", "Qwen2.5-0.5B-Instruct", 550.0, 30000.0),
    ("1.5B", "Qwen2.5-1.5B-Instruct", 250.0, 11000.0),
    ("3B", "Qwen2.5-3B-Instruct", 150.0, 6000.0),
]
DRAFTERS = [("code edit", 1200.0), ("jsonl edit", 1000.0), ("long jsonl", 700.0),
            ("story", 430.0)]


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    bad: list[str] = []
    rc, out = run([sys.executable, "tests/test_kaze.py"])
    line = [l for l in out.splitlines() if "passed" in l]
    print(f"tests            {line[-1] if line else 'FAILED'}")
    if rc != 0:
        bad.append("tests")
    rc, out = run([sys.executable, "tests/test_stress.py", "12"])
    line = [l for l in out.splitlines() if "trials clean" in l]
    print(f"stress           {line[-1] if line else 'FAILED'}")
    if rc != 0:
        bad.append("stress")
    rc, out = run([sys.executable, "tests/test_smoke.py"])
    line = [l for l in out.splitlines() if l.strip() in ("all ok",) or "wrong text" in l]
    print(f"smoke (量子化の組み合わせ)  {line[-1] if line else 'FAILED'}")
    if rc != 0:
        bad.append("smoke")
    rc, out = run([sys.executable, "tests/test_qwen38.py"])
    rc2, out2 = run([sys.executable, "tests/test_qwen38.py", "--only", "graph"])
    gline = [l for l in out2.splitlines() if l.startswith("graph:")]
    print(f"qwen3.8 CUDA Graph        {gline[-1] if gline else "FAILED"}")
    if rc2 != 0:
        bad.append("qwen3.8 graph")
    line = [l for l in out.splitlines() if l.strip() in ("ok", "FAIL")]
    print(f"qwen3.8 (Qwen3_5)          {line[-1] if line else "FAILED"}")
    if rc != 0:
        bad.append("qwen3.8")
    rc, out = run([sys.executable, "tests/test_llm.py"])
    line = [l for l in out.splitlines() if "passed" in l]
    print(f"llm (Python API)           {line[-1] if line else 'FAILED'}")
    if rc != 0:
        bad.append("llm")
    rc, out = run([sys.executable, "tests/test_server.py"])
    line = [l for l in out.splitlines() if l.strip() == "all ok" or l.startswith("failed")]
    print(f"server (HTTP E2E)          {line[-1] if line else 'FAILED'}")
    if rc != 0:
        bad.append("server")
    for label, model, thr_dec, thr_pre in DECODE:
        cfg = json.dumps({"engine": "kaze", "model": f"{MODELS}/{model}", "kind": "int4",
                          "lm_head_kind": "int4", "speculative": False})
        rc, out = run([sys.executable, "bench/compare.py", cfg])
        dec = pre = 0.0
        for tok in out.replace(",", " ").split():
            pass
        try:
            d = json.loads(out[out.index("{"):out.rindex("}") + 1])
            dec, pre = d.get("decode_tok_s", 0.0), d.get("prefill_tok_s", 0.0)
        except Exception:
            pass
        ok = dec >= thr_dec and pre >= thr_pre
        print(f"{label:16s} decode {dec:7.1f} (>= {thr_dec:6.1f})  prefill {pre:9.1f} "
              f"(>= {thr_pre:8.1f})  {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(label)
    for task, thr in DRAFTERS:
        rc, out = run([sys.executable, "bench/drafter_worker.py", task, "ours (adaptive)"])
        tps = 0.0
        for line in out.splitlines():
            if line.startswith("{"):
                tps = json.loads(line).get("tok_s", 0.0)
        ok = tps >= thr
        print(f"drafter {task:12s} {tps:7.1f} (>= {thr:6.1f})  {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(task)
    print()
    print("all ok" if not bad else "failed: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())