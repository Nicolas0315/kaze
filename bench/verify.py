"""One-shot verification: tests, stress, and the headline benchmarks with thresholds.

Run: python3 bench/verify.py        # ~8 minutes on a 3070
Exit code is non-zero when any test, child process, or threshold fails.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "models"
REGRESSIONS = ("test_regressions.py", "test_engine_regressions.py",
               "test_kernel_regressions.py", "test_loader_cli_regressions.py",
               "test_worker_failure.py", "test_entry_audit.py", "test_api_audit.py",
               "test_sampler_audit.py", "test_scheduler_audit.py", "test_weights_audit.py",
               "test_model_audit.py")

# (label, model, threshold decode, threshold prefill)
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


def _complete_count(out: str, suffix: str) -> bool:
    counts = [re.fullmatch(r"(\d+)/(\d+) " + re.escape(suffix), line.strip())
              for line in out.splitlines()]
    counts = [match for match in counts if match is not None]
    return bool(counts and int(counts[-1][1]) > 0 and counts[-1][1] == counts[-1][2])


def _metrics(out: str) -> dict:
    for line in reversed(out.splitlines()):
        line = line.removeprefix("RESULT ").strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _rate(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return 0.0


def main() -> int:
    bad: list[str] = []
    rc, out = run([sys.executable, "tests/test_kaze.py"])
    line = [l for l in out.splitlines() if "passed" in l]
    print(f"tests            {line[-1] if line else 'FAILED'}")
    if rc != 0 or not _complete_count(out, "passed"):
        bad.append("tests")
    for script in REGRESSIONS:
        rc, out = run([sys.executable, f"tests/{script}"])
        lines = out.strip().splitlines()
        ok = rc == 0 and bool(lines) and (lines[-1] == "OK" or lines[-1].startswith("OK ("))
        ok = ok and re.search(r"Ran [1-9]\d* tests? in ", out) is not None
        print(f"{script:32s} {'ok' if ok else 'FAILED'}")
        if not ok:
            bad.append(script)
    rc, out = run([sys.executable, "tests/test_stress.py", "12"])
    line = [l for l in out.splitlines() if "trials clean" in l]
    print(f"stress           {line[-1] if line else 'FAILED'}")
    if rc != 0 or not _complete_count(out, "trials clean"):
        bad.append("stress")
    rc, out = run([sys.executable, "tests/test_smoke.py"])
    line = [l for l in out.splitlines() if l.strip() == "all ok" or "wrong text" in l]
    print(f"smoke (量子化の組み合わせ)  {line[-1] if line else 'FAILED'}")
    if rc != 0 or not line or line[-1].strip() != "all ok":
        bad.append("smoke")
    rc, out = run([sys.executable, "tests/test_server.py"])
    line = [l for l in out.splitlines() if l.strip() == "all ok" or l.startswith("failed")]
    print(f"server (HTTP E2E)          {line[-1] if line else 'FAILED'}")
    if rc != 0 or not line or line[-1].strip() != "all ok":
        bad.append("server")
    for label, model, thr_dec, thr_pre in DECODE:
        cfg = json.dumps({"engine": "kaze", "model": f"{MODELS}/{model}", "kind": "int4",
                          "lm_head_kind": "int4", "speculative": False})
        rc, out = run([sys.executable, "bench/compare.py", cfg])
        metrics = _metrics(out)
        dec, pre = _rate(metrics, "decode_tok_s"), _rate(metrics, "prefill_tok_s")
        ok = rc == 0 and dec >= thr_dec and pre >= thr_pre
        print(f"{label:16s} decode {dec:7.1f} (>= {thr_dec:6.1f})  prefill {pre:9.1f} "
              f"(>= {thr_pre:8.1f})  {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(label)
    for task, thr in DRAFTERS:
        rc, out = run([sys.executable, "bench/drafter_worker.py", task, "ours (adaptive)"])
        tps = _rate(_metrics(out), "tok_s")
        ok = rc == 0 and tps >= thr
        print(f"drafter {task:12s} {tps:7.1f} (>= {thr:6.1f})  {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(task)
    print()
    print("all ok" if not bad else "failed: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
