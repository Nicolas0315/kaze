#!/usr/bin/env bash
# One-shot verification: correctness, determinism, equivalence, quality, then a benchmark.
# Usage: bench/check_all.sh [--with-sanitizer]
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python3}
fail=0

step() { printf "\n=== %s ===\n" "$1"; }
check() {  # check <name> <command...>
  local name=$1; shift
  if timeout 2400 "$@" > /tmp/kaze_check.log 2>&1; then
    printf "  PASS  %s\n" "$name"
  else
    printf "  FAIL  %s\n" "$name"
    tail -15 /tmp/kaze_check.log | sed "s/^/        /"
    fail=$((fail + 1))
  fi
}

step "unit / integration (tests/)"
check "unittest discover" $PY -m unittest discover -s tests -p "test_*.py"
check "quantized smoke" $PY tests/test_smoke.py
check "random load stress" $PY tests/test_stress.py

step "equivalence / determinism"
check "epilogue fold + determinism" $PY bench/check_epilogue_fold.py
check "split-K fold determinism" env CUDA_LAUNCH_BLOCKING=1 $PY bench/check_epilogue_fold.py
check "batched attention vs single" $PY bench/check_attn_batch.py
check "batched prefill equivalence" $PY bench/check_batched_equivalence.py
check "speculative == plain" $PY bench/check_spec_equivalence.py

step "quality"
check "needle 4K" $PY bench/needle.py --model /models/Qwen2.5-1.5B-Instruct --ctx 4096

if [ "${1:-}" = "--with-sanitizer" ]; then
  step "memory checking (slow)"
  SAN=/usr/local/cuda/bin/compute-sanitizer
  if [ -x "$SAN" ]; then
    check "memcheck int4" $SAN --tool memcheck $PY bench/sanitize_probe.py int4
    check "memcheck int8" $SAN --tool memcheck $PY bench/sanitize_probe.py int8
  else
    printf "  SKIP  compute-sanitizer not found\n"
  fi
fi

step "benchmark (0.5B int4)"
$PY bench/compare.py '{"engine":"kaze","model":"/models/Qwen2.5-0.5B-Instruct","kind":"int4","lm_head_kind":"int4","speculative":false}' 2>/dev/null | grep RESULT

printf "\n%s\n" "----------------------------------------"
if [ "$fail" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
else
  echo "$fail CHECK(S) FAILED"
fi
exit "$fail"
