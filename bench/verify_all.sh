#!/usr/bin/env bash
# Full verification: correctness, robustness under load, and the headline benchmarks.
# Usage: bench/verify_all.sh [--long]   (--long adds the 30-minute soak)
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/bin:/usr/bin:/bin
fail=0
run() {  # run <name> <cmd...>
  local name=$1; shift
  printf "== %s\n" "$name"
  if "$@" > /tmp/va.log 2>&1; then printf "   PASS\n"; else printf "   FAIL\n"; tail -6 /tmp/va.log | sed "s/^/     /"; fail=$((fail+1)); fi
}

printf "== disk\n"; df -h / | tail -1
printf "== correctness suite\n"
if ./bench/check_all.sh > /tmp/va_all.log 2>&1; then grep -E "ALL CHECKS" /tmp/va_all.log | sed "s/^/   /"; else fail=$((fail+1)); tail -8 /tmp/va_all.log | sed "s/^/   /"; fi

printf "== needle ladder (0.5B)\n"
for C in 2048 4096 8192 16384; do
  out=$(timeout 1200 python3 bench/needle.py --model /models/Qwen2.5-0.5B-Instruct --ctx $C 2>/dev/null | tail -1)
  case "$out" in *HIT*) printf "   %s\n" "$out";; *) printf "   %s  <-- FAIL\n" "$out"; fail=$((fail+1));; esac
done

printf "== KV pressure (16 concurrent, 64-block pool)\n"
if ./bench/kv_pressure.sh > /tmp/va_kv.log 2>&1 && grep -q "status codes: \[200\]" /tmp/va_kv.log; then printf "   PASS (all 200)\n"; else printf "   FAIL\n"; tail -5 /tmp/va_kv.log | sed "s/^/     /"; fail=$((fail+1)); fi

printf "== server soak (8 waves x 64 streams)\n"
if ./bench/soak_server.sh > /tmp/va_soak.log 2>&1; then
  grep -c "RESULT" /tmp/va_soak.log | sed "s/^/   waves: /"
  grep -q "worker errors in log: 0" /tmp/va_soak.log && printf "   worker errors: 0\n" || { printf "   worker errors > 0 <-- FAIL\n"; fail=$((fail+1)); }
else printf "   FAIL\n"; fail=$((fail+1)); fi

printf "\n----------------------------------------\n"
if [ "$fail" -eq 0 ]; then echo "ALL VERIFICATION PASSED"; else echo "$fail CHECK(S) FAILED"; fi
exit "$fail"