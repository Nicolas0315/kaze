#!/usr/bin/env bash
# sustained-load soak: N waves of concurrent requests against a fresh server
set -u
cd .
export PATH=/usr/local/bin:/usr/bin:/bin
PORT=8097
lsof -ti:$PORT | xargs -r kill 2>/dev/null; sleep 2
python3 -m kaze serve --model /models/Qwen2.5-0.5B-Instruct \
  --kind int4 --port $PORT --host 127.0.0.1 > /tmp/soak3.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do curl -s http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 2; done
for W in 1 2 3 4 5 6 7 8; do
  printf "wave %d: " $W
  timeout 600 python3 bench/compare_kaze_server.py $PORT 64 32 64 2>&1 | tail -1
  printf "   gpu="; nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
printf "health: "; curl -s http://127.0.0.1:$PORT/health; echo
printf "worker errors in log: "; grep -ci "Traceback" /tmp/soak3.log
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo DONE