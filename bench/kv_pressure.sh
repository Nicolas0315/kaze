#!/usr/bin/env bash
# KV pressure: more concurrent long requests than the pool can hold.
set -u
cd .
export PATH=/usr/local/bin:/usr/bin:/bin
PORT=8096
lsof -ti:$PORT | xargs -r kill 2>/dev/null; sleep 2
python3 -m kaze serve --model /models/Qwen2.5-0.5B-Instruct \
  --kind int4 --num-blocks 64 --port $PORT --host 127.0.0.1 > /tmp/kv.txt 2>&1 &
SRV=$!
for i in $(seq 1 60); do curl -s http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 2; done
python3 - <<PY
import concurrent.futures as cf, requests
url = "http://127.0.0.1:$PORT/v1/completions"
payload = {"model": "kaze", "prompt": "The history of computing " * 40,
           "max_tokens": 32, "temperature": 0.0, "ignore_eos": True}
def one(_):
    r = requests.post(url, json=payload, timeout=600)
    return r.status_code, (r.json().get("usage") or {}).get("completion_tokens")
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    outs = list(ex.map(one, range(16)))
print("status codes:", sorted({o[0] for o in outs}))
print("completed tokens:", [o[1] for o in outs])
PY
printf "health after: "; curl -s http://127.0.0.1:$PORT/health; echo
printf "tracebacks: "; grep -ci Traceback /tmp/kv.txt
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo DONE