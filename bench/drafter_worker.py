"""Measure one (task, config) pair in a fresh process: no GPU state carryover."""
import json
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "./bench")
from transformers import AutoTokenizer

from drafter_tasks import CONFIGS, TASKS
from kaze.engine import Engine, SamplingParams
from kaze.loader import load_model

MODEL = "/models/Qwen2.5-0.5B-Instruct"
KIND = "int8"
MAX_TOKENS = 96

task = sys.argv[1]
label = sys.argv[2]
spec_k = CONFIGS[label]

tok = AutoTokenizer.from_pretrained(MODEL)
model = load_model(MODEL, kind=KIND, group=128, lm_head_kind=KIND, verbose=False)
ids = tok(TASKS[task]).input_ids


def run(ids, n):
    eng = Engine(model, max_batch=2, use_cuda_graph=True, speculative=spec_k > 0, tokenizer=tok,
                 spec_k=max(spec_k, 2), verbose=False, num_blocks=1024)
    eng.capture_graphs()
    if spec_k > 0:
        eng.capture_spec_graph()
    seq = eng.add_request(list(ids), SamplingParams(max_tokens=n, temperature=0.0))
    t0 = time.perf_counter()
    while seq.state != 3:
        eng.step()
    dt = time.perf_counter() - t0
    steps = eng.stats["spec_steps"]
    per = (eng.stats["spec_tokens"] / steps) if steps else 0.0
    return len(seq.output) / dt, per


run(ids, 16)          # warm up clocks and caches
best, per = 0.0, 0.0
for _ in range(2):
    tps, p = run(ids, MAX_TOKENS)
    if tps > best:
        best, per = tps, p
print(json.dumps({"task": task, "config": label, "tok_s": round(best, 1),
                  "tokens_per_step": round(per, 2)}))
