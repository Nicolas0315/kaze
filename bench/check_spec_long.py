import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kaze.llm import LLM
from kaze.sampler import SamplingParams

REPEAT = """def process(item):
    name = item["name"]
    value = item["value"]
    if value > 0:
        return {"name": name, "value": value}
    return None
"""

def run(spec: bool, ctx: int, gen: int = 48) -> tuple[float, float]:
    llm = LLM("/models/Qwen2.5-0.5B-Instruct", kind="int4",
              lm_head_kind="int4", speculative=spec, spec_k=16, verbose=False)
    eng = llm.engine
    body = REPEAT * max(1, ctx // 12)
    ids = eng.tokenizer.encode(body)[:ctx]
    best = (0.0, 0.0)
    for _ in range(2):
        s = eng.add_request(list(ids), SamplingParams(max_tokens=gen, temperature=0.0))
        while s.state != 3 and not s.output:
            eng.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n0 = len(s.output)
        while s.state != 3:
            eng.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        produced = len(s.output) - n0
        rate = produced / dt
        if rate > best[0]:
            best = (rate, produced)
    llm.shutdown()
    return best

for ctx in (512, 1024, 4096):
    p, np_ = run(False, ctx)
    s, ns = run(True, ctx)
    print("ctx=%5d plain %7.1f tok/s   speculative %7.1f tok/s   (%.2fx)" % (
          ctx, p, s, s / p))
