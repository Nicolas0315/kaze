import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kaze.llm import LLM
from kaze.sampler import SamplingParams

PROMPT = ("You are editing a Python file.\n\n"
          "def process(item):\n    name = item[\"name\"]\n    value = item[\"value\"]\n"
          "    if value > 0:\n        return {\"name\": name, \"value\": value}\n"
          "    return None\n\n# add process_batch the same way\n")

def run(spec: bool, n: int, gen: int = 32, model="Qwen2.5-0.5B-Instruct"):
    llm = LLM("/models/" + model, kind="int4",
              lm_head_kind="int4", speculative=spec, verbose=False)
    eng = llm.engine
    ids = eng.tokenizer.encode(PROMPT)
    best, outs = 0.0, None
    for _ in range(2):
        t0 = time.perf_counter()
        seqs = [eng.add_request(list(ids), SamplingParams(max_tokens=gen, temperature=0.0))
                for _ in range(n)]
        while any(s.state != 3 for s in seqs):
            eng.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        produced = sum(len(s.output) for s in seqs)
        if produced / dt > best:
            best = produced / dt
            outs = [list(s.output) for s in seqs]
    llm.shutdown()
    return best, outs

for model in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"):
    for n in (1, 8):
        p_rate, p_out = run(False, n, model=model)
        s_rate, s_out = run(True, n, model=model)
        print("%-8s streams=%d plain %7.1f tok/s | spec %7.1f tok/s (%.2fx) | same=%s" % (
              model.split("-")[1], n, p_rate, s_rate, s_rate / p_rate, p_out == s_out))
