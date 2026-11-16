"""Re-measure the code-edit task (plain vs prompt-lookup speculation) on the current code."""
import sys, time
sys.path.insert(0, ".")
import torch
from kaze.llm import LLM
from kaze.sampler import SamplingParams

PROMPT = ("You are editing a Python file.\n\n"
          "def process(item):\n    name = item[\"name\"]\n    value = item[\"value\"]\n"
          "    if value > 0:\n        return {\"name\": name, \"value\": value}\n"
          "    return None\n\n# add process_batch the same way\n")
GEN = 128

for spec in (False, True):
    llm = LLM("/models/Qwen2.5-0.5B-Instruct", kind="int4",
              lm_head_kind="int4", speculative=spec, verbose=False, start_loop=False)
    eng = llm.engine
    ids = eng.tokenizer.encode(PROMPT)
    best = 0.0
    for _ in range(2):
        s = eng.add_request(list(ids), SamplingParams(max_tokens=GEN, temperature=0.0,
                                                     ignore_eos=True))
        while s.state != 3 and not s.output:
            eng.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n0 = len(s.output)
        while s.state != 3:
            eng.step()
        torch.cuda.synchronize()
        best = max(best, (len(s.output) - n0) / (time.perf_counter() - t0))
    st = eng.stats
    steps = st.get("spec_steps", 0)
    per = st.get("spec_tokens", 0) / steps if steps else 0
    print("spec=%-5s %.1f tok/s  (%.2f tokens/step over %d spec steps)" % (spec, best, per, steps))
    llm.shutdown()
