"""Final sweep: every shipped model loads, generates sanely, and is deterministic."""
import sys, time
sys.path.insert(0, ".")
import torch
from kaze.llm import LLM
from kaze.sampler import SamplingParams

MODELS = [
    ("Qwen2.5-0.5B-Instruct", "int4", 128, "int4"),
    ("Qwen2.5-1.5B-Instruct", "int4", 128, "int4"),
    ("Qwen2.5-3B-Instruct", "int4", 128, "int4"),
    ("Nous-Capybara-7B-V1.9", "int4", 128, "int4"),
    ("Qwen3-0.6B", "int4", 128, "int4"),
    ("tiny-qwen3moe", "int4", 128, "int4"),
]
PROMPT = "The history of computing begins with"

for name, kind, group, head in MODELS:
    path = "/models/" + name
    t0 = time.time()
    try:
        llm = LLM(path, kind=kind, group=group, lm_head_kind=head, speculative=False,
                  verbose=False, start_loop=False)
    except Exception as e:
        print("%-26s LOAD FAILED: %r" % (name, e))
        continue
    eng = llm.engine
    ids = eng.tokenizer.encode(PROMPT)
    outs = []
    for _ in range(2):
        s = eng.add_request(list(ids), SamplingParams(max_tokens=24, temperature=0.0,
                                                     ignore_eos=True))
        while s.state != 3:
            eng.step()
        torch.cuda.synchronize()
        outs.append(list(s.output))
    text = eng.tokenizer.decode(outs[0])[:48].replace("\n", " ")
    mem = torch.cuda.memory_allocated() / 1e9
    print("%-26s %5.1fs  det=%s  mem=%.2fGB  %r" % (
        name, time.time() - t0, outs[0] == outs[1], mem, text))
    del llm, eng
    torch.cuda.empty_cache()
