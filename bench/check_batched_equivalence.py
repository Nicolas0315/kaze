import os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kaze.llm import LLM
from kaze.sampler import SamplingParams

llm = LLM("/models/Qwen2.5-0.5B-Instruct", kind="int4", lm_head_kind="int4", speculative=False, verbose=False)
eng = llm.engine
ids = eng.tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 16)[:128]

def burst(batched: bool):
    eng.batched_prefill = batched
    seqs = [eng.add_request(list(ids), SamplingParams(max_tokens=8, temperature=0.0)) for _ in range(16)]
    while any(s.state != 3 for s in seqs):
        eng.step()
    torch.cuda.synchronize()
    return [list(s.output) for s in seqs]

a = burst(False)
b = burst(True)
print("identical:", a == b)
print("first:", a[0][:8])
print("hash plain:", hashlib.sha1(str(a).encode()).hexdigest()[:12], "hash batched:", hashlib.sha1(str(b).encode()).hexdigest()[:12])
llm.shutdown()
