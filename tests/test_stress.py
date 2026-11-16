"""Randomised end-to-end stress test for the shipped configuration.

Mixes batch sizes, prompt lengths, sampling parameters, speculation and CUDA graphs,
and checks that every request finishes, produces tokens, and respects max_tokens.
Run: python3 tests/test_stress.py [trials]
"""
from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer

from kaze.engine import DONE, Engine, SamplingParams
from kaze.loader import load_model

MODEL = "/models/Qwen2.5-0.5B-Instruct"
WORDS = "valley ledger engine kernel memory band width stream token model".split()


def main(trials: int = 12, seed: int = 1) -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_model(MODEL, kind="int4", group=128, lm_head_kind="int4", verbose=False)
    model.sparse_frac = 0.0
    rng = random.Random(seed)
    fails: list[tuple] = []
    for trial in range(trials):
        # every fourth trial runs a high-concurrency batch, where block allocation and
        # preemption get exercised instead of the single-stream paths
        big = rng.random() < 0.25
        batch = rng.choice([16, 32]) if big else rng.randint(1, 8)
        eng = Engine(
            model, max_batch=max(32 if big else 8, batch), use_cuda_graph=rng.random() < 0.5,
            speculative=rng.random() < 0.5, spec_k=rng.choice([4, 8, 16]),
            tokenizer=tok, verbose=False, num_blocks=2048 if big else 1024,
            # fp8v (K fp16 / V fp8) is the newest cache mode and must survive the same
            # randomised load, including prefix sharing and preemption
            cache_dtype=rng.choice(["fp16", "fp8", "fp8v", "fp8v"]),
        )
        if eng.use_cuda_graph:
            eng.capture_graphs()
        seqs = []
        for _ in range(batch):
            plen = rng.choice([1, 3, 17, 64]) if big else rng.choice([1, 3, 17, 64, 300, 1200])
            params = SamplingParams(
                max_tokens=rng.choice([1, 5, 17, 64]),
                temperature=rng.choice([0.0, 0.0, 0.7]),
                top_p=rng.choice([1.0, 0.9]),
                stop=rng.choice([(), (chr(10) * 2,), ("</s>",)]),
            )
            ids = tok(" ".join(rng.choice(WORDS) for _ in range(plen))).input_ids
            seqs.append((eng.add_request(list(ids), params), params))
        steps = 0
        while any(s.state != DONE for s, _ in seqs):
            eng.step()
            steps += 1
            if steps > 4000:
                fails.append(("no-progress", trial))
                break
        for s, params in seqs:
            if s.state != DONE:
                fails.append(("not-done", trial))
            if not s.output:
                fails.append(("empty", trial))
            if len(s.output) > params.max_tokens + 1:
                fails.append(("too-long", trial, len(s.output), params.max_tokens))
        del eng
        torch.cuda.empty_cache()
    print(f"{trials - len(fails)}/{trials} trials clean")
    for f in fails[:8]:
        print("  ", f)
    return 1 if fails else 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sys.exit(main(n))