"""A/B the batch-1 row specialisation (engine bucket 1 + TDIM=1 kernels).

Interleaved in one process for the same reason as bench/cfg_ab.py: the GPU drifts, so
only alternating measurements of the two paths are comparable.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def one_trial(llm, gen_len: int) -> float:
    from kaze.sampler import SamplingParams
    eng = llm.engine
    ids = list(torch.randint(0, 30000, (64,)).tolist())
    seq = eng.add_request(ids, SamplingParams(max_tokens=gen_len, temperature=0.0))
    while seq.state != 3:
        eng.step()
    torch.cuda.synchronize()
    return 1.0 / (sum(seq.itls) / len(seq.itls)) if seq.itls else 0.0


def main() -> None:
    ap = argparse.ArgumentParser("rows_ab")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="int8")
    ap.add_argument("--lm-head-kind", default=None)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--gen-len", type=int, default=32)
    args = ap.parse_args()
    from kaze.llm import LLM

    lm_kind = args.lm_head_kind or args.kind
    llm = LLM(args.model, kind=args.kind, lm_head_kind=lm_kind, speculative=False,
              verbose=False)
    eng = llm.engine
    wide = [b for b in eng.buckets if b >= 16] or [16]
    modes = {"wide16": wide, "row1": [1] + wide}

    def activate(name: str) -> None:
        eng.buckets = modes[name]
        eng.graphs.clear()
        eng.static.clear()
        eng.capture_graphs()

    results: dict[str, list[float]] = {k: [] for k in modes}
    for name in modes:
        activate(name)
        one_trial(llm, 8)
    for _ in range(args.trials):
        for name in modes:
            activate(name)
            results[name].append(one_trial(llm, args.gen_len))
    print(f"{os.path.basename(args.model.rstrip(chr(47)))} {args.kind}/{lm_kind}")
    for name in modes:
        xs = results[name]
        print(f"  {name:7s} median {statistics.median(xs):7.1f} tok/s   "
              + " ".join(f"{x:.1f}" for x in xs))
    llm.shutdown()


if __name__ == "__main__":
    main()
