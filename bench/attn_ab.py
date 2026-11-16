"""A/B the decode-attention split count at a long context, in one process.

Long-context decode attention is split-K over the KV blocks with grid
(batch, KV heads, splits).  On a 46-SM GPU a 2-KV-head model with the old fixed 8 splits
launches 16 CTAs, so only a third of the device participates.  This measures that.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main() -> None:
    ap = argparse.ArgumentParser("attn_ab")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="int8")
    ap.add_argument("--prompt", type=int, default=4096)
    ap.add_argument("--gen-len", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--splits", default="8,24")
    args = ap.parse_args()

    from kaze.llm import LLM
    from kaze.sampler import SamplingParams

    llm = LLM(args.model, kind=args.kind, speculative=False, verbose=False)
    model, eng = llm.model, llm.engine
    prompt = list(torch.randint(0, 30000, (args.prompt,)).tolist())
    variants = [int(s) for s in args.splits.split(",")]

    def activate(splits: int) -> None:
        model.num_splits = splits
        model.buffers.clear()
        eng.static.clear()
        eng.graphs.clear()
        eng.capture_graphs()

    def trial() -> float:
        seq = eng.add_request(prompt, SamplingParams(max_tokens=args.gen_len,
                                                    temperature=0.0))
        while seq.state != 3:
            eng.step()
        torch.cuda.synchronize()
        return 1.0 / (sum(seq.itls) / len(seq.itls)) if seq.itls else 0.0

    for s in variants:
        activate(s)
        trial()
    results: dict[int, list[float]] = {s: [] for s in variants}
    for _ in range(args.trials):
        for s in variants:
            activate(s)
            results[s].append(trial())
    print(f"{os.path.basename(args.model.rstrip(chr(47)))} ctx={args.prompt}")
    for s in variants:
        xs = results[s]
        print(f"  splits {s:3d} median {statistics.median(xs):7.1f} tok/s   "
              + " ".join(f"{x:.1f}" for x in xs))
    llm.shutdown()


if __name__ == "__main__":
    main()
