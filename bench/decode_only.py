"""Decode-only measurement, small enough to run identically in any checkout.

Used to compare a change against the committed baseline: run it in a git worktree of
HEAD and in the working tree, alternating, because the GPU drifts.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main() -> None:
    ap = argparse.ArgumentParser("decode_only")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="int8")
    ap.add_argument("--lm-head-kind", default=None)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=64,
                    help="context length to decode at (prefill first, then measure)")
    args = ap.parse_args()
    from kaze.llm import LLM
    from kaze.sampler import SamplingParams

    lm_kind = args.lm_head_kind or args.kind
    llm = LLM(args.model, kind=args.kind, lm_head_kind=lm_kind, speculative=False,
              verbose=False)
    eng = llm.engine
    ids = list(torch.randint(0, 30000, (args.prompt_len,)).tolist())
    rates = []
    for _ in range(args.trials):
        seq = eng.add_request(ids, SamplingParams(max_tokens=args.gen_len, temperature=0.0))
        while seq.state != 3:
            eng.step()
        torch.cuda.synchronize()
        if seq.itls:
            rates.append(1.0 / (sum(seq.itls) / len(seq.itls)))
    print(f"{os.path.basename(args.model.rstrip(chr(47)))} {args.kind}/{lm_kind} "
          f"ctx={args.prompt_len} median {statistics.median(rates):7.1f} tok/s  "
          + " ".join(f"{r:.1f}" for r in rates))
    llm.shutdown()


if __name__ == "__main__":
    main()
