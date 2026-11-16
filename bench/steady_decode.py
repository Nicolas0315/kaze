"""Steady-state decode throughput at N concurrent streams.

bench/run.py and bench/compare.py time the whole request (prefill + decode), which is what
a user feels but hides the decode number: at 32 streams a 128-token prompt costs 0.36 s of
prefill and only 0.14 s of decode.  This isolates the decode phase, so "N-way decode tok/s"
is measurable and comparable with llama.cpp server -np N steady state.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main() -> None:
    ap = argparse.ArgumentParser("steady_decode")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="int4")
    ap.add_argument("--lm-head-kind", default=None)
    ap.add_argument("--levels", default="8,32,64")
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=64)
    args = ap.parse_args()

    from kaze.llm import LLM
    from kaze.sampler import SamplingParams

    lm_kind = args.lm_head_kind or args.kind
    llm = LLM(args.model, kind=args.kind, lm_head_kind=lm_kind, speculative=False,
              verbose=False)
    eng = llm.engine
    print(f"{os.path.basename(args.model.rstrip(chr(47)))} {args.kind} "
          f"prompt={args.prompt} gen={args.gen_len}")
    for n in (int(v) for v in args.levels.split(",")):
        ids = list(torch.randint(0, 30000, (args.prompt,)).tolist())
        seqs = [eng.add_request(ids, SamplingParams(max_tokens=args.gen_len + 1,
                                                       temperature=0.0)) for _ in range(n)]
        # run until every stream has finished its prefill and produced one token
        while any(s.state != 3 and not s.output for s in seqs):
            eng.step()
        torch.cuda.synchronize()
        # the warmup loop above already appended one token per stream, so counting the
        # absolute output length here over-reports the rate by (gen+1)/gen -- at gen=24
        # that was a silent 4%, and worse when the warmup ran longer than one token.
        before = sum(len(s.output) for s in seqs)
        t0 = time.perf_counter()
        steps = 0
        while any(s.state != 3 for s in seqs):
            eng.step()
            steps += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        produced = sum(len(s.output) for s in seqs) - before
        ms_per_step = dt / max(steps, 1) * 1e3
        print(f"  n={n:3d}  decode-only {produced / dt:8.1f} tok/s  "
              f"({ms_per_step:5.2f} ms/step, {steps} steps, {produced} tokens)")
    llm.shutdown()


if __name__ == "__main__":
    main()
