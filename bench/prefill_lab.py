"""Where does a 512-token prefill spend its GPU time?

Prefill is the axis where Kaze leads llama.cpp by the widest margin, so a regression here
is expensive.  This prints the kernel breakdown of one 512-token prefill plus the tok/s.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main() -> None:
    ap = argparse.ArgumentParser("prefill_lab")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="int4")
    ap.add_argument("--lm-head-kind", default=None)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()

    from kaze.llm import LLM
    from kaze.sampler import SamplingParams
    from torch.profiler import ProfilerActivity, profile

    lm_kind = args.lm_head_kind or args.kind
    llm = LLM(args.model, kind=args.kind, lm_head_kind=lm_kind, speculative=False,
              verbose=False)
    eng = llm.engine
    req = lambda: eng.add_request(list(torch.randint(0, 30000, (args.prompt,)).tolist()),
                                  SamplingParams(max_tokens=1, temperature=0.0))
    seq = req()
    while seq.state != 3:
        eng.step()
    torch.cuda.synchronize()

    times = []
    for _ in range(3):
        seq = req()
        t0 = time.perf_counter()
        while seq.state != 3:
            eng.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"{os.path.basename(args.model.rstrip(chr(47)))} {args.kind}/{lm_kind} "
          f"prefill {args.prompt / best:8.0f} tok/s  ({best * 1e3:.2f} ms)")

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        seq = req()
        while seq.state != 3:
            eng.step()
        torch.cuda.synchronize()
    rows = [(e.self_device_time_total, e.key, e.count) for e in prof.key_averages()
            if e.self_device_time_total > 0]
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"  total GPU {total:.0f} us")
    for us, key, count in rows[:args.top]:
        print(f"  {us:8.0f} us {100 * us / total:5.1f}%  x{count:<4d} {key[:70]}")
    llm.shutdown()


if __name__ == "__main__":
    main()
