"""End-to-end A/B of the kernel config tables (legacy vs tuned) inside one process.

Two measurement facts drove this design:
  * in-graph kernel time drifts ~10-15% within ~30 s (bench/variance_check.py),
  * a decode benchmark on a warm GPU came out 23% slower than the same work cold,
so absolute numbers cannot be compared across runs.  Instead the two tables are
interleaved at fine granularity (mode A / mode B / mode A / ...) and only the median of
each mode is reported, which cancels slow drift.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def one_trial(llm, prompt_len: int, gen_len: int) -> float:
    from kaze.sampler import SamplingParams
    eng = llm.engine
    ids = list(torch.randint(0, 30000, (prompt_len,)).tolist())
    seq = eng.add_request(ids, SamplingParams(max_tokens=gen_len, temperature=0.0))
    while seq.state != 3:
        eng.step()
    torch.cuda.synchronize()
    return 1.0 / (sum(seq.itls) / len(seq.itls)) if seq.itls else 0.0


def main() -> None:
    ap = argparse.ArgumentParser("cfg_ab")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="int8")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lm-head-kind", default="int8")
    ap.add_argument("--modes", default="legacy,tuned")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--gen-len", type=int, default=32)
    args = ap.parse_args()

    from kaze.kernels import quant_linear as K
    from kaze.llm import LLM

    llm = LLM(args.model, kind=args.kind, group=args.group, lm_head_kind=args.lm_head_kind,
              speculative=False, start_loop=False, verbose=False)
    eng = llm.engine
    modes = args.modes.split(",")
    graph_cache: dict[str, int] = {}

    def activate(mode: str) -> None:
        if mode.startswith("splitk="):
            K.set_cfg_mode("tuned")
            K._SPLITK_MAX = int(mode.split("=")[1])
        else:
            K._SPLITK_MAX = 8
            K.set_cfg_mode(mode)
        eng.graphs.clear()
        eng.static.clear()
        eng.capture_graphs()

    results: dict[str, list[float]] = {m: [] for m in modes}
    for mode in modes:                                # one warm-up trial per mode
        activate(mode)
        one_trial(llm, 64, 8)
    for t in range(args.trials):
        for mode in modes:
            activate(mode)
            results[mode].append(one_trial(llm, 64, args.gen_len))
    name = os.path.basename(args.model.rstrip(chr(47)))
    print(f"{name} {args.kind} lm={args.lm_head_kind}")
    for mode in modes:
        xs = results[mode]
        print(f"  {mode:7s} median {statistics.median(xs):7.1f} tok/s   "
              + " ".join(f"{x:.1f}" for x in xs))
    llm.shutdown()


if __name__ == "__main__":
    main()
