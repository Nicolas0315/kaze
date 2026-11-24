"""Kaze benchmark: prefill, decode, roofline efficiency, concurrency sweep.

Numbers reported:
  prefill_tok_s   prompt tokens per second for one request
  decode_tok_s    output tokens per second for one request (batch 1)
  tpot_ms         time per output token, batch 1
  agg_tok_s       aggregate output tokens per second over N concurrent requests
  bw_gbs          effective weight bandwidth = bytes streamed / time
                  (decode streams every weight byte once per token)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from kaze.engine import SamplingParams

PEAK_BW_GBS = 448.0     # RTX 3070 HBM, adjust for your GPU


def _tokens(n: int) -> list[int]:
    return list(torch.randint(0, 30000, (n,)).tolist())


def bench_prefill(llm, lengths=(128, 512, 2048), repeats=3) -> dict[str, float]:
    if repeats <= 0 or any(n <= 0 for n in lengths):
        raise ValueError("prefill lengths and repeats must be positive")
    out = {}
    eng = llm.engine
    for n in lengths:
        best = 1e9
        for _ in range(repeats):
            seq = eng.add_request(_tokens(n), SamplingParams(max_tokens=1, temperature=0.0))
            t0 = time.perf_counter()
            while seq.state != 3:
                eng.step()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        out[f"prefill_{n}_tok_s"] = n / best
        out[f"ttft_{n}_ms"] = best * 1e3
    return out


def bench_decode(llm, prompt_len=64, gen_len=128, repeats=3) -> dict[str, float]:
    if prompt_len <= 0 or gen_len < 2 or repeats <= 0:
        raise ValueError("decode requires a positive prompt, repeats, and at least 2 output tokens")
    eng = llm.engine
    nodec = []
    for _ in range(repeats):
        seq = eng.add_request(_tokens(prompt_len), SamplingParams(max_tokens=gen_len, temperature=0.0))
        while seq.state != 3:
            eng.step()
        torch.cuda.synchronize()
        nodec.append(seq)
    best = min(nodec, key=lambda x: sum(x.itls))
    itl = sum(best.itls) / max(len(best.itls), 1)
    weight_bytes = llm.model.weight_bytes()
    streaming = (weight_bytes - llm.model.embed.numel() * 2)   # embedding rows are not streamed
    return {
        "decode_tok_s": 1.0 / itl,
        "tpot_ms": itl * 1e3,
        "ttft_ms": (best.first_token_time - best.arrival) * 1e3,
        "stream_bytes_mb": streaming / 1e6,
        "bw_gbs": streaming / itl / 1e9,
        "bw_pct": 100.0 * streaming / itl / 1e9 / PEAK_BW_GBS,
    }


def bench_concurrency(llm, levels=(1, 2, 4, 8, 16, 32), prompt_len=128, gen_len=64) -> dict:
    if prompt_len <= 0 or gen_len <= 0 or any(n <= 0 for n in levels):
        raise ValueError("concurrency levels and token lengths must be positive")
    eng = llm.engine
    rows = {}
    for n in levels:
        for _ in range(1):                      # warmup
            pass
        t0 = time.perf_counter()
        seqs = [eng.add_request(_tokens(prompt_len),
                                SamplingParams(max_tokens=gen_len, temperature=0.0))
                for _ in range(n)]
        while any(s.state != 3 for s in seqs):
            eng.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total = sum(len(s.output) for s in seqs)
        rows[n] = {"agg_tok_s": total / dt, "total_tokens": total, "wall_s": dt,
                   "per_stream_tok_s": total / dt / n}
    return rows


def run_bench(llm, args) -> None:
    print("== prefill ==")
    pre = bench_prefill(llm, lengths=(args.prompt_len,), repeats=args.repeats)
    for k, v in pre.items():
        print(f"  {k:22s} {v:12.1f}")
    print("== decode (batch 1) ==")
    dec = bench_decode(llm, args.prompt_len, args.gen_len, args.repeats)
    for k, v in dec.items():
        print(f"  {k:22s} {v:12.2f}")
    print("== concurrency ==")
    con = bench_concurrency(llm, levels=(args.concurrency,),
                            prompt_len=args.prompt_len, gen_len=args.gen_len)
    for n, row in con.items():
        print(f"  n={n:3d}  agg={row['agg_tok_s']:8.1f} tok/s   "
              f"per-stream={row['per_stream_tok_s']:7.1f} tok/s   wall={row['wall_s']:.2f}s")
    out = {"prefill": pre, "decode": dec, "concurrency": con,
           "model_bytes": llm.model.weight_bytes(), "peak_bw_gbs": PEAK_BW_GBS}
    with open(Path(__file__).resolve().with_name("results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote bench/results.json")
