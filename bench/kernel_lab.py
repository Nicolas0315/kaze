"""Kernel lab: per-kernel profile of a decode step, and GEMV/attention microbenchmarks.

Why this exists: docs/mixed.md records that an isolated microbenchmark win often fails
to transfer to the engine (single-stream throughput != latency inside a dependent step).
So this tool measures both, and reports every number against the HBM roofline:

    ideal_us = bytes_streamed / PEAK_BW

Modes:
    profile  one decode step, aggregated by kernel (torch.profiler), plus roofline
    micro    per-shape GEMV timing (latency + throughput) with config sweeps
    attn     decode attention across context lengths

Examples:
    python3 bench/kernel_lab.py profile --model /models/Qwen2.5-3B-Instruct
    python3 bench/kernel_lab.py micro   --model /models/Qwen2.5-1.5B-Instruct --sweep
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from kaze.config import ModelConfig

PEAK_BW_GBS = 448.0


# --------------------------------------------------------------------- helpers
def shapes(cfg: ModelConfig, kind: str = "int8", group: int = 128) -> dict[str, tuple[int, int, str]]:
    """(n, k, kind) for every GEMV in one decoder layer, plus the head."""
    n_kv = cfg.num_key_value_heads * cfg.head_dim
    q = cfg.num_attention_heads * cfg.head_dim
    return {
        "qkv": (q + 2 * n_kv, cfg.hidden_size, kind),
        "o_proj": (cfg.hidden_size, q, kind),
        "gate_up": (2 * cfg.intermediate_size, cfg.hidden_size, kind),
        "down_proj": (cfg.hidden_size, cfg.intermediate_size, kind),
        "lm_head": (cfg.vocab_size, cfg.hidden_size, "int8"),
    }


def weight_bytes(lin) -> int:
    b = lin.packed.numel() * lin.packed.element_size()
    if lin.scales is not None:
        b += lin.scales.numel() * lin.scales.element_size()
    return b


def make_linear(n: int, k: int, kind: str, group: int = 128, device: str = "cuda"):
    from kaze.quant import QuantLinear
    if kind == "int8":
        packed = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device)
        scales = (torch.rand(n, k // group, dtype=torch.float16, device=device) * 0.01 + 1e-3)
    else:
        packed = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device=device)
        scales = (torch.rand(n, k // group, dtype=torch.float16, device=device) * 0.01 + 1e-3)
    bias = torch.rand(n, dtype=torch.float16, device=device) * 0.1
    return QuantLinear(packed, scales, bias, n, k, group, kind)


def time_latency(fn, iters: int = 40, warmup: int = 10) -> tuple[float, float]:
    """Median/min per-launch microseconds using in-stream events (no host sync per call).

    This includes the Python/Triton launch cost (~8-15 us), so it is only meaningful for
    kernels slower than that.  Use time_graph for the in-graph kernel time.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    xs = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    return xs[len(xs) // 2], xs[0]


def time_graph(fn, reps: int = 32, replays: int = 3, warmup: int = 3) -> float:
    """Mean GPU microseconds per launch with the kernel captured in a CUDA graph.

    Decode runs inside a graph, so this is the number that predicts engine speed: the
    host-side launch cost disappears and successive launches are serialised by the
    write-after-read hazard on the output buffer, the same way a real step behaves.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(replays):
        g.replay()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1e3 / replays / reps


def time_throughput(fn, iters: int = 100, warmup: int = 20) -> float:
    """Microseconds per launch when launches are back to back (no sync in the loop)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# --------------------------------------------------------------------- profile
def do_profile(args) -> None:
    from kaze.llm import LLM
    from kaze.sampler import SamplingParams

    llm = LLM(args.model, kind=args.kind, group=args.group, lm_head_kind=args.lm_head_kind,
              speculative=False, start_loop=False, verbose=False)
    eng = llm.engine
    ids = list(torch.randint(0, 30000, (args.prompt,)).tolist())
    n = max(1, args.streams)
    seqs = [eng.add_request(list(ids), SamplingParams(max_tokens=args.steps + 8,
                                                      temperature=0.0)) for _ in range(n)]
    # prefill first, then profile only steady-state decode
    while any(s.state != 3 and not s.output for s in seqs):
        eng.step()
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(args.steps):
            eng.step()
        torch.cuda.synchronize()

    rows = []
    for evt in prof.key_averages():
        us = getattr(evt, "self_device_time_total", 0.0)
        if us <= 0:
            continue
        rows.append((us, evt.key, evt.count))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    streams = max(1, args.streams)
    print(f"# {os.path.basename(args.model.rstrip('/'))} kind={args.kind} "
          f"lm_head={args.lm_head_kind} steps={args.steps} streams={streams} "
          f"buckets={eng.buckets}")
    print(f"# total GPU time {total / args.steps:8.1f} us/step "
          f"({streams * args.steps * 1e6 / total:.1f} tok/s aggregate if decode-bound)")
    print(f"{'us/step':>9} {'%':>5} {'count':>6}  kernel")
    for us, key, count in rows[:args.top]:
        print(f"{us / args.steps:9.2f} {100 * us / total:5.1f} {count / args.steps:6.1f}  {key[:78]}")


# ----------------------------------------------------------------------- micro
def sweep_variants(n: int, k: int, kind: str) -> list[tuple[int, int, int, int, int]]:
    """(block_n, split_k, num_warps, num_stages, block_k) candidates that can plausibly win.

    block_k is a dimension of its own: 64 gives two dots per 128-wide scale group, so the
    pipeline has more, smaller stages to overlap; 128 keeps one scale per iteration.
    """
    out = []
    bks = tuple(b for b in (64, 128) if b <= k)
    if n >= 4096:                                  # wide: the device is already full
        for bk in bks:
            for bn in (32, 64, 128, 256):
                for nw in (2, 4, 8):
                    for ns in (2, 3, 4):
                        out.append((bn, 1, nw, ns, bk))
        return out
    for bk in bks:
        for bn in (16, 32, 64, 128):
            for nw in (1, 2, 4, 8):
                if nw * 32 > bn * 32:              # more warps than output rows: skip
                    continue
                for ns in (2, 3, 4, 6):
                    out.append((bn, 1, nw, ns, bk))
        for bn in (16, 32, 64, 128):
            for sk in (2, 4, 8):
                for nw in (2, 4):
                    for ns in (2, 3):
                        out.append((bn, sk, nw, ns, bk))
    return out


def do_micro(args) -> None:
    from kaze.kernels import quant_linear as K

    cfg = ModelConfig.from_pretrained(args.model)
    shp = shapes(cfg, args.kind, args.group)
    x = (torch.rand(16, cfg.hidden_size, dtype=torch.float16, device="cuda") - 0.5)
    if args.only:
        shp = {k: v for k, v in shp.items() if k in args.only.split(",")}

    print(f"# {os.path.basename(args.model.rstrip('/'))} kind={args.kind} group={args.group} "
          f"peak={PEAK_BW_GBS:.0f} GB/s")
    print(f"{'shape':10s} {'n':>7} {'k':>6} {'cfg':>14} {'graph_us':>9} {'launch_us':>9} "
          f"{'GB/s':>7} {'%peak':>6} {'ideal_us':>8}")
    for name, (n, k, kind) in shp.items():
        lin = make_linear(n, k, kind, args.group)
        xk = x[:, :k].contiguous()
        out = torch.empty((16, n), dtype=torch.float16, device="cuda")
        nbytes = weight_bytes(lin)
        ideal = nbytes / (PEAK_BW_GBS * 1e9) * 1e6

        cfg_now = K._cfg(n, k, kind)
        variants = sweep_variants(n, k, kind) if args.sweep else [cfg_now]

        rows = []
        tdim = args.tdim
        is_swiglu = name == "gate_up"
        if is_swiglu:
            variants = ([(bn, 1, nw, ns, min(args.group, k))
                         for bn in (16, 32, 64, 128)
                         for nw in (2, 4, 8)
                         for ns in (2, 3, 4)] if args.sweep else [cfg_now])
        for bn, sk, nw, ns, bk in variants:
            acc = (torch.zeros((sk, tdim, n), dtype=torch.float32, device="cuda")
                   if sk > 1 else None)

            def run(bn=bn, sk=sk, nw=nw, ns=ns, bk=bk, acc=acc):
                if is_swiglu:
                    _run_swiglu(K, xk, lin, out, bn, nw, ns, tdim)
                else:
                    _run_gemv(K, xk, lin, out, bn, sk, nw, ns, acc, bk, tdim)
            try:
                us = time_graph(run, reps=args.reps)
            except Exception as exc:                     # unsupported config: skip
                if args.all:
                    print(f"  skip {name} {bn}/{sk}/{nw}/{ns}/{bk}: {type(exc).__name__}: {exc}")
                continue
            rows.append((us, (bn, sk, nw, ns, bk)))
            if acc is not None:
                del acc
        if not rows:
            print(f"{name:10s} {n:7d} {k:6d}  (no variant compiled)")
            continue
        rows.sort()
        best_us, best_tag = rows[0]
        cur_us = next((us for us, tag in rows if tag == tuple(cfg_now)), None)
        if is_swiglu:
            launch = time_latency(lambda: _run_swiglu(K, xk, lin, out, *cfg_now[:3], tdim),
                                  iters=20)[0]
        else:
            launch = time_latency(lambda: _run_gemv(K, xk, lin, out, *cfg_now[:4], None,
                                                   cfg_now[4], tdim), iters=20)[0]
        tag = "/".join(str(t) for t in best_tag)
        cur = "/".join(str(t) for t in cfg_now)
        print(f"{name:10s} {n:7d} {k:6d} {tag:>14} {best_us:9.1f} {launch:9.1f} "
              f"{nbytes / (best_us * 1e-6) / 1e9:7.1f} "
              f"{100 * nbytes / (best_us * 1e-6) / 1e9 / PEAK_BW_GBS:5.1f}% {ideal:8.1f}"
              + ("" if best_tag == tuple(cfg_now) else
                 (f"   (cfg {cur}: {cur_us:.1f} us, gain {100 * (cur_us - best_us) / cur_us:+.1f}%)"
                  if cur_us is not None else f"   (cfg {cur}: not measured)")))
        if args.sweep and args.all:
            for us, tagv in rows[:12]:
                print(f"    {us:8.1f} us  {100 * nbytes / (us * 1e-6) / 1e9 / PEAK_BW_GBS:5.1f}%  "
                      + "/".join(str(t) for t in tagv))
        del lin
        torch.cuda.empty_cache()


def _run_swiglu(K, x, lin, out, bn, nw, ns, tdim=16):
    """gate_up matvec/GEMM that writes silu(gate) * up (the engine's fused path)."""
    nh = lin.n // 2
    block_k = min(lin.group, lin.k)
    grid = (triton_cdiv(nh, bn),)
    if lin.kind == "int8":
        K.gemv_int8_swiglu_kernel[grid](x, lin.packed, lin.scales, out, nh, lin.k,
                                        BLOCK_N=bn, BLOCK_K=block_k, GROUP=lin.group,
                                        TDIM=tdim, num_warps=nw, num_stages=ns)
    else:
        K.gemv_int4_swiglu_kernel[grid](x, lin.packed.view(torch.int8), lin.scales, out,
                                        nh, lin.k, BLOCK_N=bn, BLOCK_K=block_k,
                                        GROUP=lin.group, TDIM=tdim, num_warps=nw,
                                        num_stages=ns)


def _run_gemv(K, x, lin, out, bn, sk, nw, ns, acc=None, bk=None, tdim=16):
    """Like gemv() but with every knob forced (gemv() clamps split_k for some kinds)."""
    n, k = lin.n, lin.k
    block_k = min(bk or lin.group, lin.group, k)
    if sk > 1:
        if acc is None:
            acc = torch.zeros((sk, 16, n), dtype=torch.float32, device="cuda")
        kern = K.gemv_int4_splitk_kernel if lin.kind == "int4" else K.gemv_int8_splitk_kernel
        wptr = lin.packed.view(torch.int8) if lin.kind == "int4" else lin.packed
        kern[(triton_cdiv(n, bn), sk)](x, wptr, lin.scales, acc, n, k, BLOCK_N=bn,
                                       BLOCK_K=block_k, GROUP=lin.group, TDIM=tdim, SPLIT_K=sk,
                                       num_warps=nw, num_stages=ns)
        K.gemv_epilogue_kernel[(tdim, triton_cdiv(n, 64))](
            acc, lin.bias, out, out, n, BLOCK=64, HAS_BIAS=True, HAS_RES=False,
            TDIM=tdim, SPLIT_K=sk, num_warps=4)
    else:
        kern = (K.gemv_int4_kernel if lin.kind == "int4" else K.gemv_int8_kernel)
        wptr = lin.packed.view(torch.int8) if lin.kind == "int4" else lin.packed
        extra = {} if lin.kind == "int4" else {"SUB": max(1, lin.group // block_k)}
        kern[(triton_cdiv(n, bn),)](x, wptr, lin.scales, lin.bias, out, out, n, k,
                                    BLOCK_N=bn, BLOCK_K=block_k, GROUP=lin.group,
                                    TDIM=tdim, HAS_BIAS=True, HAS_RES=False,
                                    num_warps=nw, num_stages=ns, **extra)


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ------------------------------------------------------------------------ attn
def do_attn(args) -> None:
    from kaze.config import ModelConfig
    cfg = ModelConfig.from_pretrained(args.model)
    print("attn mode needs a built model; use profile mode for attention share")
    print("shapes:", cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)


def main() -> None:
    ap = argparse.ArgumentParser("kernel_lab")
    ap.add_argument("mode", choices=["profile", "micro", "attn"])
    ap.add_argument("--model", default="/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--kind", default="int8")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lm-head-kind", default="int8")
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--streams", type=int, default=1, help="concurrent decode rows")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--only", default="")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--all", action="store_true", help="print the top variants in a sweep")
    ap.add_argument("--reps", type=int, default=16, help="kernels per captured graph")
    ap.add_argument("--tdim", type=int, default=16,
                    help="rows in the activation buffer (1 selects the batch-1 matvec path)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    {"profile": do_profile, "micro": do_micro, "attn": do_attn}[args.mode](args)


if __name__ == "__main__":
    main()
