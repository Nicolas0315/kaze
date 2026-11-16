"""In-graph sweep of the prefill W4A8 GEMM and fused swiglu tiles at real shapes.

The decode-side lesson (eager microbenchmarks mislead; captured-graph replay is stable
to <1%) applies to prefill tiles too.  Shapes: the three big layers of 0.5B/3B at the
m=512 prefill bucket.
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kaze.kernels.quant_linear import (gemm_int4_w8a8_kernel,
                                       gemm_swiglu_w8a8_kernel)

torch.manual_seed(0)
dev = "cuda"


def capture_time(fn, reps=12, inner=4):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    e1 = torch.cuda.Event(True); e2 = torch.cuda.Event(True)
    e1.record()
    for _ in range(inner):
        g.replay()
    e2.record()
    torch.cuda.synchronize()
    return e1.elapsed_time(e2) / inner / reps * 1e3


def time_gemm(m, n, k, cfg):
    bm, bn, nw, ns = cfg
    aq = torch.ones((m, k), dtype=torch.int8, device=dev)
    as_ = torch.ones((m, 1), dtype=torch.float32, device=dev)
    wq = torch.full((n, k // 2), 17, dtype=torch.int8, device=dev)
    sw = torch.ones((n, 1), dtype=torch.float16, device=dev)
    out = torch.empty((m, n), dtype=torch.float16, device=dev)

    def launch():
        gemm_int4_w8a8_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            aq, as_, wq, sw, out, out, m, n, k, HAS_BIAS=False, IS_INT4=True,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP=128, XGROUP=128,
            num_warps=nw, num_stages=ns)

    return capture_time(launch)


def time_swiglu(m, nh, k, cfg):
    bm, bn, nw, ns = cfg
    aq = torch.ones((m, k), dtype=torch.int8, device=dev)
    as_ = torch.ones((m, 1), dtype=torch.float32, device=dev)
    wq = torch.full((2 * nh, k // 2), 17, dtype=torch.int8, device=dev)
    sw = torch.ones((2 * nh, 1), dtype=torch.float16, device=dev)
    out = torch.empty((m, nh), dtype=torch.float16, device=dev)
    outq = torch.empty((m, nh), dtype=torch.int8, device=dev)
    outs = torch.empty((m, nh), dtype=torch.float16, device=dev)

    def launch():
        gemm_swiglu_w8a8_kernel[(triton.cdiv(m, bm), triton.cdiv(2 * nh, bn))](
            aq, as_, wq, sw, out, outq, outs, m, 2 * nh, k,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP=128, XGROUP=128,
            num_warps=nw, num_stages=ns)

    return capture_time(launch)


if __name__ == "__main__":
    import triton
    M = 512
    GEMM_TILES = [(64, 128, 4, 3), (64, 128, 4, 2), (64, 128, 8, 3),
                  (32, 128, 4, 3), (64, 64, 4, 3), (128, 128, 4, 3),
                  (64, 128, 4, 4), (64, 256, 4, 3)]
    SW_TILES = [(64, 128, 8, 2), (64, 128, 8, 3), (64, 128, 4, 2),
                (32, 128, 8, 2), (64, 64, 8, 2), (128, 128, 8, 2),
                (64, 128, 8, 4), (32, 64, 8, 2)]
    SHAPES = {"0.5B": dict(hidden=896, inter=4864, qkv=1152),
              "3B": dict(hidden=2048, inter=11008, qkv=2560)}
    which = sys.argv[1] if len(sys.argv) > 1 else "0.5B"
    s = SHAPES[which]
    for name, n, k, fn, tiles in (
            ("gemm gate_up", 2 * s["inter"], s["hidden"], time_gemm, GEMM_TILES),
            ("gemm down", s["hidden"], s["inter"], time_gemm, GEMM_TILES),
            ("gemm qkv", s["qkv"], s["hidden"], time_gemm, GEMM_TILES),
            ("swiglu gate_up", s["inter"], s["hidden"], time_swiglu, SW_TILES)):
        per_call_bytes = (n * k / 2 + k * M)
        best = sweep_out = None
        res = {}
        for r in range(2):
            order = tiles if r % 2 == 0 else tiles[::-1]
            for t in order:
                res.setdefault(t, []).append(fn(M, n, k, t) if fn is time_gemm
                                             else fn(M, n, k, t))
        meds = {t: statistics.median(v) for t, v in res.items()}
        bt = min(meds, key=meds.get)
        print("%s [%d,%d] m=%d:" % (name, n, k, M))
        for t in tiles:
            mark = "  <-best" if t == bt else ""
            print("  %-18s %7.1f us %5.0f GB/s%s" % (str(t), meds[t],
                  per_call_bytes / (meds[t] / 1e3) / 1e9, mark))
