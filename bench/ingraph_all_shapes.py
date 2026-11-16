"""Comprehensive in-graph tile sweep for every decode shape we ship (0.5B-7B).

Covers the fused int4 swiglu kernel and the split-K GEMV (+epilogue) at the real
geometry of each model.  All timing is a captured CUDA graph replayed with CUDA
events, interleaved across tiles; eager sweeps have misled three times.
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kaze.kernels.quant_linear import (gemv_int4_splitk_kernel, gemv_int4_kernel,
                                       gemv_epilogue_kernel,
                                       gemv_int4_swiglu_kernel)

torch.manual_seed(0)
dev = "cuda"


def capture_time(fn, reps=30, inner=5):
    for _ in range(3):
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


def time_splitk(n, k, cfg, tdim=1):
    bn, sk, nw, ns, bk = cfg
    x = (torch.randn(tdim, k, device=dev) * 0.1).half()
    wq = torch.randint(-128, 127, (n, k // 2), dtype=torch.int8, device=dev)
    sc = torch.rand(n, max(1, k // 128), dtype=torch.float16, device=dev) * 0.01
    acc = torch.zeros(max(sk, 1), tdim, n, dtype=torch.float32, device=dev)
    out = torch.empty(tdim, n, dtype=torch.float16, device=dev)

    def launch():
        if sk > 1:
            gemv_int4_splitk_kernel[((n + bn - 1) // bn, sk)](
                x, wq, sc, acc, n, k, BLOCK_N=bn, BLOCK_K=bk, GROUP=128,
                TDIM=tdim, SPLIT_K=sk, num_warps=nw, num_stages=ns)
            gemv_epilogue_kernel[(tdim, (n + 255) // 256)](
                acc, out, out, out, n, BLOCK=256, HAS_BIAS=False, HAS_RES=False,
                TDIM=tdim, SPLIT_K=sk, num_warps=4)
        else:
            gemv_int4_kernel[((n + bn - 1) // bn,)](
                x, wq, sc, out, out, out, n, k, BLOCK_N=bn, BLOCK_K=bk,
                GROUP=128, TDIM=tdim, HAS_BIAS=False, HAS_RES=False,
                num_warps=nw, num_stages=ns)

    return capture_time(launch)


def time_swiglu(nh, k, cfg):
    bn, nw, ns = cfg
    x = (torch.randn(1, k, device=dev) * 0.1).half()
    wq = torch.randint(-128, 127, (2 * nh, k // 2), dtype=torch.int8, device=dev)
    sc = torch.rand(2 * nh, max(1, k // 128), dtype=torch.float16, device=dev) * 0.01
    out = torch.empty(1, nh, dtype=torch.float16, device=dev)

    def launch():
        gemv_int4_swiglu_kernel[((nh + bn - 1) // bn,)](
            x, wq, sc, out, nh, k, BLOCK_N=bn, BLOCK_K=128, GROUP=128, TDIM=1,
            num_warps=nw, num_stages=ns)

    return capture_time(launch)


def sweep(label, nbytes_per_call, tiles, fn):
    res = {t: [] for t in tiles}
    for r in range(3):
        order = tiles if r % 2 == 0 else tiles[::-1]
        for t in order:
            res[t].append(fn(t))
    best = min((statistics.median(v), t) for t, v in res.items())
    print(label)
    for t in tiles:
        med = statistics.median(res[t])
        mark = "  <-best" if t == best[1] else ""
        print("  %-18s %7.1f us  %5.0f GB/s%s" % (str(t), med,
              nbytes_per_call / (med / 1e3) / 1e9, mark))
    return best


MODELS = {
    "0.5B": dict(hidden=896, nh=4864, qkv=(1152, 896), o=(896, 896),
                 down=(896, 4864), head=(151936, 896)),
    "1.5B": dict(hidden=1536, nh=5504, qkv=(2048, 1536), o=(1536, 1536),
                 down=(1536, 11008), head=(151936, 1536)),
    "3B":   dict(hidden=2048, nh=5504, qkv=(2560, 2048), o=(2048, 2048),
                 down=(2048, 11008), head=(151936, 2048)),
    "7B":   dict(hidden=4096, nh=14336, qkv=(6144, 4096), o=(4096, 4096),
                 down=(4096, 14336), head=(32000, 4096)),
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "0.5B"
    m = MODELS[which]
    SW_TILE_LIMITS = [(16, 4, 2), (16, 4, 3), (16, 4, 4), (32, 4, 2), (32, 4, 3),
                      (32, 4, 4), (64, 4, 4), (64, 8, 4), (128, 8, 4)]
    nh, k = m["nh"], m["hidden"]
    sweep("swiglu nh=%d k=%d" % (nh, k), nh * k, SW_TILE_LIMITS,
          lambda t: time_swiglu(nh, k, t))
    SK_TILES = [(16, 8, 2, 2, 128), (16, 4, 2, 2, 128), (32, 4, 2, 2, 128),
                (32, 8, 2, 2, 128), (32, 8, 4, 2, 128), (32, 4, 2, 3, 128),
                (64, 4, 2, 2, 128), (64, 8, 2, 2, 128), (64, 8, 4, 2, 128)]
    for role in ("qkv", "o", "down"):
        n, kk = m[role]
        sweep("splitk %s [%d,%d]" % (role, n, kk), n * kk / 2, SK_TILES,
              lambda t, n=n, kk=kk: time_splitk(n, kk, t))
