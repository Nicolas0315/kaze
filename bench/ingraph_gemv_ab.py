"""In-graph sweep of the decode GEMV kernels (split-K matvec) at arbitrary shapes.

The eager sweep (bench/sweep_7b_decode.py) misled three times; this times a captured
graph of N launches with CUDA events instead, which is stable to <1%.
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kaze.kernels.quant_linear import (gemv_int4_splitk_kernel, gemv_int4_kernel,
                                       gemv_epilogue_kernel)

torch.manual_seed(0)
dev = "cuda"

def bench_splitk(n, k, cfg, tdim=1, reps=30):
    bn, sk, nw, ns, bk = cfg
    x = (torch.randn(tdim, k, device=dev) * 0.1).half()
    wq = torch.randint(-128, 127, (n, k // 2), dtype=torch.int8, device=dev)
    sc = torch.rand(n, k // 128, dtype=torch.float16, device=dev) * 0.01
    acc = torch.zeros(sk, tdim, n, dtype=torch.float32, device=dev)
    out = torch.empty(tdim, n, dtype=torch.float16, device=dev)

    def launch():
        gemv_int4_splitk_kernel[((n + bn - 1) // bn, sk)](
            x, wq, sc, acc, n, k, BLOCK_N=bn, BLOCK_K=bk, GROUP=128, TDIM=tdim,
            SPLIT_K=sk, num_warps=nw, num_stages=ns)
        gemv_epilogue_kernel[(tdim, (n + 255) // 256)](
            acc, out, out, out, n, BLOCK=256, HAS_BIAS=False, HAS_RES=False,
            TDIM=tdim, SPLIT_K=sk, num_warps=4)

    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(reps):
                launch()
    torch.cuda.synchronize()
    e1 = torch.cuda.Event(True); e2 = torch.cuda.Event(True)
    e1.record()
    for _ in range(5):
        g.replay()
    e2.record()
    torch.cuda.synchronize()
    return e1.elapsed_time(e2) / 5 / reps * 1e3

def sweep(name, n, k, tiles, tdim=1):
    nbytes = n * k / 2
    res = {t: [] for t in tiles}
    for r in range(3):
        order = tiles if r % 2 == 0 else tiles[::-1]
        for t in order:
            res[t].append(bench_splitk(n, k, t, tdim))
    best = min((statistics.median(v), t) for t, v in res.items())
    print("%s [%d,%d]:" % (name, n, k))
    for t in tiles:
        med = statistics.median(res[t])
        mark = " <-" if t == best[1] else ""
        print("  %-16s %7.1f us  %5.0f GB/s%s" % (str(t), med, nbytes / (med / 1e3) / 1e9, mark))
    return best

if __name__ == "__main__":
    # 7B geometry: hidden 4096, 32q+8kv x128, intermediate 14336, vocab 32000
    TILES = [(64, 8, 2, 2, 128), (64, 4, 2, 2, 128), (64, 8, 4, 2, 128),
             (64, 4, 4, 2, 128), (32, 4, 2, 2, 128), (32, 8, 2, 2, 128),
             (16, 8, 2, 2, 128), (16, 4, 2, 2, 128), (128, 4, 4, 2, 128),
             (64, 4, 4, 3, 128)]
    sweep("qkv", 6144, 4096, TILES)
    sweep("o", 4096, 4096, TILES)
    sweep("down", 4096, 14336, TILES)
