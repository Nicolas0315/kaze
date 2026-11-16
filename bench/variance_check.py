"""How much does a single in-graph kernel measurement drift?  Run the same config six
times over ~30 s and print each sample, so a config conclusion is only kept when the
gap is larger than the spread.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from bench.kernel_lab import make_linear, time_graph
from kaze.kernels import quant_linear as K


def run(kind: str, n: int, k: int, cfg: tuple[int, int, int, int, int], samples: int = 6) -> None:
    lin = make_linear(n, k, kind, 128)
    x = torch.rand(16, k, dtype=torch.float16, device="cuda") - 0.5
    out = torch.empty((16, n), dtype=torch.float16, device="cuda")
    bn, sk, nw, ns, bk = cfg
    acc = torch.zeros((sk, 16, n), dtype=torch.float32, device="cuda") if sk > 1 else None
    nb = lin.packed.numel() * lin.packed.element_size() + lin.scales.numel() * 2

    def call():
        if sk > 1:
            kern = K.gemv_int4_splitk_kernel if kind == "int4" else K.gemv_int8_splitk_kernel
            wq = lin.packed.view(torch.int8) if kind == "int4" else lin.packed
            kern[(triton_cdiv(n, bn), sk)](x, wq, lin.scales, acc, n, k, BLOCK_N=bn,
                                           BLOCK_K=bk, GROUP=128, TDIM=16, SPLIT_K=sk,
                                           num_warps=nw, num_stages=ns)
            K.gemv_epilogue_kernel[(16, triton_cdiv(n, 64))](
                acc, lin.bias, out, out, n, BLOCK=64, HAS_BIAS=True, HAS_RES=False,
                TDIM=16, SPLIT_K=sk, num_warps=4)
            return
        kern = K.gemv_int4_kernel if kind == "int4" else K.gemv_int8_kernel
        wq = lin.packed.view(torch.int8) if kind == "int4" else lin.packed
        extra = {} if kind == "int4" else {"SUB": max(1, 128 // bk)}
        kern[(triton_cdiv(n, bn),)](x, wq, lin.scales, lin.bias, out, out, n, k,
                                    BLOCK_N=bn, BLOCK_K=bk, GROUP=128, TDIM=16,
                                    HAS_BIAS=True, HAS_RES=False, num_warps=nw,
                                    num_stages=ns, **extra)

    us = []
    for _ in range(samples):
        us.append(time_graph(call, reps=10))
        torch.cuda.synchronize()
        time.sleep(4)
    lo, hi = min(us), max(us)
    samples_txt = " ".join(f"{u:.1f}" for u in us)
    print(f"{kind} {n}x{k} cfg={cfg}: {samples_txt}"
          f"  min {lo:.1f} med {sorted(us)[len(us) // 2]:.1f} max {hi:.1f}"
          f"  spread {(hi - lo) / lo * 100:.1f}%  ({nb / (lo * 1e-6) / 1e9:.0f} GB/s best)")
    del lin, x, out, acc
    torch.cuda.empty_cache()


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


if __name__ == "__main__":
    run("int4", 151936, 896, (64, 1, 4, 2, 128))
    run("int4", 151936, 896, (32, 1, 2, 4, 128))
    run("int8", 1152, 896, (64, 8, 4, 3, 128))
    run("int8", 1152, 896, (32, 1, 4, 2, 128))
