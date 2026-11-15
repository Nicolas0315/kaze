"""手書き CUDA の int4 GEMV を Triton 版と比べる（開発用）。

    python3 bench/cuda_gemv.py            # 数値一致と帯域を測る
    python3 bench/cuda_gemv.py --rows 2048

nvcc が PATH に無い環境では先に通しておく:
    PATH=/usr/local/cuda-13.4/bin:$PATH python3 bench/cuda_gemv.py

現状の実測（RTX 3070, N=37888 K=3584）:
    Triton  332 GB/s / CUDA(8レーン,8B,half2) 276 GB/s / CUDA(4レーン,16B) 150 GB/s
数値はどの構成も PyTorch 参照と cos 1.000000。
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from torch.utils.cpp_extension import load_inline

import kaze.kernels.quant_linear as K
from kaze.quant import QuantLinear

CU = pathlib.Path(__file__).resolve().parents[1] / "kaze" / "kernels" / "cuda_gemv.cu"


def build():
    return load_inline(
        name="kaze_cuda_gemv_bench",
        cpp_sources="torch::Tensor gemv1(torch::Tensor x, torch::Tensor wq,"
                    " torch::Tensor scales, int64_t n);",
        cuda_sources=CU.read_text(),
        functions=["gemv1"],
        extra_cuda_cflags=["-O3", "-arch=sm_86", "-allow-unsupported-compiler"],
        verbose=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=37888)
    ap.add_argument("--k", type=int, default=3584)
    args = ap.parse_args()
    mod = build()
    torch.manual_seed(0)

    # 数値: PyTorch で手展開した参照と比べる
    n_small, k = 64, 512
    w = (torch.randn(n_small, k, device="cuda") * 0.02).half()
    x = (torch.randn(16, k, device="cuda") * 0.5).half()
    lin = QuantLinear.from_float(w, None, kind="int4", group=128)
    w32 = lin.packed
    nib = torch.stack([(w32 >> (4 * j)) & 0xF for j in range(8)], dim=-1)
    deq = (nib.reshape(n_small, k).float() - 8.0) * lin.scales.float().repeat_interleave(128, dim=1)
    ref = x[0].float() @ deq.T
    cu = mod.gemv1(x, lin.packed.view(torch.int8), lin.scales, n_small).float()
    cos = torch.nn.functional.cosine_similarity(cu, ref, 0).item()
    print(f"cos vs torch ref: {cos:.6f}")

    # 帯域
    w2 = (torch.randn(args.rows, args.k, device="cuda") * 0.02).half()
    lin2 = QuantLinear.from_float(w2, None, kind="int4", group=128)

    def timeit(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    xb = (torch.randn(16, args.k, device="cuda") * 0.5).half()
    total = args.rows * args.k / 2
    t_tri = timeit(lambda: K.gemv(xb, lin2, 16))
    t_cu = timeit(lambda: mod.gemv1(xb, lin2.packed.view(torch.int8), lin2.scales, args.rows))
    print(f"triton {t_tri * 1e3:.3f} ms  {total / t_tri / 1e9:.0f} GB/s")
    print(f"cuda   {t_cu * 1e3:.3f} ms  {total / t_cu / 1e9:.0f} GB/s"
          f"  ({'勝ち' if t_cu < t_tri else '負け'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
