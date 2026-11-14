"""Fused RMSNorm (fp16 in, fp16 out, fp32 statistics)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(X, W, Y, YQ, YS, N, eps, QGROUP: tl.constexpr,
                   BLOCK: tl.constexpr, HAS_Q8: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    y = x * tl.rsqrt(var + eps) * w
    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)
    if HAS_Q8:
        # int8 copy for the next GEMV, quantised per QGROUP lanes: doing it here saves
        # a kernel launch per layer (which is what made W4A8 neutral before)
        ng: tl.constexpr = BLOCK // QGROUP
        y2 = tl.where(mask, y, 0.0)
        yr = tl.reshape(y2, (ng, QGROUP))
        s = tl.maximum(tl.max(tl.abs(yr), axis=1), 1e-8) / 127.0
        q = tl.reshape(yr / s[:, None], (BLOCK,))
        tl.store(YQ + row * N + cols, q.to(tl.int8), mask=mask)
        tl.store(YS + row * ng + tl.arange(0, ng), s)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor | None = None,
            out_q: torch.Tensor | None = None, out_s: torch.Tensor | None = None,
            group: int = 128):
    """x [M, N] fp16 -> [M, N] fp16, normalized over the last dim.

    Passing out_q/out_s also writes an int8 copy plus per-(row, group) scales, so the
    following GEMV can run on integer tensor cores without an extra launch.
    """
    m, n = x.shape
    if out is None:
        out = torch.empty_like(x)
    block = triton.next_power_of_2(n)
    has_q8 = out_q is not None and out_s is not None
    rmsnorm_kernel[(m,)](x, w, out, out_q if has_q8 else out, out_s if has_q8 else out,
                         n, eps, QGROUP=group, BLOCK=block, HAS_Q8=has_q8, num_warps=4)
    return out
