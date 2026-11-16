"""Fused RMSNorm (fp16 in, fp16 out, fp32 statistics)."""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Load the split-K partials as one 2D tile and reduce with a tree (instead of SPLIT_K
# dependent loads).  Measured with bench/fold_ab.py; see docs/kernels.md.
FOLD_TREE = os.environ.get("KAZE_FOLD_TREE", "1") != "0"


@triton.jit
def rmsnorm_kernel(X, W, Y, YQ, YS, PART, BIAS, N, eps, QGROUP: tl.constexpr,
                   BLOCK: tl.constexpr, HAS_Q8: tl.constexpr,
                   HAS_SPLITK: tl.constexpr = False, SPLIT_K: tl.constexpr = 1,
                   HAS_BIAS: tl.constexpr = False, HAS_RES: tl.constexpr = False,
                   TDIM: tl.constexpr = 1, TREE: tl.constexpr = False):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    if HAS_SPLITK:
        # Fold the split-K partials here instead of in a separate epilogue kernel: the
        # additions follow the epilogue's order (sp ascending, then bias, then residual,
        # then one fp16 rounding), so the bits match gemv_epilogue_kernel exactly.
        if TREE:
            # one 2D load instead of SPLIT_K dependent ones: the partials are fetched in
            # parallel and summed as a tree.  Saves most of the latency when the split-K
            # depth is 8, and stays deterministic, but is no longer bit-identical to
            # gemv_epilogue_kernel's sp-ascending sum.
            sps = tl.arange(0, SPLIT_K)
            parts = tl.load(PART + sps[:, None] * (TDIM * N) + row * N + cols[None, :],
                            mask=mask[None, :], other=0.0)
            y32 = tl.sum(parts, axis=0)
        else:
            y32 = tl.zeros((BLOCK,), dtype=tl.float32)
            for sp in tl.static_range(SPLIT_K):
                y32 += tl.load(PART + sp * (TDIM * N) + row * N + cols, mask=mask, other=0.0)
        if HAS_BIAS:
            y32 += tl.load(BIAS + cols, mask=mask, other=0.0)
        if HAS_RES:
            y32 += tl.load(X + row * N + cols, mask=mask, other=0.0)
        xh = y32.to(tl.float16)
        if HAS_RES:
            tl.store(X + row * N + cols, xh, mask=mask)
        x = xh.to(tl.float32)
    else:
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
        v = yr / s[:, None]
        # round half away from zero, exactly like quant_act_kernel, so the prefill GEMM
        # sees the same activations whether the quantisation ran here or stand-alone
        q = tl.reshape(v + tl.where(v >= 0, 0.5, -0.5), (BLOCK,))
        tl.store(YQ + row * N + cols, q.to(tl.int8), mask=mask)
        groups = tl.cdiv(N, QGROUP)
        group_ids = tl.arange(0, ng)
        tl.store(YS + row * groups + group_ids, s, mask=group_ids < groups)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor | None = None,
            out_q: torch.Tensor | None = None, out_s: torch.Tensor | None = None,
            group: int = 128, partials: torch.Tensor | None = None, bias: torch.Tensor | None = None,
            res: torch.Tensor | None = None):
    """x [M, N] fp16 -> [M, N] fp16, normalized over the last dim.

    Passing out_q/out_s also writes an int8 copy plus per-(row, group) scales, so the
    following GEMV can run on integer tensor cores without an extra launch.

    Passing partials (shape [SPLIT_K, M, N] fp32, the output of a split-K GEMV) folds the
    epilogue into this kernel: sum the splits in order, add bias, add the residual in
    res, write the fp16 result back to res, and normalize.  This is bit identical to
    gemv_epilogue_kernel and removes one launch per layer.
    """
    m, n = x.shape
    if out is None:
        out = torch.empty_like(x)
    block = triton.next_power_of_2(n)
    has_q8 = out_q is not None and out_s is not None
    has_sk = partials is not None
    dummy = out
    rmsnorm_kernel[(m,)](x if res is None else res, w, out,
                         out_q if has_q8 else dummy, out_s if has_q8 else dummy,
                         partials if has_sk else dummy,
                         bias if bias is not None else dummy,
                         n, eps, QGROUP=group, BLOCK=block, HAS_Q8=has_q8,
                         HAS_SPLITK=has_sk, SPLIT_K=partials.shape[0] if has_sk else 1,
                         HAS_BIAS=bias is not None, HAS_RES=res is not None,
                         TDIM=partials.shape[1] if has_sk else m,
                         # the 2D tile costs SPLIT_K * BLOCK * 4 B of registers/SMEM:
                         # at hidden 896 (BLOCK 1024) it is a +2.6% decode win, at 1536
                         # (BLOCK 2048) it is 64 KB and costs -7.8%, so gate on the size
                         TREE=bool(FOLD_TREE and has_sk and block <= 1024),
                         # keep 4 warps: tl.sum(x*x) is order sensitive, and 8 warps
                         # changes the fp32 reduction tree, which shows up as a 1-2 ULP
                         # difference in the logits (measured, int8 0.5B)
                         num_warps=4)
    return out
