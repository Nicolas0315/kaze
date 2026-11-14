"""Fused decode kernels: RoPE + KV scatter, and the GEMV epilogue.

On a decode step the activations are tiny (one token x hidden) while the weight
stream is huge.  Anything that touches activations must therefore be a single
kernel, otherwise the step is dominated by launch and index overhead rather than
by HBM bandwidth.  rope_scatter replaces ~17 tiny torch calls per layer (four
cos/sin gathers, rope elementwise math, cat, two index copies, two contiguous
copies) with one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rope_scatter_kernel(QKV, COS, SIN, OUTQ, KFLAT, VFLAT, KS, VS, SLOTS, POS,
                        NQ, NKV, D, QS, KVS, QN, KN, EPS,
                        HALF: tl.constexpr, BM: tl.constexpr, NH,
                        HAS_NORM: tl.constexpr = False, Q8K: tl.constexpr = False,
                        Q8V: tl.constexpr = False,
                        NS: tl.constexpr = 1, LANE: tl.constexpr = 32):
    h = tl.program_id(0)
    m = tl.program_id(1)
    offs = tl.arange(0, HALF)
    pos = tl.load(POS + m)
    stride_row = QS + 2 * KVS
    base = QKV + m.to(tl.int64) * stride_row
    c = tl.load(COS + pos.to(tl.int64) * HALF + offs)
    s = tl.load(SIN + pos.to(tl.int64) * HALF + offs)
    if h < NQ:
        p = base + h * D
        x1 = tl.load(p + offs).to(tl.float32)
        x2 = tl.load(p + HALF + offs).to(tl.float32)
        if HAS_NORM:
            var = (tl.sum(x1 * x1 + x2 * x2, axis=0)) / D
            r = tl.rsqrt(var + EPS)
            x1 = x1 * r * tl.load(QN + offs).to(tl.float32)
            x2 = x2 * r * tl.load(QN + HALF + offs).to(tl.float32)
        o = OUTQ + (m * NQ + h) * D
        tl.store(o + offs, (x1 * c - x2 * s).to(tl.float16))
        tl.store(o + HALF + offs, (x2 * c + x1 * s).to(tl.float16))
    else:
        j = h - NQ
        slot = tl.load(SLOTS + m)
        if j < NKV:
            p = base + QS + j * D
            x1 = tl.load(p + offs).to(tl.float32)
            x2 = tl.load(p + HALF + offs).to(tl.float32)
            if HAS_NORM:
                var = (tl.sum(x1 * x1 + x2 * x2, axis=0)) / D
                r = tl.rsqrt(var + EPS)
                x1 = x1 * r * tl.load(KN + offs).to(tl.float32)
                x2 = x2 * r * tl.load(KN + HALF + offs).to(tl.float32)
            dstk = KFLAT + slot.to(tl.int64) * KVS + j * D
            mask = m < NH
            k1 = x1 * c - x2 * s
            k2 = x2 * c + x1 * s
            if Q8K:
                # int8 cache, q8_0 style: one scale per LANE lanes of the head.
                # A single scale for the whole head let outliers crush the small
                # values (logits cos 0.992 against fp16).  The kernel sees the head as
                # two halves (x1 = dims 0..HALF-1, x2 = dims HALF..D-1), so each half
                # contributes its own scales, in dim order.
                CH: tl.constexpr = HALF // LANE
                base_s = KS + (slot.to(tl.int64) * NKV + j) * NS
                s1 = tl.zeros((HALF,), dtype=tl.float32)
                s2 = tl.zeros((HALF,), dtype=tl.float32)
                for ci in tl.static_range(CH):
                    sc_lo: tl.constexpr = ci * LANE
                    sel = (offs >= sc_lo) & (offs < sc_lo + LANE)
                    a1 = tl.maximum(tl.max(tl.abs(tl.where(sel, k1, 0.0)), axis=0), 1e-8) / 127.0
                    a2 = tl.maximum(tl.max(tl.abs(tl.where(sel, k2, 0.0)), axis=0), 1e-8) / 127.0
                    s1 = tl.where(sel, a1, s1)
                    s2 = tl.where(sel, a2, s2)
                    tl.store(base_s + ci, a1)
                    tl.store(base_s + CH + ci, a2)
                q1 = k1 / s1
                q2 = k2 / s2
                tl.store(dstk + offs, (q1 + tl.where(q1 >= 0, 0.5, -0.5)).to(tl.int8),
                         mask=mask)
                tl.store(dstk + HALF + offs, (q2 + tl.where(q2 >= 0, 0.5, -0.5)).to(tl.int8),
                         mask=mask)
            else:
                # cast to the cache dtype so an fp8 KV cache works unchanged
                tl.store(dstk + offs, k1.to(dstk.dtype.element_ty), mask=mask)
                tl.store(dstk + HALF + offs, k2.to(dstk.dtype.element_ty), mask=mask)
        else:
            jv = j - NKV
            p = base + QS + KVS + jv * D
            v = tl.load(p + offs)
            v2 = tl.load(p + HALF + offs)
            dstv = VFLAT + slot.to(tl.int64) * KVS + jv * D
            mask = m < NH
            if Q8V:
                CH2: tl.constexpr = HALF // LANE
                base_s = VS + (slot.to(tl.int64) * NKV + jv) * NS
                t1 = tl.zeros((HALF,), dtype=tl.float32)
                t2 = tl.zeros((HALF,), dtype=tl.float32)
                for cj in tl.static_range(CH2):
                    lo2: tl.constexpr = c * LANE
                    sel2 = (offs >= lo2) & (offs < lo2 + LANE)
                    a1 = tl.maximum(tl.max(tl.abs(tl.where(sel2, v, 0.0)), axis=0), 1e-8) / 127.0
                    a2 = tl.maximum(tl.max(tl.abs(tl.where(sel2, v2, 0.0)), axis=0), 1e-8) / 127.0
                    t1 = tl.where(sel2, a1, t1)
                    t2 = tl.where(sel2, a2, t2)
                    tl.store(base_s + c, a1)
                    tl.store(base_s + CH2 + c, a2)
                qv = v / t1
                qv2 = v2 / t2
                tl.store(dstv + offs, (qv + tl.where(qv >= 0, 0.5, -0.5)).to(tl.int8),
                         mask=mask)
                tl.store(dstv + HALF + offs, (qv2 + tl.where(qv2 >= 0, 0.5, -0.5)).to(tl.int8),
                         mask=mask)
            else:
                tl.store(dstv + offs, v.to(dstv.dtype.element_ty), mask=mask)
                tl.store(dstv + HALF + offs, v2.to(dstv.dtype.element_ty), mask=mask)


def rope_scatter(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 out_q: torch.Tensor, k_flat: torch.Tensor, v_flat: torch.Tensor,
                 slots: torch.Tensor, positions: torch.Tensor,
                 n_q_heads: int, n_kv_heads: int, head_dim: int, num_real: int,
                 q_norm: torch.Tensor | None = None, k_norm: torch.Tensor | None = None,
                 eps: float = 1e-6,
                 k_scale: torch.Tensor | None = None,
                 v_scale: torch.Tensor | None = None) -> None:
    """RoPE + KV scatter, optionally with Qwen3-style per-head RMSNorm fused in."""
    tdim = qkv.shape[0]
    grid = (n_q_heads + 2 * n_kv_heads, tdim)
    has_norm = q_norm is not None
    qn = q_norm if has_norm else qkv
    kn = k_norm if k_norm is not None else qn
    q8k = k_flat.dtype == torch.int8
    q8v = v_flat.dtype == torch.int8
    ks = k_scale if q8k else qkv
    vs = v_scale if q8v else qkv
    ns_scale = ((k_scale if q8k else v_scale).shape[-1] if (q8k or q8v) else 1)
    rope_scatter_kernel[grid](
        qkv, cos, sin, out_q, k_flat, v_flat, ks, vs, slots, positions,
        n_q_heads, n_kv_heads, head_dim, n_q_heads * head_dim, n_kv_heads * head_dim,
        qn, kn, eps,
        HALF=head_dim // 2, BM=tdim, NH=num_real, HAS_NORM=has_norm,
        Q8K=q8k, Q8V=q8v, NS=ns_scale, LANE=32, num_warps=1)


@triton.jit
def qk_norm_kernel(QKV, WQ, WK, stride_row, NQ, NKV, D: tl.constexpr,
                   EPS, BLOCK: tl.constexpr):
    """Qwen3-style per-head RMSNorm on q and k, done in place on the qkv buffer.

    Grid: (rows * (NQ + NKV),).  One program normalizes one head of one row, so
    no copy of q/k is needed before RoPE and the KV store.
    """
    pid = tl.program_id(0)
    heads = NQ + NKV
    row = pid // heads
    h = pid % heads
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    if h < NQ:
        base = QKV + row.to(tl.int64) * stride_row + h * D
        w = tl.load(WQ + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        base = QKV + row.to(tl.int64) * stride_row + NQ * D + (h - NQ) * D
        w = tl.load(WK + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    y = x * tl.rsqrt(var + EPS) * w
    tl.store(base + offs, y.to(tl.float16), mask=mask)


def qk_norm_inplace(qkv: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
                    n_q_heads: int, n_kv_heads: int, head_dim: int, eps: float) -> None:
    rows = qkv.shape[0]
    grid = (rows * (n_q_heads + n_kv_heads),)
    qk_norm_kernel[grid](qkv, wq, wk, qkv.stride(0), n_q_heads, n_kv_heads, head_dim,
                         eps, BLOCK=triton.next_power_of_2(head_dim), num_warps=1)
