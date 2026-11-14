"""Quantized linear kernels: weight-only int4 / int8 / fp16.

Decode (M <= 16) is a GEMV: weights are streamed exactly once, so the kernel is
bandwidth bound and the only things that matter are coalesced 32-bit nibble
loads, enough CTAs (split-K), and zero wasted traffic.  Prefill (M large) is a
tiled dequant-on-the-fly GEMM over the same layout.

Layout recap (see kaze/quant.py):
    packed : int32 / int8 [N, K/8 or K]
    scales : fp16 [N, K/GROUP]
    w[n, k] = (nibble - 8) * scales[n, k // GROUP]
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from kaze.quant import QuantLinear


@triton.jit
def _unpack_int4(wp, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """wp [BLOCK_N, BLOCK_K//8] int32 -> [BLOCK_N, BLOCK_K] fp16 in [-8, 7]."""
    shifts = tl.arange(0, 8) * 4
    nib = (wp[:, :, None] >> shifts[None, None, :]) & 0xF
    return tl.reshape(nib, (BLOCK_N, BLOCK_K)).to(tl.float16) - 8.0


# --------------------------------------------------------------------- decode
@triton.jit
def gemv_int4_kernel(X, WQ, SC, BIAS, Y, RES, N, K,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                     TDIM: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    offs_kb = tl.arange(0, BLOCK_K // 2)
    for k0 in range(0, K, BLOCK_K):
        b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kb[None, :],
                    mask=mask_n[:, None], other=0)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        bi = b.to(tl.int32)
        sb = s * 8.0
        lo = (bi & 0xF).to(tl.float16) * s[:, None] - sb[:, None]
        hi = ((bi >> 4) & 0xF).to(tl.float16) * s[:, None] - sb[:, None]
        w = tl.interleave(lo, hi)
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        acc += tl.dot(w, tl.trans(xs))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemv_int8_kernel(X, WQ, SC, BIAS, Y, RES, N, K,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                     SUB: tl.constexpr,
                     TDIM: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    # one scale per GROUP lanes of K: accumulate the raw int8 products for the whole
    # group, then apply the scale once (instead of multiplying every weight element)
    for k0 in range(0, K, GROUP):
        acc_g = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
        for j in tl.static_range(SUB):
            kk = k0 + j * BLOCK_K
            w = tl.load(WQ + offs_n[:, None] * K + kk + tl.arange(0, BLOCK_K)[None, :],
                        mask=mask_n[:, None], other=0).to(tl.float16)
            xs = tl.load(X + offs_m[:, None] * K + kk + tl.arange(0, BLOCK_K)[None, :])
            acc_g += tl.dot(w, tl.trans(xs))
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        acc += acc_g * s[:, None]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemv_fp16_kernel(X, W, BIAS, Y, RES, N, K,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     TDIM: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(W + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0.0)
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        acc += tl.dot(w, tl.trans(xs))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


# ----------------------------------------------------------------- split-K path
@triton.jit
def gemv_int4_splitk_kernel(X, WQ, SC, Y, N, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr,
                            SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    offs_kb = tl.arange(0, BLOCK_K // 2)
    n_kblocks = K // BLOCK_K
    for ki in range(sp, n_kblocks, SPLIT_K):
        k0 = ki * BLOCK_K
        b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kb[None, :],
                    mask=mask_n[:, None], other=0)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        bi = b.to(tl.int32)
        sb = s * 8.0
        lo = (bi & 0xF).to(tl.float16) * s[:, None] - sb[:, None]
        hi = ((bi >> 4) & 0xF).to(tl.float16) * s[:, None] - sb[:, None]
        w = tl.interleave(lo, hi)
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        acc += tl.dot(w, tl.trans(xs))
    tl.store(Y + sp * (TDIM * N) + offs_m[None, :] * N + offs_n[:, None], acc,
             mask=mask_n[:, None])


@triton.jit
def gemv_int8_splitk_kernel(X, WQ, SC, Y, N, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr,
                            SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    n_kblocks = K // BLOCK_K
    for ki in range(sp, n_kblocks, SPLIT_K):
        k0 = ki * BLOCK_K
        w = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0).to(tl.float16)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        w = w * s[:, None]
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        acc += tl.dot(w, tl.trans(xs))
    tl.store(Y + sp * (TDIM * N) + offs_m[None, :] * N + offs_n[:, None], acc,
             mask=mask_n[:, None])


@triton.jit
def gemv_epilogue_kernel(Y32, BIAS, Y, RES, N, BLOCK: tl.constexpr,
                         HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr,
                         TDIM: tl.constexpr = 16, SPLIT_K: tl.constexpr = 1):
    m = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    y = tl.zeros((BLOCK,), dtype=tl.float32)
    for sp in tl.static_range(SPLIT_K):
        y += tl.load(Y32 + sp * (TDIM * N) + m * N + n, mask=mask, other=0.0)
    if HAS_BIAS:
        y += tl.load(BIAS + n, mask=mask, other=0.0)
    if HAS_RES:
        y += tl.load(RES + m * N + n, mask=mask, other=0.0)
        tl.store(RES + m * N + n, y.to(tl.float16), mask=mask)
    else:
        tl.store(Y + m * N + n, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------- prefill
@triton.jit
def gemm_int4_kernel(A, WQ, SC, BIAS, Y, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k8 = tl.arange(0, BLOCK_K // 8)
    for k0 in range(0, K, BLOCK_K):
        wp = tl.load(WQ + offs_n[:, None] * (K // 8) + (k0 // 8) + offs_k8[None, :],
                     mask=mask_n[:, None], other=0)
        w = _unpack_int4(wp, BLOCK_N, BLOCK_K)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        w = w * s[:, None]
        a = tl.load(A + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_m[:, None], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gemm_int8_kernel(A, WQ, SC, BIAS, Y, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0).to(tl.float16)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        w = w * s[:, None]
        a = tl.load(A + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_m[:, None], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


# --------------------------------------------------------------------- dispatch
GEMM_WARPS = 4
GEMM_STAGES = 3
# int8 activations + integer tensor cores.  Correct to cos 0.99998 but it changes the
# exact bit pattern of every decode, which the engine's "verify == decode" invariants
# rely on, so it stays opt-in until a batched serving profile wants it.
W8A8_DECODE = False
# int4 weights + int8 activations (integer MMAs).  Three attempts, all neutral
# end-to-end: unfused (255.0 vs 254.9 on 1.5B), batch>=32 only (12,821 vs 12,852),
# and with the activation quantise fused into rmsnorm (253.7 vs 255.5 on 1.5B,
# 157.4 vs 157.9 on 3B).  The kernel is bandwidth bound, so cheaper ALU work buys
# nothing.  The fused path and the pre-quantised gemv arguments stay in place for a
# future kernel that actually needs int8 activations.
SPLITK_STAGES = 2      # stages 3 measured -3.4% (3B) / -2.4% (7B), so 2 stands
W4A8_DECODE = False
W8A8_SWIGLU_DECODE = False   # A/B knob: int8 dots for the decode gate_up


def _cfg(n: int, k: int, kind: str) -> tuple[int, int, int, int]:
    """(block_n, split_k, num_warps, num_stages), tuned end-to-end on RTX 3070.

    int4 uses the interleaved nibble kernel: it reaches 270-370 GB/s on the wide
    shapes and needs split-K on the narrow, deep ones (down_proj).  The int8
    numbers were tuned before that and stay as measured, since split-K inside a
    captured decode step costs more than the bandwidth it buys there.
    """
    if kind == "int4":
        if n >= 8192:
            # measured through the engine with interleaved A/B on 3B mixed
            # (block_n 64 -> 122.4 tok/s, 128 -> 133.9, 256 -> 102.1).  The wider
            # tile needs the longer K to pay off: at k=1536 (1.5B) it costs 8%.
            return (128, 1, 4, 2) if k >= 2048 else (64, 1, 4, 2)
        if k >= 8192:
            return (64, 8, 4, 2)
        if k >= 4096:
            return (64, 4, 4, 2)
        return (64, 4, 4, 2)
    if n >= 65536:
        return (32, 1, 2, 4)
    if n >= 8192:
        # gate_up: wide and deep enough to keep 4 warps busy.
        # 3B (22016x2048) stages 2 -> 141.1us, stages 3 -> 119.8us
        return (64, 1, 4, 3)
    if k >= 4096:
        # down_proj: deep K, narrow N.  Measured (1536x8960) with split 8:
        # block_n 32 -> 54.4us, block_n 64 -> 41.7us
        return (64, 8, 4, 2)
    if k > n:
        # o_proj: K wider than N, one CTA wave already fills the SMs.  Two warps win
        # the isolated microbenchmark (24.7 -> 16.6us) but not the engine, where the
        # next kernel waits on this one and latency, not throughput, is the cost.
        return (32, 1, 4, 3)
    # qkv: N wider than K.
    return (64 if n >= 2048 else 32, 1, 4, 2)


def gemv(x: torch.Tensor, lin: QuantLinear, tdim: int, row_off: int = 0,
         n_override: int | None = None, block_n: int | None = None,
         block_k: int | None = None, out: torch.Tensor | None = None,
         acc: torch.Tensor | None = None, split_k: int | None = None,
         num_warps: int | None = None, add_to: torch.Tensor | None = None,
         xq: torch.Tensor | None = None,
         xs: torch.Tensor | None = None) -> torch.Tensor:
    """x [tdim, K] fp16 -> [tdim, n] fp16 (rows >= batch size are padding).

    row_off / n_override address one expert inside a stacked [E*n, k] tensor.
    """
    n = n_override if n_override is not None else lin.n
    k = lin.k
    packed = lin.packed if row_off == 0 else lin.packed[row_off:]
    scales = lin.scales if (row_off == 0 or lin.scales is None) else lin.scales[row_off:]
    block_k = min(block_k or lin.group, lin.group, k)
    if out is None:
        out = torch.empty((tdim, n), dtype=torch.float16, device=x.device)
    cfg_bn, cfg_sk, cfg_nw, cfg_ns = _cfg(n, k, lin.kind)
    # Round-59 tried replacing the table with per-shape microbenchmark optima
    # (lm_head: bn64/bk128/sk1 measured 2.2x faster in isolation).  End to end it made
    # every model *slower* (0.5B decode 608 -> 509 tok/s), the third time a
    # microbenchmark win failed to transfer, so the engine-tuned table stands.
    # Why the table below is tuned end to end and not from microbenchmarks: an
    # isolated kernel runs with the whole GPU free, but in a decode step every kernel
    # waits on the previous one, so what matters is how fast a kernel fills the SMs.
    # down_proj (n=896) with split_k=1 is 30.4us alone (faster than split_k=4's 33.8us)
    # yet decode drops 608 -> 442 tok/s, because 14 CTAs cannot use the device while
    # 56 can.  Same story for the lm_head tile (2.2x faster alone, neutral in context)
    # and for swiglu's BLOCK_N (5% faster alone, 3% slower in context).
    if lin.kind == "int4" and k >= 4096:
        # down_proj: 8 splits beat 4 by 5% on 0.5B (14 CTAs -> 112, so the device is
        # actually full) and are neutral on 1.5B/3B, which already saturate at 4.
        # 16 splits regress (epilogue cost).  Occupancy, not isolated latency.
        cfg_sk = max(cfg_sk, 8)
    if lin.kind == "fp16" or row_off:
        cfg_sk = 1
    if lin.kind == "fp16" or row_off:
        cfg_sk = 1
    split_k = split_k if split_k is not None else cfg_sk
    block_n = block_n or cfg_bn
    num_warps = num_warps or cfg_nw
    if split_k > 1:
        # deterministic partials: [SPLIT_K, tdim, n] fp32, overwritten every call
        if acc is None:
            key = (tdim, split_k)
            acc = lin.scratch.get(key)
            if acc is None or acc.shape != (split_k, tdim, n):
                acc = torch.zeros((split_k, tdim, n), dtype=torch.float32, device=x.device)
                lin.scratch[key] = acc
        kern = gemv_int4_splitk_kernel if lin.kind == "int4" else gemv_int8_splitk_kernel
        wptr = packed.view(torch.int8) if lin.kind == "int4" else packed
        kern[(triton.cdiv(n, block_n), split_k)](
            x, wptr, scales, acc, n, k, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP=lin.group, TDIM=tdim, SPLIT_K=split_k,
            num_warps=num_warps,
            num_stages=max(cfg_ns, SPLITK_STAGES))
        # 256-wide tiles leave only 4 CTAs for a 896-column output; the reduction is
        # latency bound, not bandwidth bound, so narrower tiles fill the device
        blk = 64 if n <= 2048 else 256
        gemv_epilogue_kernel[(tdim, triton.cdiv(n, blk))](
            acc, lin.bias if lin.bias is not None else out, out,
            add_to if add_to is not None else out, n, BLOCK=blk,
            HAS_BIAS=lin.bias is not None, HAS_RES=add_to is not None,
            TDIM=tdim, SPLIT_K=split_k, num_warps=4)
        return add_to if add_to is not None else out
    grid = (triton.cdiv(n, block_n),)
    bias = lin.bias if lin.bias is not None else out
    target = out if add_to is None else add_to
    res_ptr = add_to if add_to is not None else out
    if lin.kind == "int4" and W4A8_DECODE and (xq is None or xs is None):
        xq, xs = quantize_act(x, lin.group)
    if lin.kind == "int4" and W4A8_DECODE:
        gemv_int4_w4a8_kernel[grid](xq, xs, packed.view(torch.int8), scales, bias, target,
                                    res_ptr, n, k,
                                    BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
                                    XGROUP=lin.group, TDIM=tdim,
                                    HAS_BIAS=lin.bias is not None,
                                    HAS_RES=add_to is not None,
                                    num_warps=num_warps, num_stages=cfg_ns)
    elif lin.kind == "int4":
        gemv_int4_kernel[grid](x, packed.view(torch.int8), scales, bias, target, res_ptr, n, k,
                               BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
                               TDIM=tdim, HAS_BIAS=lin.bias is not None,
                               HAS_RES=add_to is not None,
                               num_warps=num_warps, num_stages=cfg_ns)
    elif lin.kind == "int8" and tdim >= 16 and W8A8_DECODE:
        # at 16+ rows the fp16 dot is compute bound: quantise the activations and use
        # the integer tensor cores instead (the weights are already int8)
        xq, xs = quantize_act(x, 64)
        gemv_int8_w8a8_kernel[grid](xq, xs, packed, scales, bias, target, res_ptr, n, k,
                                    BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
                                    XGROUP=64, TDIM=tdim, HAS_BIAS=lin.bias is not None,
                                    HAS_RES=add_to is not None,
                                    num_warps=num_warps, num_stages=cfg_ns)
    elif lin.kind == "int8":
        gemv_int8_kernel[grid](x, packed, scales, bias, target, res_ptr, n, k,
                               BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
                               SUB=max(1, lin.group // block_k),
                               TDIM=tdim, HAS_BIAS=lin.bias is not None,
                               HAS_RES=add_to is not None,
                               num_warps=num_warps, num_stages=cfg_ns)
    else:
        gemv_fp16_kernel[grid](x, packed, bias, target, res_ptr, n, k,
                               BLOCK_N=block_n, BLOCK_K=block_k,
                               TDIM=tdim, HAS_BIAS=lin.bias is not None,
                               HAS_RES=add_to is not None,
                               num_warps=num_warps, num_stages=cfg_ns)
    return out


def gemm(a: torch.Tensor, lin: QuantLinear, block_m: int = 64, block_n: int | None = None,
         block_k: int = 64, row_off: int = 0, n_override: int | None = None,
         aq: torch.Tensor | None = None, as_: torch.Tensor | None = None) -> torch.Tensor:
    """Row offset / output-count override let one stacked tensor hold every expert."""
    m, k = a.shape
    n = n_override if n_override is not None else lin.n
    if block_n is None:
        block_n = 128 if n >= 4096 else 64
    if lin.kind == "fp16":
        wv = lin.packed[row_off:row_off + n]
        out = torch.nn.functional.linear(a, wv)
        return out + lin.bias if lin.bias is not None else out
    if lin.kind == "int4" and row_off == 0 and m >= 64:
        # prefill is compute bound (~34 TFLOPS on the fp16 path); the integer tensor
        # cores reach ~47 TFLOPS, measured 25-40% faster per GEMM.  The caller may pass
        # activations already quantised (fused into the producer).
        # BLOCK_K must equal one quantisation group, otherwise a single scale is applied
        # to two groups' worth of weights (group=64 gave KL 11.6 instead of 0.09).
        if aq is None or as_ is None:
            aq, as_ = quantize_act(a, lin.group)
        out = torch.empty((m, n), dtype=torch.float16, device=a.device)
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        gemm_int4_w8a8_kernel[grid](aq, as_, lin.packed.view(torch.int8), lin.scales,
                                    lin.bias if lin.bias is not None else out, out,
                                    m, n, k, BLOCK_M=block_m, BLOCK_N=block_n,
                                    BLOCK_K=lin.group, GROUP=lin.group, XGROUP=lin.group,
                                    HAS_BIAS=lin.bias is not None,
                                    num_warps=GEMM_WARPS, num_stages=GEMM_STAGES)
        return out
    packed = lin.packed if row_off == 0 else lin.packed[row_off:]
    scales = lin.scales if (row_off == 0 or lin.scales is None) else lin.scales[row_off:]
    if lin.group % block_k != 0:
        block_k = min(block_k, lin.group)
    out = torch.empty((m, n), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias = lin.bias if lin.bias is not None else out
    kernel = gemm_int4_kernel if lin.kind == "int4" else gemm_int8_kernel
    kernel[grid](a, packed, scales, bias, out, m, n, k,
                 BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                 GROUP=lin.group, HAS_BIAS=lin.bias is not None,
                 num_warps=GEMM_WARPS, num_stages=GEMM_STAGES)
    return out


def quant_linear(x: torch.Tensor, lin: QuantLinear, tdim: int) -> torch.Tensor:
    if x.shape[0] <= 16:
        return gemv(x, lin, 16)
    return gemm(x, lin)


# ------------------------------------------------------------------ swiglu
@triton.jit
def gemv_int8_swiglu_kernel(X, WQ, SC, Y, NH, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr):
    """gate_up GEMV that writes silu(gate) * up directly.

    NH is half of N: rows [0, NH) are the gate, [NH, 2*NH) the up projection, so
    one program computes both halves of its output rows and the activation never
    round-trips through memory.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < NH
    offs_m = tl.arange(0, TDIM)
    acc_g = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        gk = (k0 // GROUP)
        s_g = tl.load(SC + offs_n * (K // GROUP) + gk, mask=mask_n, other=0.0)
        s_u = tl.load(SC + (offs_n + NH) * (K // GROUP) + gk, mask=mask_n, other=0.0)
        wg = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                     mask=mask_n[:, None], other=0).to(tl.float16) * s_g[:, None]
        wu = tl.load(WQ + (offs_n + NH)[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                     mask=mask_n[:, None], other=0).to(tl.float16) * s_u[:, None]
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        xt = tl.trans(xs)
        acc_g += tl.dot(wg, xt)
        acc_u += tl.dot(wu, xt)
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[None, :] * NH + offs_n[:, None], act.to(tl.float16), mask=mask_n[:, None])


@triton.jit
def gemv_int4_swiglu_kernel(X, WB, SC, Y, NH, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr):
    """int4 gate_up GEMV that writes silu(gate) * up directly (interleaved unpack)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < NH
    offs_m = tl.arange(0, TDIM)
    offs_kb = tl.arange(0, BLOCK_K // 2)
    acc_g = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        gk = k0 // GROUP
        kb0 = k0 // 2
        s_g = tl.load(SC + offs_n * (K // GROUP) + gk, mask=mask_n, other=0.0)
        s_u = tl.load(SC + (offs_n + NH) * (K // GROUP) + gk, mask=mask_n, other=0.0)
        bg = tl.load(WB + offs_n[:, None] * (K // 2) + kb0 + offs_kb[None, :],
                     mask=mask_n[:, None], other=0)
        bu = tl.load(WB + (offs_n + NH)[:, None] * (K // 2) + kb0 + offs_kb[None, :],
                     mask=mask_n[:, None], other=0)
        gi = bg.to(tl.int32)
        ui = bu.to(tl.int32)
        sbg = s_g * 8.0
        sbu = s_u * 8.0
        wg = tl.interleave((gi & 0xF).to(tl.float16) * s_g[:, None] - sbg[:, None],
                           ((gi >> 4) & 0xF).to(tl.float16) * s_g[:, None] - sbg[:, None])
        wu = tl.interleave((ui & 0xF).to(tl.float16) * s_u[:, None] - sbu[:, None],
                           ((ui >> 4) & 0xF).to(tl.float16) * s_u[:, None] - sbu[:, None])
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        xt = tl.trans(xs)
        acc_g += tl.dot(wg, xt)
        acc_u += tl.dot(wu, xt)
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[None, :] * NH + offs_n[:, None], act.to(tl.float16), mask=mask_n[:, None])


@triton.jit
def gemm_swiglu_kernel(A, WQ, SC, Y, M, NH, K,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       GROUP: tl.constexpr, IS_INT4: tl.constexpr):
    """gate_up GEMM that writes silu(gate) * up, so the prefill never round-trips
    the 2*intermediate activation through memory (half a layer's bytes)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < NH
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_kb = tl.arange(0, BLOCK_K // 2)
    for k0 in range(0, K, BLOCK_K):
        if IS_INT4:
            kb = k0 // 2
            bg = tl.load(WQ + offs_n[:, None] * (K // 2) + kb + offs_kb[None, :],
                         mask=mask_n[:, None], other=0)
            bu = tl.load(WQ + (offs_n + NH)[:, None] * (K // 2) + kb + offs_kb[None, :],
                         mask=mask_n[:, None], other=0)
            sg = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            su = tl.load(SC + (offs_n + NH) * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            gi = bg.to(tl.int32)
            ui = bu.to(tl.int32)
            sbg = sg * 8.0
            sbu = su * 8.0
            wg = tl.interleave((gi & 0xF).to(tl.float16) * sg[:, None] - sbg[:, None],
                               ((gi >> 4) & 0xF).to(tl.float16) * sg[:, None] - sbg[:, None])
            wu = tl.interleave((ui & 0xF).to(tl.float16) * su[:, None] - sbu[:, None],
                               ((ui >> 4) & 0xF).to(tl.float16) * su[:, None] - sbu[:, None])
        else:
            wg = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                         mask=mask_n[:, None], other=0).to(tl.float16)
            wu = tl.load(WQ + (offs_n + NH)[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                         mask=mask_n[:, None], other=0).to(tl.float16)
            sg = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            su = tl.load(SC + (offs_n + NH) * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            wg = wg * sg[:, None]
            wu = wu * su[:, None]
        a = tl.load(A + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_m[:, None], other=0.0)
        acc_g += tl.dot(a, tl.trans(wg))
        acc_u += tl.dot(a, tl.trans(wu))
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[:, None] * NH + offs_n[None, :], act.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


def gemm_swiglu(a: torch.Tensor, lin: QuantLinear, nh: int,
                block_m: int = 64, block_n: int = 64, block_k: int = 64,
                out: torch.Tensor | None = None,
                aq: torch.Tensor | None = None,
                as_: torch.Tensor | None = None) -> torch.Tensor:
    """a [M, K] -> act [M, nh] with act = silu(gate) * up (gate_up layout)."""
    m, k = a.shape
    if lin.kind == "int4" and m >= 64:
        # fused gate_up on integer tensor cores.  BLOCK_M=32 is the whole trick: with
        # 64 the two int8 dots plus the nibble unpack spill registers and the kernel
        # runs 40% *slower* than fp16; at 32/64 it reaches 50 TFLOPS, +47% over fp16.
        if aq is None or as_ is None:
            aq, as_ = quantize_act(a, lin.group)
        nh = lin.n // 2
        out = torch.empty((m, nh), dtype=torch.float16, device=a.device)
        # bn=128 wins in the microbench (+5% at M=1081) but loses end-to-end at both
        # 0.5B and 3B, so the 64-wide tile stays
        bn = 64
        grid = (triton.cdiv(m, 32), triton.cdiv(nh, bn))
        gemm_swiglu_w8a8_kernel[grid](aq, as_, lin.packed.view(torch.int8), lin.scales, out,
                                      m, nh, k, BLOCK_M=32, BLOCK_N=bn, BLOCK_K=lin.group,
                                      GROUP=lin.group, XGROUP=lin.group,
                                      num_warps=4, num_stages=2)
        return out
    if out is None:
        out = torch.empty((m, nh), dtype=torch.float16, device=a.device)
    if lin.kind == "fp16":
        gu = a @ lin.packed.t()
        g, u = gu[:, :nh], gu[:, nh:]
        return (g * torch.sigmoid(g) * u).to(torch.float16)
    block_k = min(block_k, lin.group, k)
    w = lin.packed if lin.kind == "int8" else lin.packed.view(torch.int8)
    gemm_swiglu_kernel[(triton.cdiv(m, block_m), triton.cdiv(nh, block_n))](
        a, w, lin.scales, out, m, nh, k,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
        IS_INT4=(lin.kind == "int4"), num_warps=4, num_stages=3)
    return out


def gemv_swiglu(x: torch.Tensor, lin: QuantLinear, tdim: int, nh: int,
                block_n: int = 64, block_k: int | None = None,
                out: torch.Tensor | None = None, num_warps: int = 4,
                num_stages: int = 2, xq: torch.Tensor | None = None,
                xs: torch.Tensor | None = None) -> torch.Tensor:
    """x [tdim, K] -> act [tdim, nh] with act = silu(gate) * up."""
    k = lin.k
    block_k = min(block_k or lin.group, lin.group, k)
    if out is None:
        out = torch.empty((tdim, nh), dtype=torch.float16, device=x.device)
    if lin.kind == "int8":
        gemv_int8_swiglu_kernel[(triton.cdiv(nh, block_n),)](
            x, lin.packed, lin.scales, out, nh, k, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP=lin.group, TDIM=tdim, num_warps=num_warps, num_stages=num_stages)
        return out
    if lin.kind == "int4" and xq is not None and xs is not None:
        gemv_int4_swiglu_w8a8_kernel[(triton.cdiv(nh, 64),)](
            xq, xs, lin.packed.view(torch.int8), lin.scales, out, nh, k,
            BLOCK_N=64, BLOCK_K=lin.group, GROUP=lin.group, XGROUP=lin.group,
            TDIM=tdim, num_warps=4, num_stages=2)
        return out
    if lin.kind == "int4":
        cfg_bn, _, cfg_nw, cfg_ns = _cfg(2 * nh, k, lin.kind)
        # each iteration loads only BLOCK_N x 64 words (the 128-element quantisation
        # group caps BLOCK_K), so deeper pipelining is the only way to keep more loads
        # in flight.  Measured: 7B (nh=14336) 62.0 -> 64.1 tok/s with stages 4;
        # 0.5B (nh=4864) 640 -> 601, so only wide gate_up tensors get the deeper pipe.
        cfg_ns = 4 if nh >= 14000 else 2
        # bn=32 for small nh (more CTAs) measured -11% end to end: every CTA re-reads
        # the whole activation row, so halving the tile doubles that traffic.
        # bn=128 is measured per model below (7B has enough columns to fill it).
        gemv_int4_swiglu_kernel[(triton.cdiv(nh, cfg_bn),)](
            x, lin.packed.view(torch.int8), lin.scales, out, nh, k, BLOCK_N=cfg_bn,
            BLOCK_K=block_k, GROUP=lin.group, TDIM=tdim, num_warps=cfg_nw,
            num_stages=cfg_ns)
        return out
    # fallback: two GEMVs + silu
    gate = gemv(x, lin, tdim)
    g = gate[:, :nh].contiguous()
    u = gate[:, nh:].contiguous()
    from kaze.kernels.act import silu_mul
    return silu_mul(g, u, out=out)


@triton.jit
def moe_router_kernel(X, R, IDX, W, E, H, K,
                      BLOCK_H: tl.constexpr, BLOCK_E: tl.constexpr, TDIM: tl.constexpr,
                      NORM_TOPK: tl.constexpr):
    """Router logits + top-k in one pass, writing indices and (normalised) weights.

    Everything is preallocated so the MoE decode step can be captured in a CUDA
    graph; torch.topk would allocate and break the capture.
    """
    row = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    x = tl.load(X + row * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    acc = tl.zeros((BLOCK_E,), tl.float32)
    for h0 in range(0, H, BLOCK_H):
        r = tl.load(R + offs_e[:, None] * H + h0 + offs_h[None, :],
                    mask=mask_e[:, None] & (h0 + offs_h)[None, :] < H, other=0.0).to(tl.float32)
        xv = tl.load(X + row * H + h0 + offs_h, mask=(h0 + offs_h) < H, other=0.0).to(tl.float32)
        acc += tl.sum(r * xv[None, :], axis=1)
    logits = tl.where(mask_e, acc, -1e30)
    total = 0.0
    for j in range(K):
        best = tl.max(logits, axis=0)
        bidx = tl.argmax(logits, axis=0)
        tl.store(IDX + row * K + j, bidx.to(tl.int32))
        tl.store(W + row * K + j, best)
        logits = tl.where(offs_e == bidx, -1e30, logits)
        total += best
    if NORM_TOPK:
        for j in range(K):
            v = tl.load(W + row * K + j)
            tl.store(W + row * K + j, v / total)


# ------------------------------------------------------------------ MoE
@triton.jit
def moe_gemv_int8_kernel(X, WQ, SC, IDX, OUT, N, K, NEXP,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         GROUP: tl.constexpr, PER_ROW: tl.constexpr,
                         SAME_ROWS: tl.constexpr, IS_FP16: tl.constexpr = False):
    """One program computes BLOCK_N outputs of one (token, expert) pair.

    MoE decode is bandwidth bound: only the selected experts are read, so the
    cost per token is top_k * expert size, and an FMA reduction is enough -- the
    tensor cores would be idle anyway at batch 1.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(IDX + pid_m).to(tl.int64)
    if SAME_ROWS:
        x_row = pid_m
    else:
        x_row = pid_m // PER_ROW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), tl.float32)
    base = e * N * K
    sbase = e * N * (K // GROUP)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(WQ + base + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0).to(tl.float16)
        if not IS_FP16:
            s = tl.load(SC + sbase + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            w = w * s[:, None]
        xv = tl.load(X + x_row * K + k0 + tl.arange(0, BLOCK_K))
        acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)
    tl.store(OUT + pid_m * N + offs_n, acc.to(tl.float16), mask=mask_n)


def moe_gemv(x: torch.Tensor, lin: QuantLinear, idx: torch.Tensor, n_exp: int,
             n_out: int, same_rows: bool = False, per_row: int = 1,
             block_n: int = 128, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [rows, K] with idx [rows*per_row] expert ids -> [rows*per_row, n_out]."""
    pairs = idx.numel()
    if out is None:
        out = torch.empty((pairs, n_out), dtype=torch.float16, device=x.device)
    moe_gemv_int8_kernel[(pairs, triton.cdiv(n_out, block_n))](
        x, lin.packed, lin.scales if lin.scales is not None else lin.packed, idx, out, n_out,
        lin.k, n_exp, BLOCK_N=block_n, BLOCK_K=min(lin.group, 128, lin.k), GROUP=lin.group,
        PER_ROW=per_row, SAME_ROWS=same_rows, IS_FP16=(lin.kind == "fp16"),
        num_warps=2, num_stages=2)
    return out

def moe_router(x: torch.Tensor, router: torch.Tensor, k: int, n_exp: int,
               norm_topk: bool, idx: torch.Tensor, w: torch.Tensor) -> None:
    """x [tdim, H] -> idx/w [tdim*k] (in place, no allocations)."""
    tdim, h = x.shape
    moe_router_kernel[(tdim,)](
        x, router, idx, w, n_exp, h, k,
        BLOCK_H=triton.next_power_of_2(min(h, 1024)),
        BLOCK_E=triton.next_power_of_2(n_exp), TDIM=tdim,
        NORM_TOPK=norm_topk, num_warps=4)


@triton.jit
def quant_act_kernel(X, Q, SC, M, K, BLOCK_K: tl.constexpr, TDIM: tl.constexpr):
    """Per-(row, K-block) int8 quantisation of the activations for W8A8 GEMV."""
    pid = tl.program_id(0)          # row
    pidk = tl.program_id(1)         # k block
    offs_k = pidk * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(X + pid * K + offs_k, mask=mask, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)), 1e-8) / 127.0
    v = x / s
    q = (v + tl.where(v >= 0, 0.5, -0.5)).to(tl.int8)
    tl.store(Q + pid * K + offs_k, q, mask=mask)
    tl.store(SC + pid * tl.cdiv(K, BLOCK_K) + pidk, s)


def quantize_act(x: torch.Tensor, group: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    sc = torch.empty((m, (k + group - 1) // group), dtype=torch.float32, device=x.device)
    quant_act_kernel[(m, (k + group - 1) // group)](x, q, sc, m, k,
                                                    BLOCK_K=group, TDIM=m, num_warps=1)
    return q, sc


@triton.jit
def gemv_int8_w8a8_kernel(XQ, XS, WQ, SC, BIAS, Y, RES, N, K,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          GROUP: tl.constexpr, XGROUP: tl.constexpr,
                          TDIM: tl.constexpr, HAS_BIAS: tl.constexpr,
                          HAS_RES: tl.constexpr):
    """Decode GEMV with int8 weights *and* int8 activations (integer MMA)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0).to(tl.int8)
        xq = tl.load(XQ + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        dots = tl.dot(w, tl.trans(xq), out_dtype=tl.int32).to(tl.float32)
        ws = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        xs = tl.load(XS + offs_m * (K // XGROUP) + (k0 // XGROUP))
        acc += dots * ws[:, None] * xs[None, :]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemv_int4_planar_kernel(X, WQ, SC, BIAS, Y, RES, N, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr,
                            HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr):
    """int4 GEMV without the nibble interleave.

    Each byte of the weights holds the values for K indices (2j, 2j+1), so the low
    and high nibbles can drive two dots against the even and odd K lanes of the
    activation.  That removes the tl.interleave (which cost ~25% of this kernel) and
    keeps the memory pattern identical.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + tl.arange(0, half)[None, :],
                    mask=mask_n[:, None], other=0)
        bi = b.to(tl.int32)
        lo = (bi & 0xF).to(tl.float16)
        hi = ((bi >> 4) & 0xF).to(tl.float16)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        sb = s * 8.0
        w_lo = lo * s[:, None] - sb[:, None]
        w_hi = hi * s[:, None] - sb[:, None]
        offs_e = k0 + tl.arange(0, half) * 2
        offs_o = offs_e + 1
        xe = tl.load(X + offs_m[:, None] * K + offs_e[None, :])
        xo = tl.load(X + offs_m[:, None] * K + offs_o[None, :])
        acc += tl.dot(w_lo, tl.trans(xe))
        acc += tl.dot(w_hi, tl.trans(xo))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemv_int4_w4a8_kernel(XQ, XS, WQ, SC, BIAS, Y, RES, N, K,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          GROUP: tl.constexpr, XGROUP: tl.constexpr,
                          TDIM: tl.constexpr, HAS_BIAS: tl.constexpr,
                          HAS_RES: tl.constexpr):
    """int4 weights with int8 activations: nibbles unpack to int8 and feed the integer
    MMAs, so the weights never touch fp16 and the scale is applied once per group."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + tl.arange(0, half)[None, :],
                    mask=mask_n[:, None], other=0)
        bi = b.to(tl.int32)
        lo = (bi & 0xF).to(tl.int8) - 8
        hi = ((bi >> 4) & 0xF).to(tl.int8) - 8
        w_i8 = tl.interleave(lo, hi)
        xq = tl.load(XQ + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        dots = tl.dot(w_i8, tl.trans(xq), out_dtype=tl.int32).to(tl.float32)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        xs = tl.load(XS + offs_m * (K // XGROUP) + (k0 // XGROUP))
        acc += dots * s[:, None] * xs[None, :]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemm_int4_w8a8_kernel(AQ, AS, WQ, SC, BIAS, Y, M, N, K,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                          XGROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Prefill GEMM on integer tensor cores: int4 weights, int8 activations.

    Prefill is compute bound (measured ~70% of the fp16 tensor peak), and the integer
    path doubles the math throughput.  Activations are quantised per row per XGROUP
    lanes outside the kernel; the scales are applied to the int32 accumulator, so the
    inner loop is pure integer MMA plus the nibble unpack.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        aq = tl.load(AQ + offs_m[:, None] * K + k0 + offs_k[None, :],
                     mask=mask_m[:, None], other=0).to(tl.int8)
        b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + tl.arange(0, half)[None, :],
                    mask=mask_n[:, None], other=0)
        bi = b.to(tl.int32)
        lo = (bi & 0xF).to(tl.int8) - 8
        hi = ((bi >> 4) & 0xF).to(tl.int8) - 8
        w_i8 = tl.interleave(lo, hi)
        dots = tl.dot(aq, tl.trans(w_i8), out_dtype=tl.int32).to(tl.float32)
        sw = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        sa = tl.load(AS + offs_m * (K // XGROUP) + (k0 // XGROUP), mask=mask_m, other=0.0)
        acc += dots * sa[:, None] * sw[None, :]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=mask_m[:, None])


@triton.jit
def gemm_swiglu_w8a8_kernel(AQ, AS, WQ, SC, Y, M, NH, K,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                            XGROUP: tl.constexpr):
    """Fused gate_up + silu on integer tensor cores (prefill, int4 weights)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < NH
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        aq = tl.load(AQ + offs_m[:, None] * K + k0 + offs_k[None, :],
                     mask=mask_m[:, None], other=0).to(tl.int8)
        sa = tl.load(AS + offs_m * (K // XGROUP) + (k0 // XGROUP), mask=mask_m, other=0.0)
        bg = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + tl.arange(0, half)[None, :],
                     mask=mask_n[:, None], other=0)
        bu = tl.load(WQ + (offs_n + NH)[:, None] * (K // 2) + (k0 // 2)
                     + tl.arange(0, half)[None, :], mask=mask_n[:, None], other=0)
        gi = bg.to(tl.int32)
        ui = bu.to(tl.int32)
        wg = tl.interleave((gi & 0xF).to(tl.int8) - 8, ((gi >> 4) & 0xF).to(tl.int8) - 8)
        wu = tl.interleave((ui & 0xF).to(tl.int8) - 8, ((ui >> 4) & 0xF).to(tl.int8) - 8)
        sg = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        su = tl.load(SC + (offs_n + NH) * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        acc_g += tl.dot(aq, tl.trans(wg), out_dtype=tl.int32).to(tl.float32) * sa[:, None] * sg[None, :]
        acc_u += tl.dot(aq, tl.trans(wu), out_dtype=tl.int32).to(tl.float32) * sa[:, None] * su[None, :]
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[:, None] * NH + offs_n[None, :], act.to(tl.float16), mask=mask_m[:, None])


@triton.jit
def gemv_int4_swiglu_w8a8_kernel(XQ, XS, WQ, SC, Y, NH, K,
                                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                 GROUP: tl.constexpr, XGROUP: tl.constexpr,
                                 TDIM: tl.constexpr):
    """Decode gate_up on integer tensor cores.

    The fp16 variant spends its time converting two nibble tiles per iteration and
    keeping them in registers; int8 tiles are half the size and feed tl.dot directly.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < NH
    offs_m = tl.arange(0, TDIM)
    acc_g = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        gk = k0 // GROUP
        kb0 = k0 // 2
        bg = tl.load(WQ + offs_n[:, None] * (K // 2) + kb0 + tl.arange(0, half)[None, :],
                     mask=mask_n[:, None], other=0)
        bu = tl.load(WQ + (offs_n + NH)[:, None] * (K // 2) + kb0
                     + tl.arange(0, half)[None, :], mask=mask_n[:, None], other=0)
        gi = bg.to(tl.int32)
        ui = bu.to(tl.int32)
        wg = tl.interleave((gi & 0xF).to(tl.int8) - 8, ((gi >> 4) & 0xF).to(tl.int8) - 8)
        wu = tl.interleave((ui & 0xF).to(tl.int8) - 8, ((ui >> 4) & 0xF).to(tl.int8) - 8)
        xq = tl.load(XQ + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        sg = tl.load(SC + offs_n * (K // GROUP) + gk, mask=mask_n, other=0.0)
        su = tl.load(SC + (offs_n + NH) * (K // GROUP) + gk, mask=mask_n, other=0.0)
        xs = tl.load(XS + offs_m * (K // XGROUP) + (k0 // XGROUP))
        dg = tl.dot(wg, tl.trans(xq), out_dtype=tl.int32).to(tl.float32)
        du = tl.dot(wu, tl.trans(xq), out_dtype=tl.int32).to(tl.float32)
        acc_g += dg * sg[:, None] * xs[None, :]
        acc_u += du * su[:, None] * xs[None, :]
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[None, :] * NH + offs_n[:, None], act.to(tl.float16),
             mask=mask_n[:, None])
