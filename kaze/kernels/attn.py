"""Attention kernels.

Prefill: SDPA over a gathered contiguous prefix (flash/cuDNN backends).
Decode: a GQA-aware paged kernel with flash-decoding split-K.  Every paged
program loads each K/V element once and reuses it for all query heads of the
group, which is what keeps decode at the bandwidth roofline instead of
re-reading the cache once per query head.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

NUM_SPLITS = 8

# Prefill flash-attention tile (block_m, block_n, num_warps, num_stages).  Measured end to
# end on 0.5B int4 prefill; KAZE_ATTN_PREFILL_TILE overrides it for A/B sweeps.
_ATTN_PREFILL_TILE = (lambda raw: tuple(int(v) for v in raw.split(",")) if raw else
                      (64, 64, 4, 3))(os.environ.get("KAZE_ATTN_PREFILL_TILE"))

# Long-context decode attention: (block_n, num_warps, num_stages) for the split-K partial
# kernel.  num_stages=1 was leaving the K/V loads unpipelined, which caps the kernel at
# ~40% of HBM peak once the context is long enough to make it the bottleneck.
# Long-context decode attention: (block_n, num_warps, num_stages).  Every decision here is
# in-process A/B only (bench/attn_long_ab.py): cross-run numbers at 4K+ drift +-25% on this
# box.  Two changes, measured against the old (64, 4, 1):
#   stages 1 -> 2: 0.5B 4K +4%, 8K +6%, 16K +10%; 1.5B 16K +5%  (pipelined K/V loads)
#   block_n 64 -> 128 at stages 2: 0.5B 4K +3.6%, 8K +10%, 16K +15%; 1.5B 16K +11%
# 0.5B 16K ends at 359.7 -> 463.9 tok/s, 1.5B 16K 187.7 -> 223.5.  Splits beyond 24 add
# nothing once the loads are pipelined, and (128, 8, 3) or (32, 4, 2) are both slower.
_PARTIAL_TILE = (lambda raw: tuple(int(v) for v in raw.split(",")) if raw else
                 (128, 4, 3))(os.environ.get("KAZE_ATTN_PARTIAL_TILE"))

# Short-context (single-pass) decode attention: (block_n, num_warps, num_stages).  Swept
# kernel-level in bench/direct_tile_ab.py; override with KAZE_ATTN_DIRECT_TILE.
# The stages choice is batch dependent, measured kernel-level on 0.5B:
#   batch 16 ctx 120: stages 3 is +28.6% (occupancy: 16 rows x 128 cols x 3 stages)
#   batch 32 ctx 96/120: stages 3 is -4.0% / -6.1%
# so the default tile stays (128, 4, 1) and the wrapper bumps stages to 3 at b >= 32.
_DIRECT_TILE = (lambda raw: tuple(int(v) for v in raw.split(",")) if raw else
                (128, 4, 1))(os.environ.get("KAZE_ATTN_DIRECT_TILE"))


@triton.jit
def attn_decode_partial_kernel(Q, KT, VT, BT, SEQ, ACC, ML,
                               stride_qb, stride_kb, stride_kt, stride_bt,
                               stride_acc_b, stride_acc_h, stride_acc_s,
                               stride_ml_b, stride_ml_h, stride_ml_s,
                               scale,
                               BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                               GQA: tl.constexpr, TDIM: tl.constexpr,
                               BLOCK_N: tl.constexpr, NUM_SPLITS: tl.constexpr):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    sp = tl.program_id(2)
    seqlen = tl.load(SEQ + b)
    offs_j = tl.arange(0, TDIM)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_qb + (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=(offs_j < GQA)[:, None], other=0.0)
    n_blocks = tl.cdiv(seqlen, BLOCK_N)
    m_i = tl.full((TDIM,), -1e30, tl.float32)
    l_i = tl.zeros((TDIM,), tl.float32)
    acc = tl.zeros((TDIM, HEAD_DIM), tl.float32)
    for bi in range(sp, n_blocks, NUM_SPLITS):
        idx = bi * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = idx < seqlen
        blk = tl.load(BT + b * stride_bt + idx // BLOCK_SIZE, mask=mask, other=0).to(tl.int64)
        off = idx % BLOCK_SIZE
        ptr = blk[:, None] * stride_kb + off[:, None] * stride_kt + kvh * HEAD_DIM + offs_d[None, :]
        k = tl.load(KT + ptr, mask=mask[:, None], other=0.0).to(tl.float16)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(mask[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(VT + ptr, mask=mask[:, None], other=0.0).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    base = b * stride_acc_b + kvh * stride_acc_h + sp * stride_acc_s
    tl.store(ACC + base + offs_j[:, None] * HEAD_DIM + offs_d[None, :], acc)
    ml_base = b * stride_ml_b + kvh * stride_ml_h + sp * stride_ml_s
    tl.store(ML + ml_base + offs_j, m_i)
    tl.store(ML + ml_base + TDIM + offs_j, l_i)


@triton.jit
def attn_decode_combine_kernel(ACC, ML, OUT,
                               stride_acc_b, stride_acc_h, stride_acc_s,
                               stride_ml_b, stride_ml_h, stride_ml_s,
                               stride_ob,
                               HEAD_DIM: tl.constexpr, GQA: tl.constexpr,
                               TDIM: tl.constexpr, NUM_SPLITS: tl.constexpr):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    offs_j = tl.arange(0, TDIM)
    offs_d = tl.arange(0, HEAD_DIM)
    ml_base = b * stride_ml_b + kvh * stride_ml_h
    acc_base = b * stride_acc_b + kvh * stride_acc_h
    # Three sequential passes over the splits instead of one [S, TDIM, D] tile: with the
    # split count raised to fill the SMs (24 for a 2-KV-head model) that tile would be
    # 24*8*128*4 = 98 KB of fp32 in registers for one CTA.  Ascending sp keeps the sum
    # order fixed, so decode stays exact and reproducible.
    m_max = tl.full((TDIM,), -1e30, tl.float32)
    for sp in range(0, NUM_SPLITS):
        m_max = tl.maximum(m_max, tl.load(ML + ml_base + sp * stride_ml_s + offs_j))
    l_tot = tl.zeros((TDIM,), tl.float32)
    for sp in range(0, NUM_SPLITS):
        m = tl.load(ML + ml_base + sp * stride_ml_s + offs_j)
        l = tl.load(ML + ml_base + sp * stride_ml_s + TDIM + offs_j)
        l_tot += l * tl.exp(m - m_max)
    acc = tl.zeros((TDIM, HEAD_DIM), tl.float32)
    for sp in range(0, NUM_SPLITS):
        m = tl.load(ML + ml_base + sp * stride_ml_s + offs_j)
        a = tl.load(ACC + acc_base + sp * stride_acc_s
                    + offs_j[:, None] * HEAD_DIM + offs_d[None, :])
        acc += a * tl.exp(m - m_max)[:, None]
    acc = acc / l_tot[:, None]
    out_ptr = OUT + b * stride_ob + (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptr, acc.to(tl.float16), mask=(offs_j < GQA)[:, None])


def attn_decode_old(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                block_tables: torch.Tensor, seq_lens: torch.Tensor,
                block_size: int, gqa: int, head_dim: int,
                out: torch.Tensor, acc_buf: torch.Tensor, ml_buf: torch.Tensor,
                scale: float = 1.0, block_n: int = 128,
                num_splits: int = NUM_SPLITS) -> torch.Tensor:
    """q [B, Hq, D] fp16 -> out [B, Hq, D].

    ACC [B, KVH, SPLITS, TDIM, D] fp32, ML [B, KVH, SPLITS, 2, TDIM] fp32.
    """
    b, hq, d = q.shape
    kvh = hq // gqa
    # ACC is [B, KVH, SPLITS, TDIM, D] and TDIM is the per-KV-head query group (GQA),
    # already rounded up to 16.  Reading dim 2 here returned the split count, which
    # happened to be >= GQA for the models measured so far but is the wrong quantity.
    tdim = acc_buf.shape[3]
    grid = (b, kvh, num_splits)
    attn_decode_partial_kernel[grid](
        q, k_cache, v_cache, block_tables, seq_lens, acc_buf, ml_buf,
        q.stride(0), k_cache.stride(0), k_cache.stride(1), block_tables.stride(0),
        acc_buf.stride(0), acc_buf.stride(1), acc_buf.stride(2),
        ml_buf.stride(0), ml_buf.stride(1), ml_buf.stride(2),
        scale,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa, TDIM=tdim,
        BLOCK_N=_PARTIAL_TILE[0], NUM_SPLITS=num_splits,
        num_warps=_PARTIAL_TILE[1], num_stages=_PARTIAL_TILE[2],
    )
    attn_decode_combine_kernel[(b, kvh)](
        acc_buf, ml_buf, out,
        acc_buf.stride(0), acc_buf.stride(1), acc_buf.stride(2),
        ml_buf.stride(0), ml_buf.stride(1), ml_buf.stride(2),
        out.stride(0),
        HEAD_DIM=head_dim, GQA=gqa, TDIM=tdim, NUM_SPLITS=num_splits, num_warps=4,
    )
    return out

def attn_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                block_tables: torch.Tensor, seq_lens: torch.Tensor,
                block_size: int, gqa: int, head_dim: int,
                out: torch.Tensor, acc_buf: torch.Tensor | None = None,
                ml_buf: torch.Tensor | None = None, scale: float = 1.0,
                block_n: int = 128, num_splits: int = NUM_SPLITS,
                mode: str = "long", k_scale: torch.Tensor | None = None,
                v_scale: torch.Tensor | None = None) -> torch.Tensor:
    """mode='short' uses the single-pass kernel (no combine, no partials)."""
    b, hq, d = q.shape
    if mode == "short":
        kvh = hq // gqa
        q8k = k_cache.dtype == torch.int8
        q8v = v_cache.dtype == torch.int8
        ns = ((k_scale if q8k else v_scale).shape[-1] if (q8k or q8v) else 1)
        stages = _DIRECT_TILE[2]
        if b >= 32 and stages < 3:
            stages = 3
        attn_decode_direct_kernel[(b, kvh)](
            q, k_cache, v_cache, k_scale if q8k else q, v_scale if q8v else q,
            block_tables, seq_lens, out,
            q.stride(0), k_cache.stride(0), k_cache.stride(1), block_tables.stride(0),
            out.stride(0), scale, kvh,
            BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa, TDIM=max(16, triton.next_power_of_2(gqa)),
            BLOCK_N=_DIRECT_TILE[0], Q8K=q8k, Q8V=q8v, NS=ns, LANE=32,
            num_warps=_DIRECT_TILE[1], num_stages=stages)
        return out
    return attn_decode_old(q, k_cache, v_cache, block_tables, seq_lens, block_size,
                           gqa, head_dim, out, acc_buf, ml_buf, scale=scale,
                           block_n=block_n, num_splits=num_splits)



@triton.jit
def attn_prefill_paged_kernel(Q, KC, VC, KS, VS, BT, OUT, QS, SL,
                              stride_qm, stride_qh, stride_kb, stride_kt,
                              stride_om, stride_oh, NKV,
                              BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                              GQA: tl.constexpr, BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr, scale, Q8K: tl.constexpr = False,
                              Q8V: tl.constexpr = False,
                              NS: tl.constexpr = 1, LANE: tl.constexpr = 32):
    """Causal flash attention over a paged prefix.

    Query row i is the token at global position q_start + i and attends to keys
    0 .. q_start + i.  One program handles BLOCK_M queries of one head, so the
    KV stream is re-read per query head (fine: prefill is compute bound).
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    q_start = tl.load(QS)
    seq_len = tl.load(SL)
    kvh = pid_h // GQA
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < (seq_len - q_start)
    q = tl.load(Q + offs_m[:, None] * stride_qm + pid_h * stride_qh + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), -1e30, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    qpos = q_start + offs_m
    hi = q_start + tl.minimum(BLOCK_M * (pid_m + 1), seq_len - q_start)
    for start_n in range(0, hi, BLOCK_N):
        idx = start_n + tl.arange(0, BLOCK_N)
        valid = idx < seq_len
        blk = tl.load(BT + idx // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
        off = idx % BLOCK_SIZE
        ptr = blk[:, None] * stride_kb + off[:, None] * stride_kt + kvh * HEAD_DIM + offs_d[None, :]
        k = tl.load(KC + ptr, mask=valid[:, None], other=0.0).to(tl.float16)
        if Q8K:
            sb = KS + ((blk * BLOCK_SIZE + off) * NKV + kvh) * NS
            ks_t = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float16)
            for c in tl.static_range(NS):
                sc = tl.load(sb + c, mask=valid, other=1.0).to(tl.float16)
                ks_t += tl.where((offs_d[None, :] >= c * LANE) &
                                 (offs_d[None, :] < (c + 1) * LANE), sc[:, None], 0.0).to(tl.float16)
            k *= ks_t
        s = tl.dot(q, tl.trans(k)) * scale
        causal = qpos[:, None] >= idx[None, :]
        s = tl.where(causal & valid[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(VC + ptr, mask=valid[:, None], other=0.0).to(tl.float16)
        if Q8V:
            sb = VS + ((blk * BLOCK_SIZE + off) * NKV + kvh) * NS
            vs_t = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float16)
            for c2 in tl.static_range(NS):
                sc2 = tl.load(sb + c2, mask=valid, other=1.0).to(tl.float16)
                vs_t += tl.where((offs_d[None, :] >= c2 * LANE) &
                                 (offs_d[None, :] < (c2 + 1) * LANE), sc2[:, None], 0.0).to(tl.float16)
            v *= vs_t
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(OUT + offs_m[:, None] * stride_om + pid_h * stride_oh + offs_d[None, :],
             acc.to(tl.float16), mask=mask_m[:, None])


def attn_prefill_paged(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                       block_table: torch.Tensor, q_start, seq_len,
                       block_size: int, gqa: int, head_dim: int, scale: float,
                       out: torch.Tensor | None = None,
                       block_m: int = 64, block_n: int = 64,
                       k_scale: torch.Tensor | None = None,
                       v_scale: torch.Tensor | None = None) -> torch.Tensor:
    """q [P, Hq, D] -> [P, Hq, D]."""
    p, hq, d = q.shape
    if out is None:
        out = torch.empty_like(q)
    if not torch.is_tensor(q_start):
        q_start = torch.tensor([q_start], dtype=torch.int32, device=q.device)
    if not torch.is_tensor(seq_len):
        seq_len = torch.tensor([seq_len], dtype=torch.int32, device=q.device)
    block_m, block_n, warps, stages = _ATTN_PREFILL_TILE
    grid = (triton.cdiv(p, block_m), hq)
    q8k = k_cache.dtype == torch.int8
    q8v = v_cache.dtype == torch.int8
    ns = ((k_scale if q8k else v_scale).shape[-1] if (q8k or q8v) else 1)
    attn_prefill_paged_kernel[grid](
        q, k_cache, v_cache, k_scale if q8k else q, v_scale if q8v else q,
        block_table, out, q_start, seq_len,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),
        out.stride(0), out.stride(1), hq // gqa,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa, BLOCK_M=block_m,
        BLOCK_N=block_n, scale=scale, Q8K=q8k, Q8V=q8v, NS=ns, LANE=32,
        num_warps=warps, num_stages=stages)
    return out


@triton.jit
def attn_decode_direct_kernel(Q, KT, VT, KS, VS, BT, SEQ, OUT,
                              stride_qb, stride_kb, stride_kt, stride_bt, stride_ob,
                              scale, NKV,
                              BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                              GQA: tl.constexpr, TDIM: tl.constexpr,
                              BLOCK_N: tl.constexpr, Q8K: tl.constexpr = False,
                              Q8V: tl.constexpr = False,
                              NS: tl.constexpr = 1, LANE: tl.constexpr = 32):
    """Paged decode attention without split-K: no partial buffers, no combine.

    For short contexts (<= a few thousand tokens) one program per KV head
    already fills the GPU, so this removes a kernel launch and a round-trip
    through the partial buffer from every layer.
    """
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    seqlen = tl.load(SEQ + b)
    offs_j = tl.arange(0, TDIM)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_qb + (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=(offs_j < GQA)[:, None], other=0.0)
    m_i = tl.full((TDIM,), -1e30, tl.float32)
    l_i = tl.zeros((TDIM,), tl.float32)
    acc = tl.zeros((TDIM, HEAD_DIM), tl.float32)
    for start_n in range(0, seqlen, BLOCK_N):
        idx = start_n + tl.arange(0, BLOCK_N)
        mask = idx < seqlen
        blk = tl.load(BT + b * stride_bt + idx // BLOCK_SIZE, mask=mask, other=0).to(tl.int64)
        off = idx % BLOCK_SIZE
        ptr = blk[:, None] * stride_kb + off[:, None] * stride_kt + kvh * HEAD_DIM + offs_d[None, :]
        k = tl.load(KT + ptr, mask=mask[:, None], other=0.0).to(tl.float16)
        if Q8K:
            sb = KS + ((blk * BLOCK_SIZE + off) * NKV + kvh) * NS
            ks_t = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float16)
            for c in tl.static_range(NS):
                sc = tl.load(sb + c, mask=mask, other=1.0).to(tl.float16)
                ks_t += tl.where((offs_d[None, :] >= c * LANE) &
                                 (offs_d[None, :] < (c + 1) * LANE), sc[:, None], 0.0).to(tl.float16)
            k *= ks_t
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(mask[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(VT + ptr, mask=mask[:, None], other=0.0).to(tl.float16)
        if Q8V:
            sb = VS + ((blk * BLOCK_SIZE + off) * NKV + kvh) * NS
            vs_t = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float16)
            for c2 in tl.static_range(NS):
                sc2 = tl.load(sb + c2, mask=mask, other=1.0).to(tl.float16)
                vs_t += tl.where((offs_d[None, :] >= c2 * LANE) &
                                 (offs_d[None, :] < (c2 + 1) * LANE), sc2[:, None], 0.0).to(tl.float16)
            v *= vs_t
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(OUT + b * stride_ob + (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(tl.float16), mask=(offs_j < GQA)[:, None])

@triton.jit
def attn_decode_sparse_kernel(Q, KT, VT, BT, SEQ, SEL, OUT,
                              stride_qb, stride_kb, stride_kt, stride_bt, stride_ob,
                              stride_selb, scale,
                              N_KEEP: tl.constexpr, MB: tl.constexpr,
                              BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                              GQA: tl.constexpr, TDIM: tl.constexpr,
                              BLOCK_N: tl.constexpr):
    """Decode attention over selected 4-token micro-blocks only."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    seqlen = tl.load(SEQ + b)
    offs_j = tl.arange(0, TDIM)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_qb + (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=(offs_j < GQA)[:, None], other=0.0)
    m_i = tl.full((TDIM,), -1e30, tl.float32)
    l_i = tl.zeros((TDIM,), tl.float32)
    acc = tl.zeros((TDIM, HEAD_DIM), tl.float32)
    total = N_KEEP * MB
    for start_n in range(0, total, BLOCK_N):
        j = start_n + tl.arange(0, BLOCK_N)
        inside = j < total
        mb = tl.load(SEL + b * stride_selb + kvh * N_KEEP + (j // MB),
                     mask=inside, other=0).to(tl.int64)
        idx = mb * MB + (j % MB)
        mask = inside & (idx < seqlen)
        blk = tl.load(BT + b * stride_bt + idx // BLOCK_SIZE, mask=mask, other=0).to(tl.int64)
        off = idx % BLOCK_SIZE
        ptr = blk[:, None] * stride_kb + off[:, None] * stride_kt + kvh * HEAD_DIM + offs_d[None, :]
        k = tl.load(KT + ptr, mask=mask[:, None], other=0.0).to(tl.float16)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(mask[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(VT + ptr, mask=mask[:, None], other=0.0).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(OUT + b * stride_ob + (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(tl.float16), mask=(offs_j < GQA)[:, None])


def select_micro_blocks(q, repr_kv, n_valid, frac, recent):
    """q [B, HQ, D] -> sel [B, HKV, N_KEEP]; recent micro-blocks are always kept."""
    b, hq, d = q.shape
    hkv = repr_kv.shape[-2]          # repr is [micro_block, rep, kv_head, head_dim]
    gqa = hq // hkv
    n_keep = max(1, min(int(n_valid * frac), n_valid))
    if n_keep >= n_valid:
        return torch.arange(n_valid, device=q.device).expand(b, hkv, n_valid).contiguous()
    qv = q.float().view(b, hkv, gqa, d)
    rep = repr_kv[:n_valid].float()                       # [m, 2, hkv, d]
    scores = torch.einsum("bhgd,mrhd->bhgrm", qv, rep).amax(2).amax(2)   # [b, hkv, m]
    sel = scores.topk(n_keep, dim=-1).indices
    recent_mb = max(0, recent // 4)
    if recent_mb:
        tail = torch.arange(max(0, n_valid - recent_mb), n_valid, device=q.device)
        sel = torch.cat([sel, tail.view(1, 1, -1).expand(b, hkv, tail.numel())], dim=-1)
    return sel.contiguous()


def attn_decode_sparse(q, k_cache, v_cache, block_table, seq_lens, sel,
                       block_size, gqa, head_dim, scale, mb=4, out=None):
    b, hq, d = q.shape
    hkv = hq // gqa
    n_keep = sel.shape[-1]
    if out is None:
        out = torch.empty((b, hq, d), dtype=torch.float16, device=q.device)
    attn_decode_sparse_kernel[(b, hkv)](
        q, k_cache, v_cache, block_table, seq_lens, sel, out,
        q.stride(0), k_cache.stride(0), k_cache.stride(1), block_table.stride(0),
        out.stride(0), sel.stride(0), scale,
        N_KEEP=n_keep, MB=mb, BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa,
        TDIM=max(16, triton.next_power_of_2(gqa)), BLOCK_N=64, num_warps=4, num_stages=2)
    return out


@triton.jit
def attn_decode_sparse_partial_kernel(Q, KT, VT, BT, SEQ, SEL, ACC, ML,
                                      stride_qb, stride_kb, stride_kt, stride_bt,
                                      stride_selb, stride_acc_b, stride_acc_h, stride_acc_s,
                                      stride_ml_b, stride_ml_h, stride_ml_s,
                                      scale,
                                      N_KEEP: tl.constexpr, MB: tl.constexpr,
                                      TPB: tl.constexpr, NUM_SPLITS: tl.constexpr,
                                      BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                                      GQA: tl.constexpr, TDIM: tl.constexpr):
    """Split-K sparse decode: each program owns a strided slice of the selected blocks.

    Selection leaves the KV stream non-contiguous, so the parallelism has to come from
    splitting the selected list itself (like the dense long path splits positions).
    """
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    sp = tl.program_id(2)
    seqlen = tl.load(SEQ + b)
    offs_j = tl.arange(0, TDIM)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_qb + (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=(offs_j < GQA)[:, None], other=0.0)
    m_i = tl.full((TDIM,), -1e30, tl.float32)
    l_i = tl.zeros((TDIM,), tl.float32)
    acc = tl.zeros((TDIM, HEAD_DIM), tl.float32)
    n_tiles = tl.cdiv(N_KEEP, TPB)
    offs_t = tl.arange(0, TPB * MB)
    for ti in range(sp, n_tiles, NUM_SPLITS):
        mb_ref = ti * TPB + (offs_t // MB)
        inside = mb_ref < N_KEEP
        mb = tl.load(SEL + b * stride_selb + kvh * N_KEEP + mb_ref, mask=inside, other=0).to(tl.int64)
        idx = mb * MB + (offs_t % MB)
        mask = inside & (idx < seqlen)
        blk = tl.load(BT + b * stride_bt + idx // BLOCK_SIZE, mask=mask, other=0).to(tl.int64)
        off = idx % BLOCK_SIZE
        ptr = blk[:, None] * stride_kb + off[:, None] * stride_kt + kvh * HEAD_DIM + offs_d[None, :]
        k = tl.load(KT + ptr, mask=mask[:, None], other=0.0).to(tl.float16)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(mask[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(VT + ptr, mask=mask[:, None], other=0.0).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    base = b * stride_acc_b + kvh * stride_acc_h + sp * stride_acc_s
    tl.store(ACC + base + offs_j[:, None] * HEAD_DIM + offs_d[None, :], acc)
    ml_base = b * stride_ml_b + kvh * stride_ml_h + sp * stride_ml_s
    tl.store(ML + ml_base + offs_j, m_i)
    tl.store(ML + ml_base + TDIM + offs_j, l_i)


def attn_decode_sparse_split(q, k_cache, v_cache, block_table, seq_lens, sel,
                             block_size, gqa, head_dim, scale, acc_buf, ml_buf,
                             out=None, num_splits=8, mb=4, tpb=16) -> torch.Tensor:
    b, hq, d = q.shape
    hkv = hq // gqa
    n_keep = sel.shape[-1]
    if out is None:
        out = torch.empty((b, hq, d), dtype=torch.float16, device=q.device)
    attn_decode_sparse_partial_kernel[(b, hkv, num_splits)](
        q, k_cache, v_cache, block_table, seq_lens, sel, acc_buf, ml_buf,
        q.stride(0), k_cache.stride(0), k_cache.stride(1), block_table.stride(0),
        sel.stride(0),
        acc_buf.stride(0), acc_buf.stride(1), acc_buf.stride(2),
        ml_buf.stride(0), ml_buf.stride(1), ml_buf.stride(2),
        scale,
        N_KEEP=n_keep, MB=mb, TPB=tpb, NUM_SPLITS=num_splits,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa, TDIM=max(16, triton.next_power_of_2(gqa)),
        num_warps=4, num_stages=2)
    attn_decode_combine_kernel[(b, hkv)](
        acc_buf, ml_buf, out,
        acc_buf.stride(0), acc_buf.stride(1), acc_buf.stride(2),
        ml_buf.stride(0), ml_buf.stride(1), ml_buf.stride(2),
        out.stride(0),
        HEAD_DIM=head_dim, GQA=gqa, TDIM=max(16, triton.next_power_of_2(gqa)), NUM_SPLITS=num_splits,
        num_warps=4)
    return out


@triton.jit
def attn_prefill_batch_kernel(Q, KC, VC, BT, OUT, POS, LEN, SEQ,
                              stride_qm, stride_qh, stride_kb, stride_kt,
                              stride_om, stride_oh, stride_bt, M,
                              BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                              GQA: tl.constexpr, BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr, scale):
    """Causal flash attention where each CTA row group belongs to one sequence.

    Prefill chunks from several requests are concatenated along the row axis (the GEMMs
    do not care), so attention has to know which sequence each group of BLOCK_M rows
    came from.  The caller guarantees that a group never straddles two sequences, which
    keeps the block-table lookup one dimensional: exactly the paged kernel plus a
    per-row position instead of one global q_start.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    kvh = pid_h // GQA
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < M
    seq = tl.load(SEQ + pid_m)
    q = tl.load(Q + offs_m[:, None] * stride_qm + pid_h * stride_qh + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)
    qpos = tl.load(POS + offs_m, mask=mask_m, other=-1)
    slen = tl.load(LEN + pid_m)
    m_i = tl.full((BLOCK_M,), -1e30, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    base = BT + seq * stride_bt
    for start_n in range(0, slen, BLOCK_N):
        idx = start_n + tl.arange(0, BLOCK_N)
        valid = idx < slen
        blk = tl.load(base + idx // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
        off = idx % BLOCK_SIZE
        ptr = (blk[:, None] * stride_kb + off[:, None] * stride_kt + kvh * HEAD_DIM
               + offs_d[None, :])
        k = tl.load(KC + ptr, mask=valid[:, None], other=0.0).to(tl.float16)
        s = tl.dot(q, tl.trans(k)) * scale
        causal = qpos[:, None] >= idx[None, :]
        s = tl.where(causal & valid[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(VC + ptr, mask=valid[:, None], other=0.0).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    acc = acc / tl.maximum(l_i, 1e-20)[:, None]
    tl.store(OUT + offs_m[:, None] * stride_om + pid_h * stride_oh + offs_d[None, :],
             acc.to(tl.float16), mask=mask_m[:, None])


def attn_prefill_batch(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                      block_tables: torch.Tensor, positions: torch.Tensor,
                      lens: torch.Tensor, seq_of_group: torch.Tensor,
                      n_rows: int, block_size: int, gqa: int, head_dim: int,
                      scale: float, out: torch.Tensor | None = None,
                      block_n: int = 64) -> torch.Tensor:
    """q [M, Hq, D] (rows of several sequences) -> [M, Hq, D].

    positions [M] int32 is the absolute position of each row inside its sequence, lens
    [G] int32 the key count of each group's sequence, seq_of_group [G] int32 which
    sequence each 16-row group belongs to.  Groups must not straddle sequences.
    """
    if out is None:
        out = torch.empty_like(q)
    hq = q.shape[1]
    groups = seq_of_group.numel()
    attn_prefill_batch_kernel[(groups, hq)](
        q, k_cache, v_cache, block_tables, out, positions, lens, seq_of_group,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),
        out.stride(0), out.stride(1), block_tables.stride(0), n_rows,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa, BLOCK_M=16,
        BLOCK_N=block_n, scale=scale, num_warps=4, num_stages=2)
    return out



@triton.jit
def rope_attn_decode_kernel(PART, BIAS, COS, SIN, KC, VC, BT, OUT,
                            SLOTS, POSITIONS, SEQ_LENS,
                            NQ, NKV, D, QS, KVS, stride_kb, stride_kt, stride_bt,
                            stride_ob, BLOCK_SIZE: tl.constexpr,
                            HEAD_DIM: tl.constexpr, GQA: tl.constexpr,
                            TDIM: tl.constexpr, BLOCK_N: tl.constexpr,
                            SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr,
                            scale, GQA_PAD: tl.constexpr, HALF: tl.constexpr):
    """Decode attention for one (row, kv head) with rope folded in.

    Reads the qkv projection straight from the split-K partials, applies rope, appends the
    new K/V row to the cache and runs the attention loop, removing the separate
    rope_scatter launch (24 of ~150 launches per step at batch 1, 24 of 195 at batch 32).

    Rope is done full width: the partner element d^HALF is fetched with a shifted read of
    the same columns, so no reshape/split gymnastics are needed and the attention loop is
    the ordinary full-width one.
    """
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    offs_j = tl.arange(0, GQA_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    shift = (offs_d + HALF) % HEAD_DIM
    sign = tl.where(offs_d < HALF, -1.0, 1.0).to(tl.float32)
    slot = tl.load(SLOTS + b)
    pos = tl.load(POSITIONS + b)
    seqlen = tl.load(SEQ_LENS + b).to(tl.int64)
    cost = tl.load(COS + pos * HALF + (offs_d % HALF)).to(tl.float32)
    sint = tl.load(SIN + pos * HALF + (offs_d % HALF)).to(tl.float32)
    mask_j = offs_j < GQA

    qcols = (kvh * GQA + offs_j[:, None]) * D + offs_d[None, :]
    qshift = (kvh * GQA + offs_j[:, None]) * D + shift[None, :]
    qa = tl.zeros((GQA_PAD, HEAD_DIM), tl.float32)
    qs2 = tl.zeros((GQA_PAD, HEAD_DIM), tl.float32)
    for sp in tl.static_range(SPLIT_K):
        base = sp * (TDIM * NQ) + b * NQ
        qa += tl.load(PART + base + qcols, mask=mask_j[:, None], other=0.0)
        qs2 += tl.load(PART + base + qshift, mask=mask_j[:, None], other=0.0)
    if HAS_BIAS:
        qa += tl.load(BIAS + qcols, mask=mask_j[:, None], other=0.0)
        qs2 += tl.load(BIAS + qshift, mask=mask_j[:, None], other=0.0)
    qa = qa.to(tl.float16).to(tl.float32)
    qs2 = qs2.to(tl.float16).to(tl.float32)
    qrot = (qa * cost[None, :] + sign[None, :] * qs2 * sint[None, :]).to(tl.float16)

    kc = QS + kvh * D + offs_d
    ks = QS + kvh * D + shift
    ka = tl.zeros((HEAD_DIM,), tl.float32)
    kb = tl.zeros((HEAD_DIM,), tl.float32)
    for sp in tl.static_range(SPLIT_K):
        base = sp * (TDIM * NQ) + b * NQ
        ka += tl.load(PART + base + kc)
        kb += tl.load(PART + base + ks)
    if HAS_BIAS:
        ka += tl.load(BIAS + kc)
        kb += tl.load(BIAS + ks)
    ka = ka.to(tl.float16).to(tl.float32)
    kb = kb.to(tl.float16).to(tl.float32)
    krot = (ka * cost + sign * kb * sint).to(tl.float16)
    kslot = (slot // BLOCK_SIZE) * stride_kb + (slot % BLOCK_SIZE) * stride_kt + kvh * D
    tl.store(KC + kslot + offs_d, krot)

    vc = QS + KVS + kvh * D + offs_d
    va = tl.zeros((HEAD_DIM,), tl.float32)
    for sp in tl.static_range(SPLIT_K):
        va += tl.load(PART + sp * (TDIM * NQ) + b * NQ + vc)
    if HAS_BIAS:
        va += tl.load(BIAS + vc)
    tl.store(VC + kslot + offs_d, va.to(tl.float16))
    tl.debug_barrier()

    m_i = tl.full((GQA_PAD,), -1e30, tl.float32)
    l_i = tl.zeros((GQA_PAD,), tl.float32)
    acc = tl.zeros((GQA_PAD, HEAD_DIM), tl.float32)
    for start_n in range(0, seqlen, BLOCK_N):
        idx = start_n + tl.arange(0, BLOCK_N)
        valid = idx < seqlen
        blk = tl.load(BT + b * stride_bt + idx // BLOCK_SIZE, mask=valid,
                      other=0).to(tl.int64)
        ptr = blk * stride_kb + (idx % BLOCK_SIZE) * stride_kt + kvh * D
        k = tl.load(KC + ptr[:, None] + offs_d[None, :], mask=valid[:, None],
                    other=0.0).to(tl.float16)
        sc = tl.dot(qrot, tl.trans(k)) * scale
        sc = tl.where((pos >= idx)[None, :] & valid[None, :], sc, -1e30)
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        p = tl.exp(sc - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(VC + ptr[:, None] + offs_d[None, :], mask=valid[:, None],
                    other=0.0).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    acc = acc / tl.maximum(l_i, 1e-20)[:, None]
    ocol = (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(OUT + b * stride_ob + ocol, acc.to(tl.float16), mask=mask_j[:, None])


def rope_attn_decode(partials: torch.Tensor, bias: torch.Tensor | None,
                     cos: torch.Tensor, sin: torch.Tensor,
                     k_cache: torch.Tensor, v_cache: torch.Tensor,
                     block_tables: torch.Tensor, slots: torch.Tensor,
                     positions: torch.Tensor, seq_lens: torch.Tensor,
                     n_q: int, n_kv: int, head_dim: int, scale: float,
                     out: torch.Tensor, block_size: int, block_n: int = 64,
                     gqa: int | None = None) -> torch.Tensor:
    """Fused rope + decode attention over the split-K partials (see the kernel docstring)."""
    gqa = gqa or n_q // n_kv
    batch = slots.numel()
    qs, kvs = n_q * head_dim, n_kv * head_dim
    split_k, tdim, nq = partials.shape
    rope_attn_decode_kernel[(batch, n_kv)](
        partials, bias if bias is not None else partials, cos, sin,
        k_cache, v_cache, block_tables, out, slots, positions, seq_lens,
        nq, n_kv, head_dim, qs, kvs, k_cache.stride(0), k_cache.stride(1),
        block_tables.stride(0), out.stride(0),
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa, TDIM=tdim,
        BLOCK_N=block_n, SPLIT_K=split_k, HAS_BIAS=bias is not None, scale=scale,
        GQA_PAD=max(16, triton.next_power_of_2(gqa)), HALF=head_dim // 2,
        num_warps=4, num_stages=2)
    return out

