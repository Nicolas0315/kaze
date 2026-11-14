"""Attention kernels.

Prefill: SDPA over a gathered contiguous prefix (flash/cuDNN backends).
Decode: a GQA-aware paged kernel with flash-decoding split-K.  Every paged
program loads each K/V element once and reuses it for all query heads of the
group, which is what keeps decode at the bandwidth roofline instead of
re-reading the cache once per query head.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

NUM_SPLITS = 8


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
    offs_s = tl.arange(0, NUM_SPLITS)
    ml_base = b * stride_ml_b + kvh * stride_ml_h
    m = tl.load(ML + ml_base + offs_s[:, None] * stride_ml_s + offs_j[None, :])
    l = tl.load(ML + ml_base + offs_s[:, None] * stride_ml_s + TDIM + offs_j[None, :])
    acc = tl.load(ACC + b * stride_acc_b + kvh * stride_acc_h
                  + offs_s[:, None, None] * stride_acc_s
                  + offs_j[None, :, None] * HEAD_DIM + offs_d[None, None, :])
    m_max = tl.max(m, 0)
    w = tl.exp(m - m_max[None, :])
    l_tot = tl.sum(l * w, 0)
    acc_tot = tl.sum(acc * w[:, :, None], 0) / l_tot[:, None]
    out_ptr = OUT + b * stride_ob + (kvh * GQA + offs_j)[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptr, acc_tot.to(tl.float16), mask=(offs_j < GQA)[:, None])


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
        BLOCK_N=block_n, NUM_SPLITS=num_splits, num_warps=4, num_stages=1,
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
        attn_decode_direct_kernel[(b, kvh)](
            q, k_cache, v_cache, k_scale if q8k else q, v_scale if q8v else q,
            block_tables, seq_lens, out,
            q.stride(0), k_cache.stride(0), k_cache.stride(1), block_tables.stride(0),
            out.stride(0), scale, kvh,
            BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa, TDIM=max(16, gqa),
            BLOCK_N=block_n, Q8K=q8k, Q8V=q8v, NS=ns, LANE=32,
            num_warps=4, num_stages=1)
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
        num_warps=4, num_stages=2)
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
        TDIM=max(16, gqa), BLOCK_N=64, num_warps=4, num_stages=2)
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
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, GQA=gqa, TDIM=max(16, gqa),
        num_warps=4, num_stages=2)
    attn_decode_combine_kernel[(b, hkv)](
        acc_buf, ml_buf, out,
        acc_buf.stride(0), acc_buf.stride(1), acc_buf.stride(2),
        ml_buf.stride(0), ml_buf.stride(1), ml_buf.stride(2),
        out.stride(0),
        HEAD_DIM=head_dim, GQA=gqa, TDIM=max(16, gqa), NUM_SPLITS=num_splits,
        num_warps=4)
    return out
