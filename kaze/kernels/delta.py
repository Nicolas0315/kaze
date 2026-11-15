"""線形注意（gated delta rule）の 1 トークン分をまとめて計算するカーネル。

正規化・減衰・状態更新・出力を 1 回の起動で行います。PyTorch で書くと 1 層あたり
20 個ほどの小さなカーネルになるため、decode ではここが効きます。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def delta_step_kernel(
    Q, K, V, G, BETA, STATE, OUT,
    HK: tl.constexpr, HV: tl.constexpr, BV: tl.constexpr,
    EPS: tl.constexpr, QSCALE: tl.constexpr,
):
    h = tl.program_id(0)
    blk = tl.program_id(1)
    i = tl.arange(0, HK)
    j = blk * BV + tl.arange(0, BV)
    q = tl.load(Q + h * HK + i)
    k = tl.load(K + h * HK + i)
    v = tl.load(V + h * HV + j)
    g = tl.load(G + h)
    beta = tl.load(BETA + h)

    qn = q * tl.rsqrt(tl.sum(q * q) + EPS) * QSCALE
    kn = k * tl.rsqrt(tl.sum(k * k) + EPS)

    sp = STATE + h * HK * HV + i[:, None] * HV + j[None, :]
    s = tl.load(sp) * tl.exp(g)
    kv = tl.sum(s * kn[:, None], axis=0)
    delta = (v - kv) * beta
    s = s + kn[:, None] * delta[None, :]
    tl.store(sp, s)
    tl.store(OUT + h * HV + j, tl.sum(s * qn[:, None], axis=0))


def delta_step(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
               beta: torch.Tensor, state: torch.Tensor, out: torch.Tensor,
               block_v: int = 32, eps: float = 1e-6) -> torch.Tensor:
    """q/k: [nv, hk]、v/out: [nv, hv]、g/beta: [nv]、state: [nv, hk, hv]（in-place）。"""
    nv, hk = k.shape
    hv = v.shape[-1]
    delta_step_kernel[(nv, triton.cdiv(hv, block_v))](
        q, k, v, g, beta, state, out,
        HK=hk, HV=hv, BV=block_v, EPS=eps, QSCALE=hk ** -0.5, num_warps=4)
    return out
