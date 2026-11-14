"""Elementwise kernels: SwiGLU gate, residual add, row gather/scatter helpers."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(G, U, Y, N, stride_g, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    g = tl.load(G + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


@triton.jit
def add_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    y = tl.load(Y + row * N + cols, mask=mask, other=0.0)
    tl.store(X + row * N + cols, (x.to(tl.float32) + y.to(tl.float32)).to(tl.float16), mask=mask)


@triton.jit
def embed_dequant_kernel(IDS, WQ, SC, Y, N, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    """Gather int4 rows and dequantize: Y[row] = dequant(WQ[ids[row]])."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    token = tl.load(IDS + row)
    wp = tl.load(WQ + token.to(tl.int64) * (N // 8) + cols // 8, mask=mask, other=0)
    shifts = (cols % 8) * 4
    q = ((wp >> shifts) & 0xF).to(tl.float32) - 8.0
    s = tl.load(SC + token.to(tl.int64) * (N // GROUP) + cols // GROUP, mask=mask, other=0.0)
    tl.store(Y + row * N + cols, (q * s).to(tl.float16), mask=mask)


def silu_mul(g: torch.Tensor, u: torch.Tensor, out: torch.Tensor | None = None,
             stride_g: int | None = None, stride_out: int | None = None) -> torch.Tensor:
    """g, u share a fused [M, 2N] buffer: u = g + offset.  out is [M, N]."""
    m = g.shape[0]
    n = out.shape[1] if out is not None else (u.shape[1])
    sg = stride_g if stride_g is not None else u.shape[1]
    sy = stride_out if stride_out is not None else n
    if out is None:
        out = torch.empty((m, n), dtype=torch.float16, device=g.device)
    silu_mul_kernel[(m,)](g, u, out, n, sg, sy, BLOCK=triton.next_power_of_2(n), num_warps=8)
    return out
