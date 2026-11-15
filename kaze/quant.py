"""Weight-only quantization: int4 / int8 groups packed for Triton kernels.

Layout (int4, group size G, packed 8 values per int32 along the input dim K):

    q_packed : int32[N, K // 8]     # nibble j of word i holds k = i * 8 + j
    scales   : fp16[N, K // G]      # per-row, per-group absmax scale
    dequant  : w[n, k] = (q[n, k] - 8) * scale[n, k // G]

Symmetric, zero point 8 (unsigned nibble), 16 levels in [-8, 7] * scale.
Group size is a multiple of 8; 128 keeps a 128-wide K tile inside one group,
so a GEMM tile needs exactly one scale per row.
"""
from __future__ import annotations

import torch

NIBBLE_SHIFTS = tuple(4 * j for j in range(8))


def quantize_int4(weight: torch.Tensor, group: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """weight [N, K] fp16/fp32 -> packed int32 [N, K//8], scales fp16 [N, K//G].

    Symmetric, one scale per group of 128.  Four alternatives were measured against
    the fp32 reference on Qwen2.5-0.5B and all lost:
      - clip search over the scale:            0.9730 -> worse
      - per-output-channel normalisation:      0.9730 -> 0.9215
      - error feedback along K (alpha 1.0):    0.9730 -> 0.9411
      - group 64 instead of 128:               0.9730 -> 0.9724 (noise)
    So plain absmax per group stays until a Hessian-based (GPTQ-style) solver lands.
    """
    w = weight.detach().float()
    n, k = w.shape
    assert k % group == 0 and group % 8 == 0, (k, group)
    wg = w.view(n, k // group, group)
    scale = torch.clamp(wg.abs().amax(dim=2) / 8.0, min=1e-8)
    q = torch.round(wg / scale.unsqueeze(2)).clamp_(-8, 7).to(torch.int32) + 8
    q = q.view(n, k // 8, 8)
    packed = torch.zeros(n, k // 8, dtype=torch.int32, device=w.device)
    for j, sh in enumerate(NIBBLE_SHIFTS):
        packed |= (q[:, :, j] & 0xF) << sh
    return packed.contiguous(), scale.half().contiguous()


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, group: int = 128,
                    dtype: torch.dtype = torch.float16) -> torch.Tensor:
    n, k8 = packed.shape
    k = k8 * 8
    words = packed.view(n, k8, 1)
    q = torch.empty(n, k8, 8, dtype=torch.int32, device=packed.device)
    for j, sh in enumerate(NIBBLE_SHIFTS):
        q[:, :, j] = (words[:, :, 0] >> sh) & 0xF
    q = q.view(n, k).to(dtype) - 8.0
    s = scales.to(dtype).repeat_interleave(group, dim=1)
    return (q * s).to(dtype)


def quantize_int8(weight: torch.Tensor, group: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().float()
    n, k = w.shape
    wg = w.view(n, k // group, group)
    scale = wg.abs().amax(dim=2) / 127.0
    scale = torch.clamp(scale, min=1e-8)
    q = torch.round(wg / scale.unsqueeze(2)).clamp_(-127, 127).to(torch.int8)
    return q.view(n, k).contiguous(), scale.half().contiguous()


def dequantize_int8(q: torch.Tensor, scales: torch.Tensor, group: int = 128,
                    dtype: torch.dtype = torch.float16) -> torch.Tensor:
    s = scales.to(dtype).repeat_interleave(group, dim=1)
    return (q.to(dtype) * s).to(dtype)


class QuantLinear:
    """A linear layer whose weight stays quantized and is consumed by Triton kernels.

    kind is one of int4, int8 or fp16 (dense fallback).  Fused layers (qkv,
    gate_up) are simply several rows concatenated at build time, so one
    GEMV/GEMM serves all of them.
    """

    __slots__ = ("packed", "scales", "bias", "n", "k", "group", "kind", "dtype",
                 "scratch", "packed_tm")

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor | None, bias: torch.Tensor | None,
                 n: int, k: int, group: int, kind: str, dtype: torch.dtype = torch.float16):
        self.packed = packed
        self.packed_tm = None
        self.scales = scales
        self.bias = bias
        self.n = n
        self.k = k
        self.group = group
        self.kind = kind
        self.dtype = dtype
        self.scratch: dict = {}

    @classmethod
    def from_float(cls, weight: torch.Tensor, bias: torch.Tensor | None, kind: str = "int4",
                   group: int = 128, chunk: int = 8192) -> "QuantLinear":
        """Quantize row-chunked so a 3B vocab-wide lm_head fits on a small GPU."""
        n, k = weight.shape
        w = weight.detach()
        if (kind not in ("int4", "int8")) or k % group != 0:
            return cls(w.half().contiguous(), None,
                       bias.half().contiguous() if bias is not None else None,
                       n, k, group, "fp16", dtype=torch.float16)
        fn = quantize_int4 if kind == "int4" else quantize_int8
        packed_parts, scale_parts = [], []
        for start in range(0, n, chunk):
            p, sc = fn(w[start:start + chunk], group)
            packed_parts.append(p)
            scale_parts.append(sc)
        packed = torch.cat(packed_parts, dim=0).contiguous()
        scales = torch.cat(scale_parts, dim=0).contiguous()
        del packed_parts, scale_parts
        out = cls(packed, scales, bias.half().contiguous() if bias is not None else None,
                  n, k, group, kind, dtype=torch.float16)
        import os as _os

        want_tm = _os.environ.get("KAZE_INT4_TM")
        if kind == "int4" and n * k >= (1 << 20) and want_tm != "0":
            if want_tm == "1":
                out.make_tile_major_copy()
            else:
                # 未指定なら空き VRAM を見て決める（重みが 2 倍になるため）
                try:
                    import torch as _t

                    # カード全体の VRAM で判断する（読み込み途中の空きを見ると
                    # 8GB 級でも通ってしまい、後で OOM になる）
                    total = _t.cuda.get_device_properties(0).total_memory
                    if total >= 20 * (1 << 30):
                        out.make_tile_major_copy()
                except Exception:
                    pass
        return out

    def make_tile_major_copy(self) -> None:
        """decode 用に [K/G, N, G/2] の複製を作る（2-D の重みは prefill 用に残す）。"""
        if self.kind != "int4" or self.packed.dim() != 2:
            return
        g = self.group
        n, half = self.packed.shape
        blocks = self.k // g
        src = self.packed.view(torch.int8)
        dst = src.new_empty((blocks, n, g // 2))
        step = max(1, (1 << 22) // max(n * (g // 2), 1))
        for b0 in range(0, blocks, step):
            b1 = min(blocks, b0 + step)
            dst[b0:b1] = src[:, b0 * (g // 2):b1 * (g // 2)].view(n, b1 - b0, g // 2).permute(1, 0, 2)
        self.packed_tm = dst.contiguous()

    @property
    def nbytes(self) -> int:
        b = self.packed.numel() * self.packed.element_size()
        if self.scales is not None:
            b += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            b += self.bias.numel() * self.bias.element_size()
        return b

    def dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        if self.kind == "int4":
            return dequantize_int4(self.packed, self.scales, self.group, dtype)
        if self.kind == "int8":
            return dequantize_int8(self.packed, self.scales, self.group, dtype)
        return self.packed.to(dtype)

    def to_dense(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return self.dequantize(dtype)


def concat_rows(layers: list[QuantLinear]) -> QuantLinear:
    """Fuse several same-K linears into one wide linear (qkv, gate+up)."""
    kinds = {l.kind for l in layers}
    assert len(kinds) == 1, kinds
    kind = kinds.pop()
    k = layers[0].k
    packed = torch.cat([l.packed for l in layers], dim=0).contiguous()
    scales = None
    if layers[0].scales is not None:
        scales = torch.cat([l.scales for l in layers], dim=0).contiguous()
    bias = None
    if any(l.bias is not None for l in layers):
        bias = torch.cat(
            [l.bias if l.bias is not None else torch.zeros(l.n, dtype=torch.float16,
                                                          device=l.packed.device)
             for l in layers], dim=0).contiguous()
    return QuantLinear(packed, scales, bias, sum(l.n for l in layers), k,
                       layers[0].group, kind, layers[0].dtype)
