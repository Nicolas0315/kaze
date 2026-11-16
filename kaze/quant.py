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

from typing import Iterator

import torch

NIBBLE_SHIFTS = tuple(4 * j for j in range(8))


def _validation_chunks(weight: torch.Tensor) -> Iterator[torch.Tensor]:
    # Keep validation temporaries bounded even for vocab-wide heads. Slicing
    # also handles noncontiguous matrices without copying the full weight.
    limit = 1024 * 1024
    width = min(weight.shape[1], limit)
    rows = max(1, limit // width)
    for start in range(0, weight.shape[0], rows):
        for column in range(0, weight.shape[1], width):
            yield weight[start:start + rows, column:column + width]


def _validate_weight(weight: torch.Tensor, group: int, *, int4: bool = False) -> None:
    if type(group) is not int or group <= 0 or (int4 and group % 8):
        raise ValueError("group must be positive (and a multiple of 8 for int4)")
    if weight.ndim != 2 or any(d <= 0 for d in weight.shape):
        raise ValueError("weight must be a nonempty [N, K] matrix")
    if not weight.is_floating_point():
        raise ValueError("weight must contain finite floating point values")
    for part in _validation_chunks(weight):
        if not torch.isfinite(part).all().item():
            raise ValueError("weight must contain finite floating point values")


def _stored_scale(absmax: torch.Tensor, divisor: float) -> torch.Tensor:
    # Quantize against the actual fp16 scale consumed by the kernels, including
    # subnormal scales; a float32-only minimum rounds to zero when stored.
    scale = (absmax / divisor).clamp(min=2.0 ** -24).half()
    if not torch.isfinite(scale).all().item():
        raise ValueError("quantization scale exceeds the finite fp16 range")
    return scale.float()


def _validate_dequant(packed: torch.Tensor, scales: torch.Tensor, group: int,
                      *, int4: bool = False) -> None:
    if type(group) is not int or group <= 0 or (int4 and group % 8):
        raise ValueError("invalid quantization group")
    if packed.ndim != 2 or packed.dtype != (torch.int32 if int4 else torch.int8):
        raise ValueError("invalid packed tensor shape or dtype")
    n, width = packed.shape
    k = width * 8 if int4 else width
    if k % group or tuple(scales.shape) != (n, k // group):
        raise ValueError("scales shape must match packed rows and groups")
    if scales.device != packed.device:
        raise ValueError("packed values and scales must be on the same device")


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
    _validate_weight(weight, group, int4=True)
    w = weight.detach().float()
    n, k = w.shape
    if k % group:
        raise ValueError("input width must be divisible by group")
    wg = w.view(n, k // group, group)
    scale = _stored_scale(wg.abs().amax(dim=2), 8.0)
    q = torch.round(wg / scale.unsqueeze(2)).clamp_(-8, 7).to(torch.int32) + 8
    q = q.view(n, k // 8, 8)
    packed = torch.zeros(n, k // 8, dtype=torch.int32, device=w.device)
    for j, sh in enumerate(NIBBLE_SHIFTS):
        packed |= (q[:, :, j] & 0xF) << sh
    return packed.contiguous(), scale.half().contiguous()


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, group: int = 128,
                    dtype: torch.dtype = torch.float16) -> torch.Tensor:
    _validate_dequant(packed, scales, group, int4=True)
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
    _validate_weight(weight, group)
    w = weight.detach().float()
    n, k = w.shape
    if k % group:
        raise ValueError("input width must be divisible by group")
    wg = w.view(n, k // group, group)
    scale = _stored_scale(wg.abs().amax(dim=2), 127.0)
    q = torch.round(wg / scale.unsqueeze(2)).clamp_(-127, 127).to(torch.int8)
    return q.view(n, k).contiguous(), scale.half().contiguous()


def dequantize_int8(q: torch.Tensor, scales: torch.Tensor, group: int = 128,
                    dtype: torch.dtype = torch.float16) -> torch.Tensor:
    _validate_dequant(q, scales, group)
    s = scales.to(dtype).repeat_interleave(group, dim=1)
    return (q.to(dtype) * s).to(dtype)


class QuantLinear:
    """A linear layer whose weight stays quantized and is consumed by Triton kernels.

    kind is one of int4, int8 or fp16 (dense fallback).  Fused layers (qkv,
    gate_up) are simply several rows concatenated at build time, so one
    GEMV/GEMM serves all of them.
    """

    __slots__ = ("packed", "scales", "bias", "n", "k", "group", "kind", "dtype", "scratch")

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor | None, bias: torch.Tensor | None,
                 n: int, k: int, group: int, kind: str, dtype: torch.dtype = torch.float16):
        self.packed = packed
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
        if kind not in ("int4", "int8", "fp16"):
            raise ValueError(f"unknown quantization kind: {kind!r}")
        _validate_weight(weight, group, int4=kind == "int4")
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("chunk must be a positive integer")
        n, k = weight.shape
        if bias is not None and (tuple(bias.shape) != (n,) or bias.device != weight.device
                                 or not bias.is_floating_point() or not torch.isfinite(bias).all().item()
                                 or (bias.abs() > torch.finfo(torch.float16).max).any().item()):
            raise ValueError("bias must be finite, floating point, same-device, and have shape [N]")
        w = weight.detach()
        if kind == "fp16" or k % group != 0:
            for part in _validation_chunks(w):
                if (part.abs() > torch.finfo(torch.float16).max).any().item():
                    raise ValueError("weight exceeds the finite fp16 range for dense fallback")
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
        return cls(packed, scales, bias.half().contiguous() if bias is not None else None,
                   n, k, group, kind, dtype=torch.float16)

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
    if not layers:
        raise ValueError("layers must not be empty")
    if any(l.group != layers[0].group for l in layers):
        raise ValueError("layers must have the same quantization group")
    if any(l.k != layers[0].k or l.kind != layers[0].kind for l in layers):
        raise ValueError("layers must have the same input width and kind")
    kinds = {l.kind for l in layers}
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
