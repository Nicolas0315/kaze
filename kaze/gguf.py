"""GGUF v2/v3 reader and ggml block dequantisation, mapped onto kaze's layout.

Stage A of kaze's GGUF support: one .gguf file is parsed, the weights are
dequantised to fp16 and exposed under the HuggingFace names that
kaze.loader.load_model already consumes, so the existing builder can
requantise them into kaze's own int4/int8 layout.

Block formats implemented (all little endian, one entry in GGML_TYPES):

    F32 / F16 / BF16                  unquantised passthrough
    Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0  32-value legacy blocks
    Q4_K / Q5_K / Q6_K / Q8_K        256-value super blocks

Every other format (Q2_K, Q3_K, IQ*, TQ*, MXFP4) raises instead of returning
garbage - see docs/gguf.md for the reason each one is out of scope.
"""
from __future__ import annotations

import math
import os
import struct

import torch

from kaze.config import ModelConfig

GGUF_MAGIC = b"GGUF"
GGUF_VERSIONS = (2, 3)

# ggml tensor data types: id -> (name, elements per block, bytes per block).
# A block is the smallest addressable unit; a tensor row is a whole number of
# blocks, which is why every quantised row width is a multiple of the block.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}

_UNQUANTISED_DTYPES = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}

# metadata value types
_TYPE_UINT8, _TYPE_INT8, _TYPE_UINT16, _TYPE_INT16 = 0, 1, 2, 3
_TYPE_UINT32, _TYPE_INT32, _TYPE_FLOAT32, _TYPE_BOOL = 4, 5, 6, 7
_TYPE_STRING, _TYPE_ARRAY, _TYPE_UINT64, _TYPE_INT64, _TYPE_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FORMATS = {_TYPE_UINT8: "<B", _TYPE_INT8: "<b", _TYPE_UINT16: "<H",
                   _TYPE_INT16: "<h", _TYPE_UINT32: "<I", _TYPE_INT32: "<i",
                   _TYPE_FLOAT32: "<f", _TYPE_BOOL: "<?", _TYPE_UINT64: "<Q",
                   _TYPE_INT64: "<q", _TYPE_FLOAT64: "<d"}
_SCALAR_MIN_BYTES = {_TYPE_UINT8: 1, _TYPE_INT8: 1, _TYPE_UINT16: 2, _TYPE_INT16: 2,
                     _TYPE_UINT32: 4, _TYPE_INT32: 4, _TYPE_FLOAT32: 4, _TYPE_BOOL: 1,
                     _TYPE_STRING: 8, _TYPE_UINT64: 8, _TYPE_INT64: 8, _TYPE_FLOAT64: 8}

DEFAULT_ALIGNMENT = 32


class TensorInfo:
    """One entry of the GGUF tensor table.

    shape is in torch order, i.e. the ggml dimension list reversed: ggml stores
    the fastest-moving (input/channel) dimension first, torch the last.
    """

    __slots__ = ("name", "shape", "ggml_type", "offset", "nbytes")

    def __init__(self, name: str, shape: tuple[int, ...], ggml_type: int,
                 offset: int, nbytes: int):
        self.name = name
        self.shape = shape
        self.ggml_type = ggml_type
        self.offset = offset
        self.nbytes = nbytes

    @property
    def type_name(self) -> str:
        return GGML_TYPES[self.ggml_type][0]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"TensorInfo({self.name!r}, {self.shape}, {self.type_name}, "
                f"offset={self.offset}, nbytes={self.nbytes})")


def _align(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def _read_string(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    if length > (1 << 32):
        raise ValueError(f"GGUF: implausible string length {length}")
    raw = f.read(length)
    if len(raw) != length:
        raise IOError(f"GGUF: truncated string ({len(raw)} of {length} bytes)")
    return raw.decode("utf-8", "replace")


def _read_value(f, value_type: int, remaining: int):
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is not None:
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if value_type == _TYPE_STRING:
        return _read_string(f)
    if value_type == _TYPE_ARRAY:
        element_type, count = struct.unpack("<IQ", f.read(12))
        minimum = _SCALAR_MIN_BYTES.get(element_type)
        if minimum is None:
            raise ValueError(f"GGUF: unknown array element type {element_type}")
        # a corrupt count must not turn into a giant allocation
        if count * minimum > remaining:
            raise ValueError(f"GGUF: array of {count} elements does not fit the file")
        return [_read_value(f, element_type, remaining) for _ in range(count)]
    raise ValueError(f"GGUF: unknown metadata value type {value_type}")


class GGUFReader:
    """Parses a GGUF file and dequantises its tensors on demand."""

    def __init__(self, path: str):
        self.path = path
        self.file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"{path}: not a GGUF file (magic {magic!r})")
            (version,) = struct.unpack("<I", f.read(4))
            if version not in GGUF_VERSIONS:
                raise ValueError(f"{path}: unsupported GGUF version {version} "
                                 f"(this reader handles {GGUF_VERSIONS})")
            self.version = version
            self.n_tensors, self.n_kv = struct.unpack("<QQ", f.read(16))
            self.metadata: dict[str, object] = {}
            for _ in range(self.n_kv):
                key = _read_string(f)
                (value_type,) = struct.unpack("<I", f.read(4))
                self.metadata[key] = _read_value(f, value_type, self.file_size - f.tell())
            self.tensors: dict[str, TensorInfo] = {}
            for _ in range(self.n_tensors):
                name = _read_string(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                if not 1 <= n_dims <= 8:
                    raise ValueError(f"{path}: tensor {name!r} has {n_dims} dimensions")
                dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
                ggml_type, offset = struct.unpack("<IQ", f.read(12))
                if ggml_type not in GGML_TYPES:
                    raise ValueError(
                        f"{path}: tensor {name!r} uses unsupported ggml type {ggml_type}; "
                        "supported: " + ", ".join(t[0] for t in GGML_TYPES.values()))
                shape = tuple(int(d) for d in reversed(dims))
                elements = math.prod(shape)
                block_elems = GGML_TYPES[ggml_type][1]
                if elements % block_elems:
                    raise ValueError(f"{path}: tensor {name!r} shape {shape} is not a "
                                     f"multiple of the {block_elems}-value block")
                nbytes = elements // block_elems * GGML_TYPES[ggml_type][2]
                if name in self.tensors:
                    raise ValueError(f"{path}: duplicate tensor name {name!r}")
                self.tensors[name] = TensorInfo(name, shape, ggml_type, offset, nbytes)
            alignment = self.metadata.get("general.alignment", DEFAULT_ALIGNMENT)
            if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0:
                raise ValueError(f"{path}: invalid alignment {alignment!r}")
            self.alignment = alignment
            self.data_offset = _align(f.tell(), alignment)
        for info in self.tensors.values():
            if self.data_offset + info.offset + info.nbytes > self.file_size:
                raise ValueError(f"{path}: tensor {info.name!r} runs past the end of the "
                                 f"file ({self.file_size} bytes); the file is truncated")
        self._file = open(path, "rb")

    # ------------------------------------------------------------- metadata
    @property
    def architecture(self) -> str:
        arch = self.metadata.get("general.architecture")
        if not isinstance(arch, str) or not arch:
            raise ValueError(f"{self.path}: general.architecture is missing")
        return arch

    def arch_value(self, key: str, default=None):
        return self.metadata.get(f"{self.architecture}.{key}", default)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "GGUFReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- data
    def _read(self, offset: int, nbytes: int) -> bytes:
        self._file.seek(self.data_offset + offset)
        buf = self._file.read(nbytes)
        if len(buf) != nbytes:
            raise IOError(f"{self.path}: short read at {offset} ({len(buf)} of {nbytes} "
                          "bytes); the file is truncated")
        return buf

    def raw_tensor(self, name: str) -> torch.Tensor:
        """The stored bytes as uint8, shaped [n_blocks, bytes_per_block]."""
        info = self.tensors[name]
        block_bytes = GGML_TYPES[info.ggml_type][2]
        buf = self._read(info.offset, info.nbytes)
        return torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(-1, block_bytes)

    def tensor(self, name: str, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantise one tensor into torch order [rows, columns]."""
        info = self.tensors[name]
        if info.ggml_type in _UNQUANTISED_DTYPES:
            source = _UNQUANTISED_DTYPES[info.ggml_type]
            flat = torch.frombuffer(bytearray(self._read(info.offset, info.nbytes)),
                                    dtype=source)
            return flat.reshape(info.shape).to(dtype).contiguous()
        _, block_elems, block_bytes = GGML_TYPES[info.ggml_type]
        k = info.shape[-1]
        rows = math.prod(info.shape[:-1])
        blocks_per_row = k // block_elems
        row_bytes = blocks_per_row * block_bytes
        # cap the temporary at ~8 MB of raw blocks: the fp32 intermediate is
        # ~2.8x that for a K-block format, and the whole tensor is fp16 anyway
        chunk_rows = max(1, (8 << 20) // row_bytes)
        flat = torch.empty(rows * k, dtype=dtype)
        for start in range(0, rows, chunk_rows):
            take = min(chunk_rows, rows - start)
            buf = self._read(info.offset + start * row_bytes, take * row_bytes)
            blocks = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(
                take * blocks_per_row, block_bytes)
            flat[start * k:(start + take) * k] = dequantize_blocks(
                info.ggml_type, blocks).to(dtype).reshape(-1)
        return flat.reshape(info.shape)


# --------------------------------------------------------------- dequant
def _raw_bytes(raw: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """A fresh contiguous byte copy.

    torch's dtype views need a storage offset that the target dtype can divide,
    and .contiguous() keeps the offset of an already-contiguous slice, so a
    one-block row would hit "storage_offset() must be divisible by 4" here.
    Copying is what actually guarantees offset 0.
    """
    out = torch.empty((raw.shape[0], stop - start), dtype=torch.uint8)
    out.copy_(raw[:, start:stop])
    return out


def _f16_column(raw: torch.Tensor, offset: int) -> torch.Tensor:
    """Per-block fp16 header field (scale, min, ...) as float32 [n]."""
    return _raw_bytes(raw, offset, offset + 2).view(torch.float16).to(torch.float32).squeeze(1)


def _f32_column(raw: torch.Tensor, offset: int) -> torch.Tensor:
    return _raw_bytes(raw, offset, offset + 4).view(torch.float32).squeeze(1)


def _i32_column(raw: torch.Tensor, offset: int) -> torch.Tensor:
    return _raw_bytes(raw, offset, offset + 4).view(torch.int32).squeeze(1)


def _dequant_q4_0(raw: torch.Tensor) -> torch.Tensor:
    d = _f16_column(raw, 0)
    qs = raw[:, 2:18].to(torch.int32)
    lo = (qs & 0x0F) - 8
    hi = (qs >> 4) - 8
    q = torch.cat([lo, hi], dim=1).to(torch.float32)
    return q * d.unsqueeze(1)


def _dequant_q4_1(raw: torch.Tensor) -> torch.Tensor:
    d = _f16_column(raw, 0)
    m = _f16_column(raw, 2)
    qs = raw[:, 4:20].to(torch.int32)
    q = torch.cat([qs & 0x0F, qs >> 4], dim=1).to(torch.float32)
    return q * d.unsqueeze(1) + m.unsqueeze(1)


def _dequant_q5(raw: torch.Tensor, *, with_min: bool) -> torch.Tensor:
    """Q5_0 (offset -16, scale only) and Q5_1 (affine) share the 5th-bit packing."""
    d = _f16_column(raw, 0)
    m = None
    header = 2
    if with_min:
        m = _f16_column(raw, 2)
        header = 4
    qh = _i32_column(raw, header)
    qs = raw[:, header + 4:header + 20].to(torch.int32)
    bits = torch.arange(32, device=raw.device, dtype=torch.int32)
    high = (qh.unsqueeze(1) >> bits) & 1                     # [n, 32]
    lo = (qs & 0x0F) | (high[:, :16] << 4)
    hi = (qs >> 4) | (high[:, 16:] << 4)
    q = torch.cat([lo, hi], dim=1).to(torch.float32)
    if with_min:
        return q * d.unsqueeze(1) + m.unsqueeze(1)
    return (q - 16.0) * d.unsqueeze(1)


def _dequant_q8_0(raw: torch.Tensor) -> torch.Tensor:
    d = _f16_column(raw, 0)
    qs = _raw_bytes(raw, 2, 34).view(torch.int8).to(torch.float32)
    return qs * d.unsqueeze(1)


def _k_scale_min(scales: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The 8 (6-bit scale, 6-bit min) pairs packed into a K-block's 12 bytes."""
    d_parts, m_parts = [], []
    for j in range(8):
        if j < 4:
            scale = scales[:, j] & 63
            minimum = scales[:, j + 4] & 63
        else:
            scale = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
            minimum = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
        d_parts.append(scale.to(torch.float32))
        m_parts.append(minimum.to(torch.float32))
    return d_parts, m_parts


def _dequant_k(raw: torch.Tensor, *, five_bit: bool) -> torch.Tensor:
    """Q4_K and Q5_K: 8 sub-blocks of 32 values share one 256-value super block."""
    n = raw.shape[0]
    d = _f16_column(raw, 0)
    dmin = _f16_column(raw, 2)
    scale_parts, min_parts = _k_scale_min(raw[:, 4:16])
    header = 16
    qh = None
    if five_bit:
        qh = raw[:, 16:48].to(torch.int32)                   # [n, 32] 5th bits
        header = 48
    qs = raw[:, header:header + 128].to(torch.int32)
    out = torch.empty(n, 256, dtype=torch.float32, device=raw.device)
    for group in range(4):
        q = qs[:, 32 * group:32 * group + 32]
        lo = q & 0x0F
        hi = q >> 4
        if five_bit:
            u1 = 1 << (2 * group)
            u2 = u1 << 1
            lo = lo | (((qh & u1) != 0).to(torch.int32) << 4)
            hi = hi | (((qh & u2) != 0).to(torch.int32) << 4)
        d1 = d * scale_parts[2 * group]
        m1 = dmin * min_parts[2 * group]
        d2 = d * scale_parts[2 * group + 1]
        m2 = dmin * min_parts[2 * group + 1]
        out[:, 64 * group:64 * group + 32] = lo.to(torch.float32) * d1.unsqueeze(1) - m1.unsqueeze(1)
        out[:, 64 * group + 32:64 * group + 64] = hi.to(torch.float32) * d2.unsqueeze(1) - m2.unsqueeze(1)
    return out


def _dequant_q6_k(raw: torch.Tensor) -> torch.Tensor:
    n = raw.shape[0]
    d = _f16_column(raw, 208)
    ql = raw[:, 0:128].to(torch.int32)
    qh = raw[:, 128:192].to(torch.int32)
    sc = _raw_bytes(raw, 192, 208).view(torch.int8).to(torch.float32)   # [n, 16]
    out = torch.empty(n, 256, dtype=torch.float32, device=raw.device)

    def wide_scale(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # the 32 values of one half-block use sc[is] for l < 16 and sc[is + 1] above
        return torch.cat([a.unsqueeze(1).expand(n, 16), b.unsqueeze(1).expand(n, 16)], dim=1)

    for half in range(2):
        ql_half = ql[:, 64 * half:64 * half + 64]
        qh_half = qh[:, 32 * half:32 * half + 32]
        sc_half = sc[:, 8 * half:8 * half + 8]
        low_a, low_b = ql_half[:, :32] & 0x0F, ql_half[:, 32:] & 0x0F
        high_a, high_b = ql_half[:, :32] >> 4, ql_half[:, 32:] >> 4
        q1 = (low_a | (((qh_half >> 0) & 3) << 4)) - 32
        q2 = (low_b | (((qh_half >> 2) & 3) << 4)) - 32
        q3 = (high_a | (((qh_half >> 4) & 3) << 4)) - 32
        q4 = (high_b | (((qh_half >> 6) & 3) << 4)) - 32
        base = 128 * half
        out[:, base:base + 32] = (q1.to(torch.float32)
                                  * wide_scale(sc_half[:, 0], sc_half[:, 1])
                                  * d.unsqueeze(1))
        out[:, base + 32:base + 64] = (q2.to(torch.float32)
                                       * wide_scale(sc_half[:, 2], sc_half[:, 3])
                                       * d.unsqueeze(1))
        out[:, base + 64:base + 96] = (q3.to(torch.float32)
                                       * wide_scale(sc_half[:, 4], sc_half[:, 5])
                                       * d.unsqueeze(1))
        out[:, base + 96:base + 128] = (q4.to(torch.float32)
                                        * wide_scale(sc_half[:, 6], sc_half[:, 7])
                                        * d.unsqueeze(1))
    return out


def _dequant_q8_k(raw: torch.Tensor) -> torch.Tensor:
    d = _f32_column(raw, 0)
    qs = _raw_bytes(raw, 4, 260).view(torch.int8).to(torch.float32)
    return qs * d.unsqueeze(1)


_DEQUANTIZERS = {
    2: _dequant_q4_0,
    3: _dequant_q4_1,
    6: lambda raw: _dequant_q5(raw, with_min=False),
    7: lambda raw: _dequant_q5(raw, with_min=True),
    8: _dequant_q8_0,
    12: lambda raw: _dequant_k(raw, five_bit=False),
    13: lambda raw: _dequant_k(raw, five_bit=True),
    14: _dequant_q6_k,
    15: _dequant_q8_k,
}


def dequantize_blocks(ggml_type: int, blocks: torch.Tensor) -> torch.Tensor:
    """[n_blocks, block_bytes] uint8 -> [n_blocks * block_elems] float32.

    Kept separate from the reader so the block math can be tested against
    hand-computed values without a GGUF file.
    """
    if ggml_type not in _DEQUANTIZERS:
        name = GGML_TYPES.get(ggml_type, ("unknown", 0, 0))[0]
        raise ValueError(f"ggml type {name} ({ggml_type}) has no dequantiser")
    _, block_elems, block_bytes = GGML_TYPES[ggml_type]
    if blocks.dtype != torch.uint8 or blocks.ndim != 2 or blocks.shape[1] != block_bytes:
        raise ValueError(f"expected uint8 blocks of {block_bytes} bytes, got "
                         f"{tuple(blocks.shape)} {blocks.dtype}")
    return _DEQUANTIZERS[ggml_type](blocks).reshape(-1)


# ------------------------------------------------------------ name mapping
# GGUF block-local names (llama.cpp convention) -> kaze/HF module weights.
_BLOCK_WEIGHTS = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.bias": "self_attn.o_proj.bias",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.router.weight",
}

# tensors llama.cpp writes that kaze either recomputes or does not need
_IGNORED_TOP_LEVEL = {"rope_freqs.weight", "token_types.weight"}

# llama.cpp's HF converter stores the llama-arch q/k rows in ggml's interleaved
# head layout (conversion/llama.py: LlamaModel.permute); llama4, apertus and
# mistral set undo_permute = False, and the qwen converters never permute.
_PERMUTE_QK_ARCHS = {"llama"}


def unpermute_qk(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    """Undo llama.cpp's q/k row permutation, back to the HF (head-major) layout."""
    if weight.ndim != 2 or n_head <= 0 or weight.shape[0] % n_head:
        raise ValueError(f"cannot unpermute {tuple(weight.shape)} with {n_head} heads")
    head_dim = weight.shape[0] // n_head
    if head_dim % 2:
        raise ValueError(f"cannot unpermute an odd head width ({head_dim})")
    return (weight.reshape(n_head, head_dim // 2, 2, *weight.shape[1:])
            .swapaxes(1, 2).reshape(weight.shape).contiguous())

# fused tensors that have to be split because kaze keeps the projections apart
# which config attribute holds the head count for a permuted projection
_PERMUTED_HEADS = {"attn_q.weight": "num_attention_heads",
                   "attn_k.weight": "num_key_value_heads"}

_FUSED = {
    "attn_qkv.weight": (("self_attn.q_proj.weight", "q"),
                        ("self_attn.k_proj.weight", "kv"),
                        ("self_attn.v_proj.weight", "kv")),
    "attn_qkv.bias": (("self_attn.q_proj.bias", "q"),
                      ("self_attn.k_proj.bias", "kv"),
                      ("self_attn.v_proj.bias", "kv")),
    "ffn_gate_up.weight": (("mlp.gate_proj.weight", "inter"),
                           ("mlp.up_proj.weight", "inter")),
}


def config_from_gguf(reader: GGUFReader) -> ModelConfig:
    """Build a ModelConfig from GGUF metadata (the file is self-describing)."""
    arch = reader.architecture

    def need(key: str) -> int:
        value = reader.metadata.get(f"{arch}.{key}")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{reader.path}: missing or invalid metadata {arch}.{key}")
        return value

    embedding = reader.tensors.get("token_embd.weight")
    if embedding is None or len(embedding.shape) != 2:
        raise ValueError(f"{reader.path}: no 2-D token_embd.weight tensor")
    hidden = need("embedding_length")
    heads = need("attention.head_count")
    if embedding.shape[1] != hidden:
        raise ValueError(f"{reader.path}: token_embd is {embedding.shape} but "
                         f"{arch}.embedding_length is {hidden}")
    kv_heads = reader.metadata.get(f"{arch}.attention.head_count_kv", heads)
    if not isinstance(kv_heads, int) or isinstance(kv_heads, bool) or kv_heads <= 0:
        raise ValueError(f"{reader.path}: invalid {arch}.attention.head_count_kv")
    head_dim = reader.metadata.get(f"{arch}.attention.key_length")
    if head_dim is None:
        head_dim = hidden // heads
    experts = reader.metadata.get(f"{arch}.expert_count", 0)
    if isinstance(experts, int) and experts > 0:
        raise ValueError(f"{reader.path}: mixture-of-experts GGUF is not supported yet "
                         f"({experts} experts); convert the fused ffn_*_exps tensors first")
    has_output = "output.weight" in reader.tensors
    scaling = reader.metadata.get(f"{arch}.rope.scaling.type")
    rope_scaling = None
    if isinstance(scaling, str) and scaling not in ("", "none"):
        rope_scaling = {"rope_type": scaling,
                        "factor": float(reader.metadata.get(f"{arch}.rope.scaling.factor", 1.0))}
        original = reader.metadata.get(f"{arch}.rope.scaling.original_context_length")
        if isinstance(original, int) and original > 0:
            rope_scaling["original_max_position_embeddings"] = original
    return ModelConfig(
        vocab_size=embedding.shape[0],
        hidden_size=hidden,
        intermediate_size=need("feed_forward_length"),
        num_hidden_layers=need("block_count"),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        rms_norm_eps=float(reader.metadata.get(
            f"{arch}.attention.layer_norm_rms_epsilon",
            reader.metadata.get(f"{arch}.attention.layer_norm_epsilon", 1e-6))),
        rope_theta=float(reader.metadata.get(f"{arch}.rope.freq_base", 10000.0)),
        max_position_embeddings=int(reader.metadata.get(f"{arch}.context_length", 32768)),
        tie_word_embeddings=not has_output,
        head_dim=int(head_dim),
        qkv_bias="blk.0.attn_q.bias" in reader.tensors,
        qk_norm="blk.0.attn_q_norm.weight" in reader.tensors,
        rope_scaling=rope_scaling,
    )


def _check_config_matches(hf: ModelConfig, gguf: ModelConfig, path: str) -> None:
    for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                 "num_attention_heads", "num_key_value_heads", "head_dim"):
        if getattr(hf, name) != getattr(gguf, name):
            raise ValueError(f"{path}: HF config {name}={getattr(hf, name)} disagrees with "
                             f"the GGUF weights ({getattr(gguf, name)})")


class GGUFWeights:
    """Adapter: a GGUF file presented as the HF-named weights kaze's loader wants.

    keys() and get() mirror the safetensors reader in kaze.loader, so
    load_model() can build the same KazeModel from either source.
    """

    def __init__(self, path: str, *, tokenizer_dir: str | None = None,
                 cfg: ModelConfig | None = None, verbose: bool = False):
        self.path = path
        self.reader = GGUFReader(path)
        try:
            gguf_cfg = config_from_gguf(self.reader)
            self.config_source = "gguf"
            resolved = cfg
            if resolved is None and tokenizer_dir:
                config_path = os.path.join(tokenizer_dir, "config.json")
                if os.path.exists(config_path):
                    resolved = ModelConfig.from_pretrained(tokenizer_dir)
                    _check_config_matches(resolved, gguf_cfg, path)
                    self.config_source = config_path
        except Exception:
            self.reader.close()
            raise
        self.cfg = resolved or gguf_cfg
        if not self.cfg.tie_word_embeddings and "output.weight" not in self.reader.tensors:
            # the file has no separate head, so the embedding matrix is reused
            self.cfg.tie_word_embeddings = True
        self._direct: dict[str, str] = {}
        self._split: dict[str, tuple[str, int, int]] = {}
        self._permute: dict[str, int] = {}
        self._keys: list[str] = []
        self._build_index()
        if verbose:
            print(f"  gguf: {os.path.basename(path)} v{self.reader.version}, "
                  f"{len(self.reader.tensors)} tensors, config from {self.config_source}",
                  flush=True)
        if (self.reader.metadata.get("general.architecture") in ("llama", "llama4")
                and "rope_freqs.weight" in self.reader.tensors
                and self.config_source == "gguf"):
            print("  warning: this GGUF carries rope_freqs (llama3-style rope scaling) but "
                  "no config.json was found; pass tokenizer_dir so the rope settings match "
                  "the weights", flush=True)

    # ---------------------------------------------------------------- index
    def _add(self, hf_name: str) -> None:
        self._keys.append(hf_name)

    def _build_index(self) -> None:
        cfg = self.cfg
        tensors = self.reader.tensors
        permute_qk = self.reader.architecture in _PERMUTE_QK_ARCHS
        for target, source in (("model.embed_tokens.weight", "token_embd.weight"),
                               ("model.norm.weight", "output_norm.weight"),
                               ("model.lm_head.weight", "output.weight")):
            if source in tensors:
                self._direct[target] = source
                self._add(target)
        prefix = "blk."
        top_level = {"token_embd.weight", "output_norm.weight", "output.weight"}
        for name in sorted(tensors):
            if not name.startswith(prefix):
                if name in top_level or name in _IGNORED_TOP_LEVEL:
                    continue
                raise ValueError(f"{self.path}: unexpected top-level tensor {name!r}")
            head, _, suffix = name[len(prefix):].partition(".")
            if not head.isdigit():
                raise ValueError(f"{self.path}: malformed block tensor {name!r}")
            layer = int(head)
            base = f"model.layers.{layer}."
            if suffix in _FUSED:
                offset = 0
                for hf_suffix, width in _FUSED[suffix]:
                    size = {"q": cfg.q_size, "kv": cfg.kv_size,
                            "inter": cfg.intermediate_size}[width]
                    key = base + hf_suffix
                    self._split[key] = (name, offset, offset + size)
                    self._add(key)
                    offset += size
                    if permute_qk:
                        heads = {"q": cfg.num_attention_heads,
                                 "kv": cfg.num_key_value_heads}.get(width)
                        if heads:
                            self._permute[key] = heads
                rows = tensors[name].shape[0]
                if offset != rows:
                    raise ValueError(f"{self.path}: {name} has {rows} rows but the config "
                                     f"implies {offset}; are the shapes right?")
            elif suffix in _BLOCK_WEIGHTS:
                key = base + _BLOCK_WEIGHTS[suffix]
                self._direct[key] = name
                self._add(key)
                if permute_qk:
                    heads = _PERMUTED_HEADS.get(suffix)
                    if heads:
                        self._permute[key] = getattr(cfg, heads)
            else:
                raise ValueError(f"{self.path}: unsupported block tensor {name!r} "
                                 "(see docs/gguf.md for the mapped names)")

    def keys(self) -> list[str]:
        return list(self._keys)

    # ----------------------------------------------------------------- data
    def get(self, name: str) -> torch.Tensor:
        if name in self._split:
            source, start, stop = self._split[name]
            tensor = self.reader.tensor(source)[start:stop].contiguous()
        elif name in self._direct:
            tensor = self.reader.tensor(self._direct[name])
        else:
            raise KeyError(name)
        heads = self._permute.get(name)
        return unpermute_qk(tensor, heads) if heads else tensor

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> "GGUFWeights":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def resolve_gguf(path: str) -> str | None:
    """The GGUF file named by path, or None when path is an HF weight directory.

    A directory counts as GGUF when it holds exactly one .gguf file, which is
    what a "config.json + tokenizer + model-Q4_K_M.gguf" directory looks like.
    """
    if os.path.isdir(path):
        found = sorted(f for f in os.listdir(path) if f.lower().endswith(".gguf"))
        if not found:
            return None
        if len(found) > 1:
            raise ValueError(f"{path}: more than one .gguf file ({', '.join(found)}); "
                             "pass the file itself")
        return os.path.join(path, found[0])
    if path.lower().endswith(".gguf"):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    return None
