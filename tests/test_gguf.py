"""GGUF reader and ggml block dequant regressions: python3 tests/test_gguf.py.

The block expectations here are written out from the ggml format definition
(no reference implementation is imported), and the integration test re-checks
one real GGUF file when bench/llamacpp/models has one.
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kaze.gguf import (GGUFReader, GGUFWeights, config_from_gguf,
                       dequantize_blocks, resolve_gguf, unpermute_qk)
from kaze.loader import load_model

ROOT = Path(__file__).resolve().parents[1]
QWEN_GGUF = ROOT / "bench/llamacpp/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
QWEN_HF = Path("/models/Qwen2.5-0.5B-Instruct")
LLAMA_GGUF = ROOT / "bench/llamacpp/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
LLAMA_HF = Path("/models/Llama-3.2-1B-Instruct")


# ------------------------------------------------------------------ helpers
def f16(value: float) -> bytes:
    return struct.pack("<e", value)


def f32(value: float) -> bytes:
    return struct.pack("<f", value)


def blocks(*rows: bytes) -> torch.Tensor:
    width = len(rows[0])
    assert all(len(r) == width for r in rows)
    return torch.frombuffer(bytearray(b"".join(rows)), dtype=torch.uint8).view(len(rows), width)


def kv(key: str, value) -> bytes:
    head = struct.pack("<Q", len(key.encode())) + key.encode()
    if isinstance(value, bool):
        return head + struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return head + struct.pack("<II", 4, value)
    if isinstance(value, float):
        return head + struct.pack("<I", 6) + f32(value)
    if isinstance(value, str):
        raw = value.encode()
        return head + struct.pack("<I", 8) + struct.pack("<Q", len(raw)) + raw
    if isinstance(value, list):
        return (head + struct.pack("<IIQ", 9, 4, len(value))
                + b"".join(struct.pack("<I", v) for v in value))
    raise TypeError(f"unsupported metadata value {value!r}")


def write_gguf(path, metadata: dict, tensors: list, alignment: int = 32,
               version: int = 3) -> None:
    """tensors: [(name, ggml_type, ggml_dims, raw_bytes)] - dims in ggml order."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata)))
    for key, value in metadata.items():
        out += kv(key, value)
    offsets, data = [], bytearray()
    for name, ggml_type, dims, raw in tensors:
        offsets.append(len(data))
        data += raw
        data += b"\x00" * ((-len(data)) % alignment)
    for (name, ggml_type, dims, raw), offset in zip(tensors, offsets):
        out += struct.pack("<Q", len(name.encode())) + name.encode()
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", ggml_type, offset)
    out += b"\x00" * ((-len(out)) % alignment)
    Path(path).write_bytes(bytes(out + data))


# --------------------------------------------------------- block dequant
class Q8_0(unittest.TestCase):
    def test_values_scale_the_signed_bytes(self):
        values = list(range(-16, 16))
        raw = f16(0.5) + bytes(v & 0xFF for v in values)
        got = dequantize_blocks(8, blocks(raw))
        self.assertEqual(got.tolist(), [0.5 * v for v in values])

    def test_wrong_block_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "uint8 blocks of 34 bytes"):
            dequantize_blocks(8, torch.zeros(1, 18, dtype=torch.uint8))

    def test_unknown_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no dequantiser"):
            dequantize_blocks(10, torch.zeros(1, 84, dtype=torch.uint8))


class Q4_0(unittest.TestCase):
    def test_nibbles_are_split_across_the_two_halves(self):
        # byte j holds value j in the low nibble and value j+16 in the high nibble
        qs = bytes((j & 0x0F) | ((0x0F - j) << 4) for j in range(16))
        got = dequantize_blocks(2, blocks(f16(0.25) + qs)).tolist()
        self.assertEqual(got[:16], [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                                    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
        self.assertEqual(got[16:], [1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0,
                                    -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0])


class Q5(unittest.TestCase):
    def test_q5_0_uses_the_high_bit_plane(self):
        raw = f16(1.0) + struct.pack("<I", 0) + bytes(16)
        self.assertEqual(dequantize_blocks(6, blocks(raw)).tolist(), [-16.0] * 32)
        # low nibbles 15 / high nibbles 0, then the same with every 5th bit set
        raw = f16(1.0) + struct.pack("<I", 0) + bytes([0x0F] * 16)
        got = dequantize_blocks(6, blocks(raw)).tolist()
        self.assertEqual(got[:16], [-1.0] * 16)                # 15 - 16
        self.assertEqual(got[16:], [-16.0] * 16)               # 0 - 16
        raw = f16(1.0) + struct.pack("<I", 0xFFFFFFFF) + bytes([0x0F] * 16)
        got = dequantize_blocks(6, blocks(raw)).tolist()
        self.assertEqual(got[:16], [15.0] * 16)                # 15 + 16 - 16
        self.assertEqual(got[16:], [0.0] * 16)                 # 0 + 16 - 16

    def test_q5_1_is_affine(self):
        # all nibbles 0, no high bits: every value is the minimum
        raw = f16(0.5) + f16(-1.5) + struct.pack("<I", 0) + bytes(16)
        self.assertEqual(dequantize_blocks(7, blocks(raw)).tolist(), [-1.5] * 32)


class Q4K(unittest.TestCase):
    @staticmethod
    def header(scale: float = 1.0, dmin: float = 0.0, scales: bytes = None) -> bytes:
        # scales for eight sub-blocks: scale=1, min=0 (the 6-bit pairs, parts 0-3 then 4-7)
        return f16(scale) + f16(dmin) + (scales or bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]))

    def test_low_and_high_nibbles_land_in_their_own_32_value_half(self):
        qs = bytes([0x21] * 32 + [0x32] * 32 + [0x43] * 32 + [0x54] * 32)
        got = dequantize_blocks(12, blocks(self.header() + qs)).tolist()
        for group, (low, high) in enumerate(((1, 2), (2, 3), (3, 4), (4, 5))):
            self.assertEqual(got[64 * group:64 * group + 32], [float(low)] * 32)
            self.assertEqual(got[64 * group + 32:64 * group + 64], [float(high)] * 32)

    def test_six_bit_min_is_subtracted_from_its_own_sub_block(self):
        # sub-block 2 (the low half of the second 64) gets min = 4
        header = self.header(dmin=1.0, scales=bytes([1, 1, 1, 1, 0, 0, 4, 0, 1, 1, 1, 1]))
        qs = bytes([0x21] * 128)
        got = dequantize_blocks(12, blocks(header + qs)).tolist()
        self.assertEqual(got[64:96], [-3.0] * 32)          # 1 * 1 - 1 * 4
        self.assertEqual(got[96:128], [2.0] * 32)          # sub-block 3 keeps min 0
        self.assertEqual(got[:32], [1.0] * 32)
        self.assertEqual(got[160:192], [2.0] * 32)

    def test_super_block_scale_multiplies_the_six_bit_scale(self):
        got = dequantize_blocks(12, blocks(self.header(scale=0.5) + bytes([0x21] * 128))).tolist()
        self.assertEqual(got[0], 0.5)
        self.assertEqual(got[32], 1.0)


class Q5K(unittest.TestCase):
    def test_high_bit_plane_is_shifted_per_64_value_group(self):
        # qh byte bit 1 << 2g is the 5th bit of the low half of group g
        header = Q4K.header() + bytes([0x01] * 32)
        got = dequantize_blocks(13, blocks(header + bytes([0xF0] * 128))).tolist()
        self.assertEqual(got[:32], [16.0] * 32)     # low nibble 0 + 16
        self.assertEqual(got[32:64], [15.0] * 32)   # high nibble 15
        self.assertEqual(got[64:128], [0.0] * 32 + [15.0] * 32)
        self.assertEqual(got[192:256], [0.0] * 32 + [15.0] * 32)


class Q6K(unittest.TestCase):
    @staticmethod
    def block(ql: bytes, qh: bytes, scales: bytes, d: float = 1.0) -> bytes:
        return ql + qh + scales + f16(d)

    def test_quarters_are_ordered_and_centred_on_32(self):
        ql = bytes([0x02] * 32 + [0x03] * 32) + bytes(64)
        raw = self.block(ql, bytes(64), bytes([1] * 16))
        got = dequantize_blocks(14, blocks(raw)).tolist()
        self.assertEqual(got[0:32], [-30.0] * 32)     # 2 - 32
        self.assertEqual(got[32:64], [-29.0] * 32)    # ql[32:] low nibble 3
        self.assertEqual(got[64:96], [-32.0] * 32)    # ql[0:32] high nibble 0
        self.assertEqual(got[96:128], [-32.0] * 32)
        self.assertEqual(got[128:256], [-32.0] * 128)

    def test_scale_index_changes_every_16_values(self):
        raw = self.block(bytes(128), bytes(64), bytes([2, 3] + [1] * 14))
        got = dequantize_blocks(14, blocks(raw)).tolist()
        self.assertEqual(got[0:16], [-64.0] * 16)     # -32 * 2
        self.assertEqual(got[16:32], [-96.0] * 16)    # -32 * 3

    def test_high_bits_are_packed_two_bits_per_quarter(self):
        raw = self.block(bytes(128), bytes([0xFF] * 64), bytes([1] * 16))
        got = dequantize_blocks(14, blocks(raw)).tolist()
        self.assertEqual(got[0], 16.0)                # (0 | 3 << 4) - 32
        self.assertEqual(got[96], 16.0)


class Q8K(unittest.TestCase):
    def test_float_scale_times_signed_bytes(self):
        qs = bytes(range(256))
        raw = f32(0.25) + qs + bytes(32)
        got = dequantize_blocks(15, blocks(raw)).tolist()
        self.assertEqual(got[:4], [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(got[127], 127 * 0.25)
        self.assertEqual(got[128], -128 * 0.25)


# ------------------------------------------------------------- file parsing
def tiny_meta(arch: str = "qwen2") -> dict:
    return {
        "general.architecture": arch,
        "general.alignment": 32,
        f"{arch}.embedding_length": 32,
        f"{arch}.feed_forward_length": 32,
        f"{arch}.block_count": 1,
        f"{arch}.attention.head_count": 2,
        f"{arch}.attention.head_count_kv": 1,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-5,
        f"{arch}.rope.freq_base": 10000.0,
        f"{arch}.context_length": 64,
        "tokenizer.ggml.tokens": [1, 2, 3],
    }


TINY_META = tiny_meta()


def tiny_tensors(scale: float = 1.0) -> list:
    """One-layer qwen2-shaped model; shapes are written in ggml (reversed) order."""
    hidden, inter, vocab = 32, 32, 32
    q_size, kv_size = 32, 16
    def dense(shape, base):
        count = 1
        for d in shape:
            count *= d
        values = [base + i * scale for i in range(count)]
        raw = b"".join(f32(v) for v in values)
        return raw
    emb = b"".join(f16(0.5) + bytes((i % 200) & 0xFF for i in range(32)) for _ in range(vocab))
    # one row per block: 32 columns is exactly one Q8_0 block
    return [
        ("token_embd.weight", 8, (hidden, vocab), emb),
        ("output_norm.weight", 0, (hidden,), b"".join(f32(1.0 + i) for i in range(hidden))),
        ("blk.0.attn_norm.weight", 0, (hidden,), b"".join(f32(0.5 + i) for i in range(hidden))),
        ("blk.0.ffn_norm.weight", 0, (hidden,), b"".join(f32(0.25 + i) for i in range(hidden))),
        ("blk.0.attn_qkv.weight", 0, (hidden, q_size + 2 * kv_size),
         dense((q_size + 2 * kv_size, hidden), 0.0)),
        ("blk.0.attn_output.weight", 0, (hidden, hidden), dense((hidden, hidden), 100.0)),
        ("blk.0.ffn_gate.weight", 0, (hidden, inter), dense((inter, hidden), 200.0)),
        ("blk.0.ffn_up.weight", 0, (hidden, inter), dense((inter, hidden), 300.0)),
        ("blk.0.ffn_down.weight", 0, (inter, hidden), dense((hidden, inter), 400.0)),
    ]


class Reader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "tiny.gguf"
        write_gguf(self.path, TINY_META, tiny_tensors())
        self.addCleanup(self._tmp.cleanup)

    def test_header_metadata_and_shapes(self):
        with GGUFReader(str(self.path)) as r:
            self.assertEqual(r.version, 3)
            self.assertEqual(r.n_tensors, 9)
            self.assertEqual(r.architecture, "qwen2")
            self.assertEqual(r.arch_value("embedding_length"), 32)
            self.assertEqual(r.metadata["tokenizer.ggml.tokens"], [1, 2, 3])
            self.assertAlmostEqual(r.metadata["qwen2.rope.freq_base"], 10000.0, places=3)
            self.assertEqual(r.data_offset % r.alignment, 0)
            info = r.tensors["blk.0.attn_qkv.weight"]
            self.assertEqual(info.shape, (64, 32))          # ggml dims are reversed
            self.assertEqual(info.type_name, "F32")
            self.assertEqual(r.tensors["token_embd.weight"].type_name, "Q8_0")
            self.assertEqual(r.tensors["token_embd.weight"].nbytes, 32 * 34)

    def test_version_two_files_parse_the_same_way(self):
        old = self.dir / "v2.gguf"
        write_gguf(old, TINY_META, tiny_tensors(), version=2)
        with GGUFReader(str(old)) as r:
            self.assertEqual(r.version, 2)
            self.assertEqual(r.tensors["token_embd.weight"].shape, (32, 32))

    def test_float_passthrough_and_dequantised_rows(self):
        with GGUFReader(str(self.path)) as r:
            norm = r.tensor("blk.0.attn_norm.weight")
            self.assertEqual(tuple(norm.shape), (32,))
            self.assertAlmostEqual(norm[0].item(), 0.5, places=6)
            self.assertAlmostEqual(norm[5].item(), 5.5, places=6)
            qkv = r.tensor("blk.0.attn_qkv.weight")
            self.assertAlmostEqual(qkv[0, 0].item(), 0.0, places=6)
            self.assertAlmostEqual(qkv[1, 3].item(), 35.0, places=6)
            emb = r.tensor("token_embd.weight")
            self.assertEqual(tuple(emb.shape), (32, 32))
            self.assertAlmostEqual(emb[0, 0].item(), 0.0, places=6)
            self.assertAlmostEqual(emb[0, 7].item(), 3.5, places=6)

    def test_truncated_file_is_rejected(self):
        data = self.path.read_bytes()
        short = self.dir / "short.gguf"
        short.write_bytes(data[:len(data) // 2])
        with self.assertRaisesRegex(ValueError, "truncated|past the end"):
            GGUFReader(str(short))

    def test_bad_magic_and_version_are_rejected(self):
        broken = self.dir / "bad.gguf"
        data = bytearray(self.path.read_bytes())
        broken.write_bytes(b"NOPE" + bytes(data[4:]))
        with self.assertRaisesRegex(ValueError, "not a GGUF"):
            GGUFReader(str(broken))
        broken.write_bytes(bytes(data[:4]) + struct.pack("<I", 9) + bytes(data[8:]))
        with self.assertRaisesRegex(ValueError, "unsupported GGUF version 9"):
            GGUFReader(str(broken))

    def test_unsupported_quant_type_fails_loudly(self):
        path = self.dir / "q2k.gguf"
        write_gguf(path, TINY_META,
                   [("token_embd.weight", 10, (32, 32), bytes(84))])
        with self.assertRaisesRegex(ValueError, "unsupported ggml type 10"):
            GGUFReader(str(path))

    def test_implausible_array_length_is_rejected(self):
        path = self.dir / "huge.gguf"
        out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
        key = b"general.architecture"
        out += struct.pack("<Q", len(key)) + key
        # a uint32 array that claims 2**40 elements in a 40-byte file
        out += struct.pack("<IIQ", 9, 4, 1 << 40)
        path.write_bytes(bytes(out))
        with self.assertRaisesRegex(ValueError, "does not fit the file"):
            GGUFReader(str(path))


# --------------------------------------------------------------- mapping
class Mapping(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "tiny.gguf"
        write_gguf(self.path, TINY_META, tiny_tensors())
        self.addCleanup(self._tmp.cleanup)

    def test_config_comes_from_the_file(self):
        with GGUFReader(str(self.path)) as r:
            cfg = config_from_gguf(r)
        self.assertEqual((cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size), (32, 32, 32))
        self.assertEqual(cfg.num_key_value_heads, 1)
        self.assertEqual(cfg.head_dim, 16)                 # hidden / heads
        self.assertTrue(cfg.tie_word_embeddings)           # no output.weight
        self.assertFalse(cfg.qkv_bias)                     # no attn_q.bias
        self.assertAlmostEqual(cfg.rms_norm_eps, 1e-5, places=9)

    def test_gguf_names_map_to_kaze_modules_and_qkv_is_split(self):
        weights = GGUFWeights(str(self.path))
        self.addCleanup(weights.close)
        keys = set(weights.keys())
        for expected in ("model.embed_tokens.weight", "model.norm.weight",
                         "model.layers.0.input_layernorm.weight",
                         "model.layers.0.post_attention_layernorm.weight",
                         "model.layers.0.self_attn.q_proj.weight",
                         "model.layers.0.self_attn.k_proj.weight",
                         "model.layers.0.self_attn.v_proj.weight",
                         "model.layers.0.self_attn.o_proj.weight",
                         "model.layers.0.mlp.gate_proj.weight",
                         "model.layers.0.mlp.up_proj.weight",
                         "model.layers.0.mlp.down_proj.weight"):
            self.assertIn(expected, keys)
        self.assertNotIn("model.lm_head.weight", keys)     # tied embeddings
        q = weights.get("model.layers.0.self_attn.q_proj.weight")
        k = weights.get("model.layers.0.self_attn.k_proj.weight")
        v = weights.get("model.layers.0.self_attn.v_proj.weight")
        self.assertEqual((q.shape, k.shape, v.shape), ((32, 32), (16, 32), (16, 32)))
        # attn_qkv rows are filled with 0, 1, 2, ... so the split is visible
        self.assertAlmostEqual(q[0, 1].item(), 1.0, places=5)
        self.assertAlmostEqual(k[0, 0].item(), 32 * 32, places=5)
        self.assertAlmostEqual(v[0, 0].item(), 48 * 32, places=5)

    def test_hf_config_is_checked_against_the_weights(self):
        hf = self.dir / "hf"
        hf.mkdir()
        (hf / "config.json").write_text('{"vocab_size": 32, "hidden_size": 64, '
                                        '"intermediate_size": 32, "num_hidden_layers": 1, '
                                        '"num_attention_heads": 2, "num_key_value_heads": 1}',
                                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hidden_size=64 disagrees"):
            GGUFWeights(str(self.path), tokenizer_dir=str(hf))

    def test_resolve_gguf_accepts_a_single_file_directory(self):
        self.assertEqual(resolve_gguf(str(self.path)), str(self.path))
        self.assertEqual(resolve_gguf(str(self.dir)), str(self.path))
        second = self.dir / "other.gguf"
        write_gguf(second, TINY_META, tiny_tensors())
        with self.assertRaisesRegex(ValueError, "more than one"):
            resolve_gguf(str(self.dir))
        hf = self.dir / "hf"
        hf.mkdir()
        self.assertIsNone(resolve_gguf(str(hf)))
        with self.assertRaises(FileNotFoundError):
            resolve_gguf(str(self.dir / "missing.gguf"))

    def test_load_model_builds_a_kaze_model_from_a_gguf(self):
        model = load_model(str(self.path), kind="fp16", device="cpu", verbose=False)
        cfg = model.cfg
        self.assertEqual(cfg.num_hidden_layers, 1)
        self.assertEqual(tuple(model.embed.shape), (32, 32))
        layer = model.layers[0]
        self.assertEqual(layer.wqkv.n, 64)                 # q + k + v fused again
        self.assertEqual(layer.wo.n, 32)
        self.assertEqual(layer.wgate_up.n, 64)
        self.assertEqual(layer.wdown.n, 32)
        self.assertEqual(model.lm_head.n, 32)              # tied head
        self.assertGreater(model.weight_bytes(), 0)


# ---------------------------------------------------- llama q/k permutation
def converter_permute(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    """The formula in llama.cpp's conversion/llama.py LlamaModel.permute."""
    d = weight.shape[0] // n_head
    return (weight.reshape(n_head, 2, d // 2, *weight.shape[1:])
            .swapaxes(1, 2).reshape(weight.shape))


def llama_tensors(permuted: bool = True) -> list:
    """Separate q/k/v like a real llama GGUF; q/k rows in ggml's layout."""
    hidden, inter, vocab = 32, 32, 32
    q_dim, kv_dim = 32, 16
    q = torch.arange(q_dim * hidden, dtype=torch.float32).reshape(q_dim, hidden)
    k = torch.arange(kv_dim * hidden, dtype=torch.float32).reshape(kv_dim, hidden)
    if permuted:
        q = converter_permute(q, 2)          # 2 query heads of 16
        k = converter_permute(k, 1)          # 1 kv head of 16
    emb = b"".join(f16(0.5) + bytes((i % 200) & 0xFF for i in range(32)) for _ in range(vocab))
    def raw(t: torch.Tensor) -> bytes:
        return t.contiguous().numpy().tobytes()
    return [
        ("token_embd.weight", 8, (hidden, vocab), emb),
        ("output_norm.weight", 0, (hidden,), raw(torch.ones(hidden))),
        ("blk.0.attn_norm.weight", 0, (hidden,), raw(torch.ones(hidden))),
        ("blk.0.ffn_norm.weight", 0, (hidden,), raw(torch.ones(hidden))),
        ("blk.0.attn_q.weight", 0, (hidden, q_dim), raw(q)),
        ("blk.0.attn_k.weight", 0, (hidden, kv_dim), raw(k)),
        ("blk.0.attn_v.weight", 0, (hidden, kv_dim),
         raw(torch.arange(kv_dim * hidden, dtype=torch.float32).reshape(kv_dim, hidden))),
        ("blk.0.attn_output.weight", 0, (hidden, hidden), raw(torch.ones(hidden, hidden))),
        ("blk.0.ffn_gate.weight", 0, (hidden, inter), raw(torch.ones(inter, hidden))),
        ("blk.0.ffn_up.weight", 0, (hidden, inter), raw(torch.ones(inter, hidden))),
        ("blk.0.ffn_down.weight", 0, (inter, hidden), raw(torch.ones(hidden, inter))),
    ]


class LlamaPermutation(unittest.TestCase):
    def test_unpermute_reverses_the_converter_formula(self):
        q = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
        for heads in (2, 4):
            self.assertTrue(torch.equal(unpermute_qk(converter_permute(q, heads), heads), q))
        with self.assertRaisesRegex(ValueError, "cannot unpermute"):
            unpermute_qk(q[:30], 2)          # 30 rows is not a whole number of heads

    def test_llama_q_and_k_are_restored_to_head_major_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llama.gguf"
            write_gguf(path, tiny_meta("llama"), llama_tensors())
            weights = GGUFWeights(str(path))
            self.addCleanup(weights.close)
            q = weights.get("model.layers.0.self_attn.q_proj.weight")
            k = weights.get("model.layers.0.self_attn.k_proj.weight")
            self.assertTrue(torch.equal(q, torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)))
            self.assertTrue(torch.equal(k, torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)))

    def test_qwen_files_are_not_permuted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen.gguf"
            write_gguf(path, tiny_meta("qwen2"), llama_tensors())
            weights = GGUFWeights(str(path))
            self.addCleanup(weights.close)
            q = weights.get("model.layers.0.self_attn.q_proj.weight")
            self.assertFalse(torch.equal(q, torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)))


# ------------------------------------------------------------ integration
@unittest.skipUnless(QWEN_GGUF.exists(), f"missing {QWEN_GGUF}")
class RealQwenGguf(unittest.TestCase):
    """Reads the Q4_K_M file that bench/llamacpp/models keeps for comparisons."""

    def test_real_file_dequantises_to_the_huggingface_weights(self):
        with GGUFReader(str(QWEN_GGUF)) as r:
            cfg = config_from_gguf(r)
            self.assertEqual((cfg.vocab_size, cfg.hidden_size), (151936, 896))
            q8 = r.tensor("token_embd.weight")
            self.assertEqual(tuple(q8.shape), (151936, 896))
            self.assertTrue(torch.isfinite(q8).all().item())
            # attn_q is Q5_0 in this file: the dequantised row must track the fp16 original
            if not (QWEN_HF / "model.safetensors").exists():
                self.skipTest("HF sibling weights are absent")
            from safetensors import safe_open
            with safe_open(str(QWEN_HF / "model.safetensors"), framework="pt",
                           device="cpu") as f:
                ref = f.get_tensor("model.layers.0.self_attn.q_proj.weight").float()
            got = r.tensor("blk.0.attn_q.weight", torch.float32)
            cos = torch.nn.functional.cosine_similarity(
                got[:64].flatten(), ref[:64].flatten(), dim=0).item()
            self.assertGreater(cos, 0.995)


@unittest.skipUnless(LLAMA_GGUF.exists() and (LLAMA_HF / "model.safetensors").exists(),
                     "needs the Llama-3.2-1B GGUF and its HF sibling")
class RealLlamaGguf(unittest.TestCase):
    """The llama arch stores q/k permuted, so this pins the restored layout."""

    def test_q_projections_match_the_huggingface_weights_after_unpermute(self):
        from safetensors import safe_open
        with GGUFReader(str(LLAMA_GGUF)) as r:
            self.assertIn("rope_freqs.weight", r.tensors)      # dropped, not mapped
            got = GGUFWeights(str(LLAMA_GGUF))
            self.addCleanup(got.close)
            self.assertEqual(got.cfg.num_key_value_heads, 8)
            q = got.get("model.layers.0.self_attn.q_proj.weight").float()
            k = got.get("model.layers.0.self_attn.k_proj.weight").float()
        with safe_open(str(LLAMA_HF / "model.safetensors"), framework="pt", device="cpu") as f:
            for mine, name in ((q, "model.layers.0.self_attn.q_proj.weight"),
                               (k, "model.layers.0.self_attn.k_proj.weight")):
                ref = f.get_tensor(name).float()
                cos = torch.nn.functional.cosine_similarity(
                    mine.flatten(), ref.flatten(), dim=0).item()
                self.assertGreater(cos, 0.99)


if __name__ == "__main__":
    unittest.main()
