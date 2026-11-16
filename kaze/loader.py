"""Safetensors -> KazeModel, with on-the-fly weight-only quantization."""
from __future__ import annotations

import gc
import json
import os
from typing import Iterator

import torch
from safetensors import safe_open

from kaze.config import ModelConfig
from kaze.model import KazeModel, LayerWeights
from kaze.quant import QuantLinear, concat_rows


import struct

ST_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
             "F64": torch.float64, "I32": torch.int32, "I64": torch.int64,
             "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def _read_safetensors_header(path: str) -> tuple[int, dict]:
    """(byte offset of the data section, {tensor name: header entry})."""
    # Use the format's validator without materializing any tensors or faulting
    # their pages; keep sequential reads for the actual weight data below.
    try:
        with safe_open(path, framework="pt", device="cpu"):
            pass
    except Exception as exc:
        raise ValueError(f"invalid safetensors file {path}: {exc}") from exc
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    return base, {k: v for k, v in hdr.items() if k != "__metadata__"}


def _shard_files(path: str) -> list[str]:
    """safetensors first, then the older PyTorch .bin shards (some fine-tunes)."""
    for index_name, pattern in (("model.safetensors.index.json", ".safetensors"),
                                ("pytorch_model.bin.index.json", ".bin")):
        idx = os.path.join(path, index_name)
        if os.path.exists(idx):
            with open(idx, encoding="utf-8") as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            root = os.path.realpath(path)
            for shard in shards:
                if (not isinstance(shard, str) or os.path.isabs(shard)
                        or os.path.commonpath([root, os.path.realpath(os.path.join(root, shard))]) != root
                        or not shard.endswith(pattern)):
                    raise ValueError(f"invalid shard path: {shard!r}")
            if not shards:
                raise ValueError(f"empty shard index: {idx}")
            return [os.path.join(path, s) for s in shards]
        files = sorted(f for f in os.listdir(path) if f.endswith(pattern))
        if files:
            return [os.path.join(path, f) for f in files]
    raise FileNotFoundError(f"no safetensors or .bin weights in {path}")


class _Reader:
    """Reads safetensors with plain sequential reads.

    mmap (safe_open) is faster on a local SSD, but on a network-backed
    filesystem it degrades badly: every faulting page costs a round trip and the
    kernel cannot keep 28 GB resident, which turns a 30 s load into an hour.
    One pread per tensor keeps the readahead working (measured 1.6 GB/s on a
    Colab VM vs 6 MB/s through mmap).
    """

    def __init__(self, path: str):
        self.files = _shard_files(path)
        self.index: dict[str, tuple[int, dict]] = {}
        self.bin: dict[str, dict] = {}
        self._keys: list[str] = []
        for f in self.files:
            previous_keys = set(self._keys)
            if f.endswith(".safetensors"):
                base, meta = _read_safetensors_header(f)
                self.index[f] = (base, meta)
                self._keys.extend(meta.keys())
            else:
                d = torch.load(f, map_location="cpu", weights_only=True)
                if not isinstance(d, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                                                 for k, v in d.items()):
                    raise ValueError(f"{f}: expected a tensor weight mapping")
                self.bin[f] = d
                self._keys.extend(list(d.keys()))
            duplicate = previous_keys.intersection(self._keys[len(previous_keys):])
            if duplicate:
                raise ValueError(f"duplicate tensor names across shards: {sorted(duplicate)}")

    def keys(self) -> list[str]:
        return self._keys

    def get(self, name: str) -> torch.Tensor:
        for f, (base, meta) in self.index.items():
            info = meta.get(name)
            if info is None:
                continue
            a, b = info["data_offsets"]
            dtype = ST_DTYPES[info["dtype"]]
            with open(f, "rb", buffering=0) as fh:
                fh.seek(base + a)
                buf = fh.read(b - a)
            if len(buf) != b - a:
                raise IOError(f"{name}: short read from {f} ({len(buf)} of {b - a} bytes); "
                              f"the file is truncated - re-download it")
            if not buf:
                return torch.empty(info["shape"], dtype=dtype)
            flat = torch.frombuffer(bytearray(buf), dtype=dtype)
            return flat.reshape(info["shape"])
        for f, d in self.bin.items():
            if name in d:
                return d.pop(name)
        raise KeyError(name)


def _q(t: torch.Tensor, bias: torch.Tensor | None, kind: str, group: int, device,
       ffn: bool = False) -> QuantLinear:
    """ffn=True marks the MLP tensors, which can take a wider quantisation step.

    kind="mixed" keeps attention at int8 (scores are sensitive) and puts the MLP at
    int4 (it holds most of the bytes but tolerates the coarser grid).
    """
    if kind == "mixed":
        kind = "int4" if ffn else "int8"
    return QuantLinear.from_float(t.to(device, torch.float32), None if bias is None else bias.to(device),
                                  kind=kind, group=group)


def load_model(path: str, kind: str = "int8", group: int = 128, device: str = "cuda",
               dtype: torch.dtype = torch.float16, lm_head_kind: str | None = None,
               verbose: bool = True) -> KazeModel:
    # Only these quantisation group widths are verified on the GPU paths.  group 32 hits an
    # illegal memory access in the int4 decode matvec (measured 2026-11-16), and anything
    # outside {64, 128} would at best be untested: fail loudly instead of risking a
    # corrupted CUDA context or silently wrong output.  64 is the quality option: KL
    # 0.0126 -> 0.0064 on 0.5B for -3% decode (bench/kl_multi.py, docs/evidence.md).
    if kind in ("int4", "int8") and group not in (64, 128) and device != "cpu":
        raise ValueError(
            f"group {group} is not supported for {kind} weights on the GPU: use 64 or 128 "
            "(128 = fastest, 64 = half the KL for -3% decode). Other widths are untested "
            "and group 32 crashes the int4 decode kernel with an illegal memory access.")
    cfg = ModelConfig.from_pretrained(path)
    dev = torch.device(device)
    reader = _Reader(path)
    keys = reader.keys()
    prefix = "model." if any(k.startswith("model.embed_tokens") for k in keys) else ""
    def checked(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        tensor = reader.get(name)
        if tuple(tensor.shape) != shape or not tensor.is_floating_point():
            raise ValueError(f"{name}: expected floating weight shape {shape}, got {tuple(tensor.shape)} ({tensor.dtype})")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}: weight must be finite")
        return tensor

    embed_w = checked(prefix + "embed_tokens.weight", (cfg.vocab_size, cfg.hidden_size))
    embed = embed_w.to(dev, dtype).contiguous()
    del embed_w
    gc.collect()

    layers: list[LayerWeights] = []
    p = prefix
    for i in range(cfg.num_hidden_layers):
        b = f"{p}layers.{i}."
        if any(k.startswith((b + "mlp.", b + "block_sparse_moe.")) and k.endswith(".bias") for k in keys):
            raise ValueError(f"{b}mlp: bias weights are not supported")
        def g(name: str) -> torch.Tensor:
            shapes = {
                "input_layernorm.weight": (cfg.hidden_size,),
                "post_attention_layernorm.weight": (cfg.hidden_size,),
                "self_attn.q_proj.weight": (cfg.q_size, cfg.hidden_size),
                "self_attn.k_proj.weight": (cfg.kv_size, cfg.hidden_size),
                "self_attn.v_proj.weight": (cfg.kv_size, cfg.hidden_size),
                "self_attn.o_proj.weight": (cfg.hidden_size, cfg.q_size),
                "self_attn.q_proj.bias": (cfg.q_size,),
                "self_attn.k_proj.bias": (cfg.kv_size,),
                "self_attn.v_proj.bias": (cfg.kv_size,),
                "self_attn.o_proj.bias": (cfg.hidden_size,),
                "self_attn.q_norm.weight": (cfg.head_dim,),
                "self_attn.k_norm.weight": (cfg.head_dim,),
                "mlp.gate_proj.weight": (cfg.intermediate_size, cfg.hidden_size),
                "mlp.up_proj.weight": (cfg.intermediate_size, cfg.hidden_size),
                "mlp.down_proj.weight": (cfg.hidden_size, cfg.intermediate_size),
            }
            return checked(b + name, shapes[name])
        an = g("input_layernorm.weight").to(dev, dtype)
        fn = g("post_attention_layernorm.weight").to(dev, dtype)
        wq = g("self_attn.q_proj.weight")
        wk = g("self_attn.k_proj.weight")
        wv = g("self_attn.v_proj.weight")
        bq = g("self_attn.q_proj.bias") if b + "self_attn.q_proj.bias" in keys else None
        bk = g("self_attn.k_proj.bias") if b + "self_attn.k_proj.bias" in keys else None
        bv = g("self_attn.v_proj.bias") if b + "self_attn.v_proj.bias" in keys else None
        wo = g("self_attn.o_proj.weight")
        bo = g("self_attn.o_proj.bias") if b + "self_attn.o_proj.bias" in keys else None
        is_moe = (b + "mlp.router.weight" in keys or b + "block_sparse_moe.gate.weight" in keys)
        if not is_moe:
            wg = g("mlp.gate_proj.weight")
            wu = g("mlp.up_proj.weight")
            wd = g("mlp.down_proj.weight")
        qkv_parts = []
        for w, bias in ((wq, bq), (wk, bk), (wv, bv)):
            qkv_parts.append(_q(w, bias, kind, group, dev))
            del w
        wqkv = concat_rows(qkv_parts)
        del qkv_parts
        wo_q = _q(wo, bo, kind, group, dev)
        del wo
        if is_moe:
            gu = wdown = None
        else:
            gu = concat_rows([_q(wg, None, kind, group, dev, ffn=True), _q(wu, None, kind, group, dev, ffn=True)])
            del wg, wu
            wdown = _q(wd, None, kind, group, dev, ffn=True)
            del wd
        qn = g("self_attn.q_norm.weight").to(dev, dtype) if b + "self_attn.q_norm.weight" in keys else None
        kn = g("self_attn.k_norm.weight").to(dev, dtype) if b + "self_attn.k_norm.weight" in keys else None
        if (qn is None) != (kn is None) or (cfg.qk_norm and qn is None):
            raise ValueError(f"{b}self_attn: q_norm and k_norm weights must be present together")
        router = wgu_e = wdn_e = None
        moe_i = 0
        if b + "mlp.router.weight" in keys or b + "block_sparse_moe.gate.weight" in keys:
            rk = b + "mlp.router.weight" if b + "mlp.router.weight" in keys else b + "block_sparse_moe.gate.weight"
            router = checked(rk, (cfg.num_experts, cfg.hidden_size)).to(dev, dtype).contiguous()
            n_exp = router.shape[0]
            if b + "mlp.experts.gate_up_proj" in keys:          # fused per-expert tensors
                wgu = reader.get(b + "mlp.experts.gate_up_proj")
                wdn = reader.get(b + "mlp.experts.down_proj")
                moe_i = wgu.shape[1] // 2
                wgu_e = _q(wgu.reshape(n_exp * 2 * moe_i, wgu.shape[2]), None, kind, group, dev, ffn=True)
                wdn_e = _q(wdn.reshape(n_exp * wdn.shape[1], wdn.shape[2]), None, kind, group, dev, ffn=True)
            else:                                               # one tensor per expert
                gu_parts, dn_parts = [], []
                for e in range(n_exp):
                    pre = (b + f"mlp.experts.{e}." if b + f"mlp.experts.{e}.gate_proj.weight" in keys
                           else b + f"block_sparse_moe.experts.{e}.")
                    gu_parts.append(concat_rows([_q(reader.get(pre + "gate_proj.weight"), None, kind, group, dev, ffn=True),
                                                 _q(reader.get(pre + "up_proj.weight"), None, kind, group, dev, ffn=True)]))
                    dn_parts.append(_q(reader.get(pre + "down_proj.weight"), None, kind, group, dev, ffn=True))
                moe_i = gu_parts[0].n // 2
                wgu_e = concat_rows(gu_parts)
                wdn_e = concat_rows(dn_parts)
                del gu_parts, dn_parts
        layers.append(LayerWeights(an, wqkv, wo_q, fn, gu, wdown, qn, kn, router, wgu_e, wdn_e, moe_i))
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if verbose:
            print(f"  layer {i + 1}/{cfg.num_hidden_layers}", flush=True)

    final_norm = checked(p + "norm.weight", (cfg.hidden_size,)).to(dev, dtype)
    lm_kind = lm_head_kind or ("int8" if kind == "mixed" else kind)
    head_key = next((k for k in (p + "lm_head.weight", "lm_head.weight") if k in keys), None)
    if head_key is not None:
        lm = _q(checked(head_key, (cfg.vocab_size, cfg.hidden_size)), None, lm_kind, group, dev)
    elif cfg.tie_word_embeddings:
        lm = QuantLinear.from_float(embed.float(), None, kind=lm_kind, group=group)
    else:
        raise KeyError("lm_head.weight is missing but tie_word_embeddings is False")
    model = KazeModel(cfg, embed, layers, final_norm, lm, dev, dtype=dtype)
    if verbose:
        print(f"  weights: {model.weight_bytes() / 1e6:.1f} MB", flush=True)
    if lm_kind == "int4" and lm_head_kind is None:
        # measured on 0.5B: int8 head takes KL from 0.093 to 0.069 for 9% less speed and
        # 13% more weight memory.  The smaller the model, the more this matters.
        if cfg.hidden_size * cfg.vocab_size <= 2.0e8:
            print("  hint: --lm-head-kind int8 improves KL by ~26% here for ~9% slower "
                  "decode", flush=True)
    return model
