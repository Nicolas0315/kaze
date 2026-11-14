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
            return [os.path.join(path, s) for s in shards]
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if files:
        return [os.path.join(path, f) for f in files]
    files = sorted(f for f in os.listdir(path)
                   if f.endswith(".bin") and not f.endswith(".index.json"))
    if not files:
        raise FileNotFoundError(f"no safetensors or .bin weights in {path}")
    return [os.path.join(path, f) for f in files]


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
            if f.endswith(".safetensors"):
                base, meta = _read_safetensors_header(f)
                self.index[f] = (base, meta)
                self._keys.extend(meta.keys())
            else:
                d = torch.load(f, map_location="cpu", weights_only=True)
                self.bin[f] = d
                self._keys.extend(list(d.keys()))

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


def _resolve_model_path(path: str) -> str:
    """Accept a Hugging Face repo id as well as a local directory.

    ("Qwen/Qwen2.5-0.5B-Instruct" is downloaded into the HF cache on first use.)
    """
    if os.path.isdir(path):
        return path
    if path.count("/") == 1 and not os.path.exists(path):
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=path)
        except Exception:
            return path
    return path


def load_model(path: str, kind: str = "int8", group: int = 128, device: str = "cuda",
               dtype: torch.dtype = torch.float16, lm_head_kind: str | None = None,
               verbose: bool = True) -> KazeModel:
    path = _resolve_model_path(path)
    cfg = ModelConfig.from_pretrained(path)
    dev = torch.device(device)
    reader = _Reader(path)
    keys = reader.keys()
    prefix = "model." if any(k.startswith("model.embed_tokens") for k in keys) else ""
    embed_w = reader.get(prefix + "embed_tokens.weight")
    embed = embed_w.to(dev, dtype).contiguous()
    del embed_w
    gc.collect()

    layers: list[LayerWeights] = []
    p = prefix
    for i in range(cfg.num_hidden_layers):
        b = f"{p}layers.{i}."
        def g(name: str) -> torch.Tensor:
            return reader.get(b + name)
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
        router = wgu_e = wdn_e = None
        moe_i = 0
        if b + "mlp.router.weight" in keys or b + "block_sparse_moe.gate.weight" in keys:
            rk = b + "mlp.router.weight" if b + "mlp.router.weight" in keys else b + "block_sparse_moe.gate.weight"
            router = reader.get(rk).to(dev, dtype).contiguous()
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
                                                 _q(reader.get(pre + "up_proj.weight"), None, kind, group, dev)]))
                    dn_parts.append(_q(reader.get(pre + "down_proj.weight"), None, kind, group, dev, ffn=True))
                moe_i = gu_parts[0].n // 2
                wgu_e = _stack_quant(gu_parts)
                wdn_e = _stack_quant(dn_parts)
                del gu_parts, dn_parts
        layers.append(LayerWeights(an, wqkv, wo_q, fn, gu, wdown, qn, kn, router, wgu_e, wdn_e, moe_i))
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(f"  layer {i + 1}/{cfg.num_hidden_layers}", flush=True)

    final_norm = reader.get(p + "norm.weight").to(dev, dtype)
    lm_kind = lm_head_kind or ("int8" if kind == "mixed" else kind)
    if p + "lm_head.weight" in keys:
        lm = _q(reader.get(p + "lm_head.weight"), None, lm_kind, group, dev)
    else:
        lm = QuantLinear.from_float(embed.float(), None, kind=lm_kind, group=group)
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
