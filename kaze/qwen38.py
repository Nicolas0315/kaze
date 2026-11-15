"""Qwen3.8（Qwen3_5 text）の実装。

このアーキテクチャは full attention と gated delta net（線形注意）が交互に並ぶ hybrid 構成で、
attention 側には出力ゲートと部分 RoPE が付きます。ここでは**正しさ優先**で書いていて、
線形注意の再帰はトークンごとに素直に回します（速くはないが、参照実装と同じ計算）。

履歴を持たないので decode でも毎回プロンプト全体を計算し直します。
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(positions: torch.Tensor, dim: int, theta: float, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = positions.float()[:, None] * inv[None, :]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
    return cos, sin


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float,
             plus_one: bool = True) -> torch.Tensor:
    """Qwen3_5 の RMSNorm は重みを (1 + w) として掛ける（初期値はゼロ）。"""
    v = x.float().pow(2).mean(-1, keepdim=True)
    y = x.float() * torch.rsqrt(v + eps)
    y = y * (1.0 + w.float()) if plus_one else y * w.float()
    return y.to(x.dtype)


class Mlp:
    def __init__(self, prefix: str, sd: dict, dtype, device):
        g = lambda n: sd[f"{prefix}.{n}"].to(device=device, dtype=dtype)
        self.gate, self.up, self.down = g("gate_proj.weight"), g("up_proj.weight"), g("down_proj.weight")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (F.silu(x @ self.gate.T) * (x @ self.up.T)) @ self.down.T


class FullAttention:
    """full attention 層。q_proj は出力ゲートと半分ずつに分かれる。"""

    def __init__(self, cfg, prefix: str, sd: dict, dtype, device):
        g = lambda n: sd[f"{prefix}.{n}"].to(device=device, dtype=dtype)
        self.q, self.k, self.v, self.o = (g("q_proj.weight"), g("k_proj.weight"),
                                          g("v_proj.weight"), g("o_proj.weight"))
        self.qn, self.kn = g("q_norm.weight"), g("k_norm.weight")
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.rope_dim = int(self.hd * cfg.partial_rotary_factor)
        self.eps, self.gate_type = cfg.rms_norm_eps, cfg.output_gate_type
        self.scale = self.hd ** -0.5

    def __call__(self, x, cos, sin):
        b, t, _ = x.shape
        qg = (x @ self.q.T).view(b, t, self.nh, self.hd * 2)
        q, gate = qg.chunk(2, dim=-1)
        q = rms_norm(q, self.qn, self.eps)
        k = rms_norm((x @ self.k.T).view(b, t, self.nkv, self.hd), self.kn, self.eps)
        v = (x @ self.v.T).view(b, t, self.nkv, self.hd)
        c, s, rd = cos[:, :, None, :], sin[:, :, None, :], self.rope_dim
        if rd < self.hd:
            q = torch.cat([q[..., :rd] * c + rotate_half(q[..., :rd]) * s, q[..., rd:]], -1)
            k = torch.cat([k[..., :rd] * c + rotate_half(k[..., :rd]) * s, k[..., rd:]], -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale,
                                             enable_gqa=self.nh != self.nkv)
        out = out.transpose(1, 2).reshape(b, t, self.nh * self.hd)
        gate = gate.reshape(b, t, -1)
        # 参照実装は output_gate_type に関わらず sigmoid を掛ける
        gate = torch.sigmoid(gate)
        return (out * gate.float()).to(x.dtype) @ self.o.T


class GatedDeltaNet:
    """線形注意層（gated delta rule）。参照実装と同じ逐次更新。"""

    def __init__(self, cfg, prefix: str, sd: dict, dtype, device):
        g = lambda n: sd[f"{prefix}.{n}"].to(device=device, dtype=dtype)
        self.qkv, self.z = g("in_proj_qkv.weight"), g("in_proj_z.weight")
        self.b_w, self.a_w = g("in_proj_b.weight"), g("in_proj_a.weight")
        self.conv_w = g("conv1d.weight")
        self.A_log = sd[f"{prefix}.A_log"].to(device=device, dtype=torch.float32)
        self.dt_bias = sd[f"{prefix}.dt_bias"].to(device=device, dtype=torch.float32)
        self.norm_w, self.out = g("norm.weight"), g("out_proj.weight")
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.hk, self.hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.eps = cfg.rms_norm_eps
        self.kdim, self.vdim = self.nk * self.hk, self.nv * self.hv

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = torch.split(x @ self.qkv.T, [self.kdim, self.kdim, self.vdim], dim=-1)
        z = x @ self.z.T
        beta = torch.sigmoid(x @ self.b_w.T)
        a = x @ self.a_w.T
        # depthwise 因果 conv（活性 silu）
        seq = torch.cat([q, k, v], -1).transpose(1, 2)
        w = self.conv_w.squeeze(1)
        ksz = w.shape[-1]
        pad = F.pad(seq, (ksz - 1, 0))
        conv = sum(w[:, i][None, :, None] * pad[:, :, i:i + t] for i in range(ksz))
        q, k, v = torch.split(F.silu(conv).transpose(1, 2), [self.kdim, self.kdim, self.vdim], -1)
        q = q.view(b, t, self.nk, self.hk)
        k = k.view(b, t, self.nk, self.hk)
        v = v.view(b, t, self.nv, self.hv)
        if self.nv > self.nk:
            rep = self.nv // self.nk
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        decay = -self.A_log.exp()[None, None, :] * F.softplus(a.float() + self.dt_bias)
        # 参照カーネルは use_qk_l2norm_in_kernel=True なので、q と k を head ごとに L2 正規化する
        q = F.normalize(q.float(), dim=-1, eps=1e-6) / (self.hk ** 0.5)
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        v = v.float()
        state = torch.zeros((b, self.nv, self.hk, self.hv), dtype=torch.float32, device=x.device)
        out = torch.zeros((b, t, self.nv, self.hv), dtype=torch.float32, device=x.device)
        for i in range(t):
            state = state * decay[:, i].exp()[..., None, None]
            kv_mem = (state * k[:, i][..., None]).sum(-2)
            delta = (v[:, i] - kv_mem) * beta[:, i][..., None]
            state = state + k[:, i][..., None] * delta[..., None, :]
            out[:, i] = (state * q[:, i][..., None]).sum(-2)
        # RMSNormGated は value head ごと（head_v_dim 単位）に正規化する
        v_ = out.pow(2).mean(-1, keepdim=True)
        out = rms_norm(out, self.norm_w, self.eps, plus_one=False).float()
        out = out * F.silu(z.reshape(b, t, self.nv, self.hv).float())
        out = out.reshape(b, t, self.vdim).to(x.dtype)
        return out @ self.out.T


class Qwen38Model:
    """Qwen3.8（text）の本体。"""

    def __init__(self, cfg, sd: dict, dtype=torch.float16, device="cuda"):
        self.cfg, self.device = cfg, device
        self.embed = sd["model.embed_tokens.weight"].to(device=device, dtype=dtype)
        self.norm_w = sd["model.norm.weight"].to(device=device, dtype=dtype)
        self.head = sd["lm_head.weight"].to(device=device, dtype=dtype)
        self.layers = []
        for i, lt in enumerate(cfg.layer_types):
            p = f"model.layers.{i}"
            mixer = (GatedDeltaNet(cfg, p + ".linear_attn", sd, dtype, device) if lt == "linear_attention"
                     else FullAttention(cfg, p + ".self_attn", sd, dtype, device))
            self.layers.append((lt, mixer, Mlp(p + ".mlp", sd, dtype, device),
                                sd[p + ".input_layernorm.weight"].to(device=device, dtype=dtype),
                                sd[p + ".post_attention_layernorm.weight"].to(device=device, dtype=dtype)))

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.view(-1)
        t = ids.numel()
        x = self.embed[ids].unsqueeze(0)
        pos = torch.arange(t, device=x.device)
        cos, sin = rope_tables(pos, int(self.cfg.head_dim * self.cfg.partial_rotary_factor),
                              self.cfg.rope_theta, x.dtype)
        cos, sin = cos[None], sin[None]
        for lt, mixer, mlp, ln1, ln2 in self.layers:
            h = rms_norm(x, ln1, self.cfg.rms_norm_eps)
            x = x + (mixer(h, cos, sin) if lt == "full_attention" else mixer(h))
            x = x + mlp(rms_norm(x, ln2, self.cfg.rms_norm_eps))
        x = rms_norm(x, self.norm_w, self.cfg.rms_norm_eps)
        return (x @ self.head.T)[:, -1]


class Qwen38Config:
    """config.json の text_config をそのまま持つ薄い箱。"""

    def __init__(self, text_config: dict):
        c = dict(text_config)
        rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        self.__dict__.update(c)
        self.rope_theta = float(rope.get("rope_theta", c.get("rope_theta", 10000.0)))
        self.partial_rotary_factor = float(c.get("partial_rotary_factor", 1.0))
        self.output_gate_type = c.get("output_gate_type", "sigmoid")
        self.rms_norm_eps = float(c.get("rms_norm_eps", 1e-6))
        interval = int(c.get("full_attention_interval", 4))
        self.layer_types = c.get("layer_types") or [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(int(c["num_hidden_layers"]))]

    @classmethod
    def load(cls, path: str) -> "Qwen38Config":
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg.get("text_config", cfg))
