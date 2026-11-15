"""Qwen3.8（Qwen3_5 text）の実装。

full attention と gated delta net（線形注意）が交互に並ぶ hybrid 構成で、attention 側には
出力ゲートと部分 RoPE が付きます。KV キャッシュ（full attention）と再帰状態・conv 履歴
（線形注意）を持つので、decode は 1 トークンあたり一定のコストで進みます。
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


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, plus_one: bool = True) -> torch.Tensor:
    """Qwen3_5 の RMSNorm は重みを (1 + w) として掛ける（初期値はゼロ）。"""
    v = x.float().pow(2).mean(-1, keepdim=True)
    y = x.float() * torch.rsqrt(v + eps)
    y = y * (1.0 + w.float()) if plus_one else y * w.float()
    return y.to(x.dtype)


def recurrent_delta_rule(q, k, v, g, beta, initial_state=None, return_state: bool = False):
    b, t, h, dk = k.shape
    dv = v.shape[-1]
    q = F.normalize(q.float(), dim=-1, eps=1e-6) / (dk ** 0.5)
    k = F.normalize(k.float(), dim=-1, eps=1e-6)
    v = v.float()
    if initial_state is None:
        state = torch.zeros((b, h, dk, dv), dtype=torch.float32, device=q.device)
    else:
        state = initial_state.float()
    out = torch.zeros((b, t, h, dv), dtype=torch.float32, device=q.device)
    for i in range(t):
        state = state * g[:, i].exp()[..., None, None]
        kv_mem = (state * k[:, i][..., None]).sum(-2)
        delta = (v[:, i] - kv_mem) * beta[:, i][..., None]
        state = state + k[:, i][..., None] * delta[..., None, :]
        out[:, i] = (state * q[:, i][..., None]).sum(-2)
    return (out, state) if return_state else out


def _reference_chunk_rule():
    # transformers に入っている chunked 実装（無ければ None）。
    # 自前の chunked 版は数値が合わなかったので、使える環境では検証済みの実装に任せる。
    for mod in ("transformers.models.qwen3_next.modeling_qwen3_next",
                "transformers.models.qwen3_5.modeling_qwen3_5"):
        try:
            m = __import__(mod, fromlist=["torch_chunk_gated_delta_rule"])
            fn = getattr(m, "torch_chunk_gated_delta_rule", None)
            if fn is not None:
                return fn
        except Exception:
            continue
    return None


_CHUNK_RULE = _reference_chunk_rule()


def delta_rule(q, k, v, g, beta, chunk_size: int = 64, initial_state=None,
               return_state: bool = False):
    if _CHUNK_RULE is not None:
        out, st = _CHUNK_RULE(q, k, v, g, beta, chunk_size=chunk_size,
                              initial_state=initial_state,
                              output_final_state=return_state,
                              use_qk_l2norm_in_kernel=True)
        return (out, st) if return_state else out
    return recurrent_delta_rule(q, k, v, g, beta, initial_state, return_state)


class Qwen38Cache:
    """full attention は K/V、線形注意は再帰状態と conv 履歴を持つ。"""

    def __init__(self, cfg, batch: int = 1, device=None, dtype=torch.float16):
        n = int(cfg.num_hidden_layers)
        self.k: list[torch.Tensor | None] = [None] * n
        self.v: list[torch.Tensor | None] = [None] * n
        self.state: list[torch.Tensor | None] = [None] * n
        self.conv: list[torch.Tensor | None] = [None] * n
        self.len = 0
        self.device = device
        self.dtype = dtype


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
        self.eps = cfg.rms_norm_eps
        self.scale = self.hd ** -0.5

    def __call__(self, x, cos, sin, cache=None, layer: int | None = None):
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
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        past = 0
        if cache is not None and layer is not None:
            if cache.k[layer] is not None:
                past = cache.k[layer].shape[2]
                k = torch.cat([cache.k[layer], k], dim=2)
                v = torch.cat([cache.v[layer], v], dim=2)
            cache.k[layer], cache.v[layer] = k, v
        if past == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale,
                                                 enable_gqa=self.nh != self.nkv)
        else:
            total = past + t
            qi = torch.arange(t, device=q.device)[:, None] + past
            ki = torch.arange(total, device=q.device)[None, :]
            mask = ki <= qi
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale,
                                                 enable_gqa=self.nh != self.nkv)
        out = out.transpose(1, 2).reshape(b, t, self.nh * self.hd)
        gate = torch.sigmoid(gate.reshape(b, t, -1))
        return (out * gate.float()).to(x.dtype) @ self.o.T


class GatedDeltaNet:
    """線形注意層（gated delta rule）。再帰状態と conv 履歴をキャッシュできる。"""

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
        self.chunk_size = 64

    def __call__(self, x: torch.Tensor, cache=None, layer: int | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = torch.split(x @ self.qkv.T, [self.kdim, self.kdim, self.vdim], dim=-1)
        z = x @ self.z.T
        beta = torch.sigmoid(x @ self.b_w.T)
        a = x @ self.a_w.T
        # depthwise 因果 conv（活性 silu）。キャッシュがあれば直前の入力を前置きする
        seq = torch.cat([q, k, v], -1).transpose(1, 2)
        w = self.conv_w.squeeze(1)
        ksz = w.shape[-1]
        if cache is not None and layer is not None and cache.conv[layer] is not None:
            seq = torch.cat([cache.conv[layer], seq], dim=2)
        if cache is not None and layer is not None:
            cache.conv[layer] = seq[:, :, -(ksz - 1):].detach() if ksz > 1 else None
        pad = F.pad(seq, (ksz - 1, 0))
        # 履歴を前置きした場合は、末尾 t 個の出力に対応する窓を使う
        off = seq.shape[-1] - t
        conv = sum(w[:, i][None, :, None] * pad[:, :, off + i:off + i + t] for i in range(ksz))
        q, k, v = torch.split(F.silu(conv).transpose(1, 2), [self.kdim, self.kdim, self.vdim], -1)
        q = q.view(b, t, self.nk, self.hk)
        k = k.view(b, t, self.nk, self.hk)
        v = v.view(b, t, self.nv, self.hv)
        if self.nv > self.nk:
            rep = self.nv // self.nk
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        decay = -self.A_log.exp()[None, None, :] * F.softplus(a.float() + self.dt_bias)
        q = q.float()
        k = k.float()
        v = v.float()
        state = cache.state[layer] if (cache is not None and layer is not None) else None
        out, new_state = delta_rule(q, k, v, decay, beta, self.chunk_size,
                                    initial_state=state, return_state=True)
        if cache is not None and layer is not None:
            cache.state[layer] = new_state.detach()
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

    def new_cache(self, batch: int = 1) -> Qwen38Cache:
        return Qwen38Cache(self.cfg, batch, self.device)

    def __call__(self, ids: torch.Tensor, cache: "Qwen38Cache | None" = None,
                 start_pos: int = 0) -> torch.Tensor:
        ids = ids.view(-1)
        t = ids.numel()
        x = self.embed[ids].unsqueeze(0)
        rd = int(self.cfg.head_dim * self.cfg.partial_rotary_factor)
        positions = torch.arange(start_pos, start_pos + t, device=x.device)
        cos, sin = rope_tables(positions, rd, self.cfg.rope_theta, x.dtype)
        cos, sin = cos[None], sin[None]
        for i, (lt, mixer, mlp, ln1, ln2) in enumerate(self.layers):
            h = rms_norm(x, ln1, self.cfg.rms_norm_eps)
            mix = (mixer(h, cos, sin, cache, i) if lt == "full_attention"
                   else mixer(h, cache, i))
            x = x + mix
            x = x + mlp(rms_norm(x, ln2, self.cfg.rms_norm_eps))
        if cache is not None:
            cache.len = start_pos + t
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


def load_qwen38(path: str, dtype=None, device: str | None = None):
    # safetensors を全部読んで text モデルを組み立てる
    import glob
    import os

    import safetensors.torch as st

    path = str(path)
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    sd = {}
    for f in files:
        sd.update(st.load_file(f))
    cfg = Qwen38Config.load(os.path.join(path, "config.json"))
    return Qwen38Model(cfg, sd, dtype=dtype or torch.float16,
                       device=device or ("cuda" if torch.cuda.is_available() else "cpu"))


def generate(model: Qwen38Model, ids, max_tokens: int = 32, temperature: float = 0.0,
             eos_token_ids=(), use_cache: bool = True) -> list[int]:
    # キャッシュがあれば 1 トークンずつ進む（無ければ毎回作り直す）
    dev = model.device
    seq = list(ids)
    out: list[int] = []
    stop = set(eos_token_ids)
    cache = model.new_cache() if use_cache else None
    if cache is not None:
        logits = model(torch.tensor(seq, dtype=torch.long, device=dev), cache, 0)[0]
    else:
        logits = model(torch.tensor(seq, dtype=torch.long, device=dev))[0]
    for _ in range(max_tokens):
        if temperature and temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1).item())
        else:
            nxt = int(logits.argmax().item())
        out.append(nxt)
        seq.append(nxt)
        if nxt in stop:
            break
        if cache is not None:
            logits = model(torch.tensor([nxt], dtype=torch.long, device=dev), cache, cache.len)[0]
        else:
            logits = model(torch.tensor(seq, dtype=torch.long, device=dev))[0]
    return out
