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
    # decode（1〜数トークン）は chunk に詰めると無駄が多いので逐次で回す
    if q.shape[1] <= 8:
        return recurrent_delta_rule(q, k, v, g, beta, initial_state, return_state)
    if _CHUNK_RULE is not None:
        out, st = _CHUNK_RULE(q, k, v, g, beta, chunk_size=chunk_size,
                              initial_state=initial_state,
                              output_final_state=return_state,
                              use_qk_l2norm_in_kernel=True)
        return (out, st) if return_state else out
    return recurrent_delta_rule(q, k, v, g, beta, initial_state, return_state)


class StaticCache:
    """CUDA Graph 用の固定長キャッシュ。位置はテンソルで持つので、グラフ内で進められる。"""

    def __init__(self, cfg, max_len: int, batch: int = 1, device="cuda",
                 dtype=torch.float16):
        n = int(cfg.num_hidden_layers)
        self.max_len = max_len
        self.k = [None] * n
        self.v = [None] * n
        self.state = [None] * n
        self.conv = [None] * n
        conv_dim = (2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
                    + cfg.linear_num_value_heads * cfg.linear_value_head_dim)
        ksz = int(cfg.linear_conv_kernel_dim)
        for i, lt in enumerate(cfg.layer_types):
            if lt == "full_attention":
                self.k[i] = torch.zeros(batch, cfg.num_key_value_heads, max_len,
                                        cfg.head_dim, dtype=dtype, device=device)
                self.v[i] = torch.zeros_like(self.k[i])
            else:
                self.state[i] = torch.zeros(batch, cfg.linear_num_value_heads,
                                            cfg.linear_key_head_dim,
                                            cfg.linear_value_head_dim,
                                            dtype=torch.float32, device=device)
                self.conv[i] = torch.zeros(batch, conv_dim, max(ksz - 1, 1),
                                           dtype=dtype, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)


class QLinear:
    """重みを int8/int4 に量子化して、エンジン本体の Triton カーネルで積む層。"""

    def __init__(self, w: torch.Tensor, kind: str, group: int, device, dtype):
        from kaze.quant import QuantLinear

        self.kind = kind
        self.lin = QuantLinear.from_float(w.to(device), None, kind=kind, group=group)
        self.n, self.k = w.shape
        self._pad = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import kaze.kernels.quant_linear as K

        shape = x.shape[:-1]
        rows = 1
        for s in shape:
            rows *= s
        flat = x.reshape(rows, self.k).contiguous()
        if rows == 1:
            # gemv は tdim 行を前提に読むので、1 行でも 16 行に詰めてから渡す。
            # バッファは使い回す（毎トークン確保すると呼び出し回数ぶん無駄が出る）
            if self._pad is None or self._pad.device != flat.device:
                self._pad = torch.zeros((16, self.k), dtype=flat.dtype, device=flat.device)
            self._pad[0].copy_(flat[0])
            out = K.gemv(self._pad, self.lin, 16)[:1]
        else:
            out = K.gemm(flat, self.lin)
        return out.reshape(*shape, self.n)


def linear(w: torch.Tensor, kind: str, group: int, device, dtype):
    """kind が int8/int4 なら量子化し、そうでなければ普通の線形層として返す。"""
    if kind in ("int8", "int4", "mixed"):
        return QLinear(w, "int8" if kind == "mixed" else kind, group, device, dtype)
    return Dense(w, dtype, device)


class Dense:
    """量子化しない普通の線形層（小さい行列や fp16 用）。"""

    def __init__(self, w: torch.Tensor, dtype, device):
        self.w = w.to(device=device, dtype=dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T


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
    def __init__(self, prefix: str, sd: dict, dtype, device, kind="fp16", group=128):
        get = lambda n: linear(sd[f"{prefix}.{n}"], kind, group, device, dtype)
        self.gate, self.up, self.down = (get("gate_proj.weight"), get("up_proj.weight"),
                                         get("down_proj.weight"))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class FullAttention:
    """full attention 層。q_proj は出力ゲートと半分ずつに分かれる。"""

    def __init__(self, cfg, prefix: str, sd: dict, dtype, device, kind="fp16", group=128):
        q = lambda n: linear(sd[f"{prefix}.{n}"], kind, group, device, dtype)
        self.q, self.k, self.v, self.o = (q("q_proj.weight"), q("k_proj.weight"),
                                          q("v_proj.weight"), q("o_proj.weight"))
        g = lambda n: sd[f"{prefix}.{n}"].to(device=device, dtype=dtype)
        self.qn, self.kn = g("q_norm.weight"), g("k_norm.weight")
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.rope_dim = int(self.hd * cfg.partial_rotary_factor)
        self.eps = cfg.rms_norm_eps
        self.scale = self.hd ** -0.5

    def __call__(self, x, cos, sin, cache=None, layer: int | None = None):
        b, t, _ = x.shape
        qg = self.q(x).view(b, t, self.nh, self.hd * 2)
        q, gate = qg.chunk(2, dim=-1)
        q = rms_norm(q, self.qn, self.eps)
        k = rms_norm(self.k(x).view(b, t, self.nkv, self.hd), self.kn, self.eps)
        v = self.v(x).view(b, t, self.nkv, self.hd)
        c, s, rd = cos[:, :, None, :], sin[:, :, None, :], self.rope_dim
        if rd < self.hd:
            q = torch.cat([q[..., :rd] * c + rotate_half(q[..., :rd]) * s, q[..., rd:]], -1)
            k = torch.cat([k[..., :rd] * c + rotate_half(k[..., :rd]) * s, k[..., rd:]], -1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if cache is not None and getattr(cache, "max_len", None) and layer is not None:
            # 固定長バッファに書き、全長に対してマスク付きで注意する（グラフ化できる形）
            ks, vs = cache.k[layer], cache.v[layer]
            step = torch.arange(t, device=q.device)
            ks.index_copy_(2, cache.pos + step, k)
            vs.index_copy_(2, cache.pos + step, v)
            qi = (cache.pos + step)[:, None]
            ki = torch.arange(cache.max_len, device=q.device)[None, :]
            mask = ki <= qi
            out = F.scaled_dot_product_attention(q, ks, vs, attn_mask=mask,
                                                 scale=self.scale,
                                                 enable_gqa=self.nh != self.nkv)
            out = out.transpose(1, 2).reshape(b, t, self.nh * self.hd)
            gate = torch.sigmoid(gate.reshape(b, t, -1))
            return self.o((out * gate.float()).to(x.dtype))
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
        return self.o((out * gate.float()).to(x.dtype))


class GatedDeltaNet:
    """線形注意層（gated delta rule）。再帰状態と conv 履歴をキャッシュできる。"""

    def __init__(self, cfg, prefix: str, sd: dict, dtype, device, kind="fp16", group=128):
        g = lambda n: sd[f"{prefix}.{n}"].to(device=device, dtype=dtype)
        q = lambda n: linear(sd[f"{prefix}.{n}"], kind, group, device, dtype)
        self.qkv, self.z = q("in_proj_qkv.weight"), q("in_proj_z.weight")
        self.b_w, self.a_w = g("in_proj_b.weight"), g("in_proj_a.weight")
        self.conv_w = g("conv1d.weight")
        self.A_log = sd[f"{prefix}.A_log"].to(device=device, dtype=torch.float32)
        self.dt_bias = sd[f"{prefix}.dt_bias"].to(device=device, dtype=torch.float32)
        self.norm_w = g("norm.weight")
        self.out = q("out_proj.weight")
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.hk, self.hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.eps = cfg.rms_norm_eps
        self.kdim, self.vdim = self.nk * self.hk, self.nv * self.hv
        self.chunk_size = 64

    def __call__(self, x: torch.Tensor, cache=None, layer: int | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = torch.split(self.qkv(x), [self.kdim, self.kdim, self.vdim], dim=-1)
        z = self.z(x)
        beta = torch.sigmoid(x @ self.b_w.T)
        a = x @ self.a_w.T
        # depthwise 因果 conv（活性 silu）。キャッシュがあれば直前の入力を前置きする
        seq = torch.cat([q, k, v], -1).transpose(1, 2)
        w = self.conv_w.squeeze(1)
        ksz = w.shape[-1]
        fixed = cache is not None and getattr(cache, "max_len", None) and layer is not None
        if fixed:
            hist = cache.conv[layer]
            seq = torch.cat([hist, seq], dim=2)
            hist.copy_(seq[:, :, -(ksz - 1):]) if ksz > 1 else None
        elif cache is not None and layer is not None and cache.conv[layer] is not None:
            seq = torch.cat([cache.conv[layer], seq], dim=2)
        if cache is not None and layer is not None and not fixed:
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
        state = None
        if cache is not None and layer is not None:
            state = cache.state[layer]
        out, new_state = delta_rule(q, k, v, decay, beta, self.chunk_size,
                                    initial_state=state, return_state=True)
        if state is not None and new_state is not None:
            state.copy_(new_state)          # 固定長バッファに書き戻す（グラフ化できる形）
        elif cache is not None and layer is not None:
            cache.state[layer] = new_state.detach()   # 動的キャッシュは差し替え
        v_ = out.pow(2).mean(-1, keepdim=True)
        out = rms_norm(out, self.norm_w, self.eps, plus_one=False).float()
        out = out * F.silu(z.reshape(b, t, self.nv, self.hv).float())
        out = out.reshape(b, t, self.vdim).to(x.dtype)
        return self.out(out)


class Qwen38Model:
    """Qwen3.8（text）の本体。"""

    def __init__(self, cfg, sd: dict, dtype=torch.float16, device="cuda",
                 kind: str = "fp16", group: int = 128):
        self.cfg, self.device = cfg, device
        self.embed = sd["model.embed_tokens.weight"].to(device=device, dtype=dtype)
        self.norm_w = sd["model.norm.weight"].to(device=device, dtype=dtype)
        self.head = linear(sd["lm_head.weight"], kind, group, device, dtype)
        self.layers = []
        for i, lt in enumerate(cfg.layer_types):
            p = f"model.layers.{i}"
            mixer = (GatedDeltaNet(cfg, p + ".linear_attn", sd, dtype, device, kind, group)
                     if lt == "linear_attention"
                     else FullAttention(cfg, p + ".self_attn", sd, dtype, device, kind, group))
            self.layers.append((lt, mixer, Mlp(p + ".mlp", sd, dtype, device, kind, group),
                                sd[p + ".input_layernorm.weight"].to(device=device, dtype=dtype),
                                sd[p + ".post_attention_layernorm.weight"].to(device=device, dtype=dtype)))

    def new_cache(self, batch: int = 1) -> Qwen38Cache:
        return Qwen38Cache(self.cfg, batch, self.device)

    def step(self, ids: torch.Tensor, cache) -> torch.Tensor:
        # 固定長キャッシュで 1 チャンク進める（CUDA Graph のキャプチャ対象）
        return self(ids, cache)

    def __call__(self, ids: torch.Tensor, cache: "Qwen38Cache | None" = None,
                 start_pos: int = 0) -> torch.Tensor:
        ids = ids.view(-1)
        t = ids.numel()
        x = self.embed[ids].unsqueeze(0)
        rd = int(self.cfg.head_dim * self.cfg.partial_rotary_factor)
        if cache is not None and getattr(cache, "max_len", None):
            positions = cache.pos + torch.arange(t, device=x.device)
        else:
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
            if getattr(cache, "max_len", None):
                cache.pos += t          # 位置の前進は 1 ステップ 1 回
            else:
                cache.len = start_pos + t
        x = rms_norm(x, self.norm_w, self.cfg.rms_norm_eps)
        return self.head(x)[:, -1]


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


def load_qwen38(path: str, dtype=None, device: str | None = None,
                 kind: str = "fp16", group: int = 128, verbose: bool = False):
    """チェックポイントを読んで text モデルを組み立てる。

    配布物は VLM 形式（`Qwen3_5ForConditionalGeneration`）なので、重みのキーは
    `model.language_model.*` になっています。text 側だけを取り出して `model.*` に直し、
    視覚側（`model.visual.*`）や MTP の重みは捨てます。
    """
    import glob
    import os

    import safetensors.torch as st

    path = str(path)
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    if dtype is None:
        dtype = torch.float16
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Qwen38Config.load(os.path.join(path, "config.json"))

    sd = {}
    dropped = 0
    for f in files:
        for k, v in st.load_file(f, device="cpu").items():
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model."):]
            elif k.startswith(("model.visual.", "visual.", "mtp.")):
                dropped += 1
                continue
            # 途中で fp16 に落としておくと、55GB のチェックポイントでも RAM が半分で済む
            sd[k] = v.to(dtype) if v.is_floating_point() else v
    if "model.embed_tokens.weight" not in sd:
        head = [k for k in sd if k.endswith("embed_tokens.weight")]
        raise KeyError(
            "text 側の重みが見つかりません（model.embed_tokens.weight が無い）。"
            f" 先頭のキー: {list(sd)[:5]}")
    if "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    if verbose:
        print(f"{len(files)} シャード / {len(sd)} テンソル（視覚側 {dropped} 個を除外）")
    model = Qwen38Model(cfg, sd, dtype=dtype, device=device, kind=kind, group=group)
    del sd
    return model

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


class GraphDecoder:
    """1 トークン decode を CUDA Graph に固めて投げる。"""

    def __init__(self, model: "Qwen38Model", max_len: int = 2048):
        self.model = model
        self.cache = StaticCache(model.cfg, max_len, device=model.device,
                                 dtype=torch.float16)
        self.ids = torch.zeros(1, dtype=torch.long, device=model.device)
        # 暖機（カーネルのコンパイルと作業領域の確保）
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.model(self.ids, self.cache)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self.model(self.ids, self.cache)
        self.reset()

    def reset(self):
        c = self.cache
        for buf in (c.k, c.v, c.conv, c.state):
            for item in buf:
                if item is not None:
                    item.zero_()
        c.pos.zero_()

    def prefill(self, ids) -> torch.Tensor:
        out = self.model(torch.tensor(list(ids), dtype=torch.long, device=self.ids.device),
                         self.cache)
        return out[0]

    def decode(self, token: int) -> torch.Tensor:
        self.ids[0] = token
        self.graph.replay()
        return self.logits[0]


def generate_graph(model: "Qwen38Model", ids, max_tokens: int = 64, max_len: int = 2048,
                   eos_token_ids=()) -> list[int]:
    """CUDA Graph 版の貪欲生成。"""
    dec = GraphDecoder(model, max_len=max_len)
    logits = dec.prefill(ids)
    out: list[int] = []
    stop = set(eos_token_ids)
    for _ in range(max_tokens):
        nxt = int(logits.argmax().item())
        out.append(nxt)
        if nxt in stop:
            break
        logits = dec.decode(nxt)
    return out


def stream_tokens(model: "Qwen38Model", ids, max_tokens: int = 32,
                  temperature: float = 0.0, eos_token_ids=(), graph: bool = False,
                  max_len: int | None = None, decoder=None):
    """トークンを 1 個ずつ返す。サーバのストリーミングと CLI で共用。"""
    dev = model.device
    stop = set(eos_token_ids)
    if decoder is not None:
        decoder.reset()
        dec = decoder
        logits = dec.prefill(ids)
    elif graph:
        dec = GraphDecoder(model, max_len=max_len or max(len(ids) + max_tokens + 8, 256))
        logits = dec.prefill(ids)
    else:
        dec = None
        cache = model.new_cache()
        logits = model(torch.tensor(list(ids), dtype=torch.long, device=dev), cache, 0)[0]
    for _ in range(max_tokens):
        if temperature and temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1).item())
        else:
            nxt = int(logits.argmax().item())
        yield nxt
        if nxt in stop:
            return
        if dec is not None:
            logits = dec.decode(nxt)
        else:
            logits = model(torch.tensor([nxt], dtype=torch.long, device=dev), cache, cache.len)[0]
