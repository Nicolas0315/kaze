"""Model configuration (HuggingFace config.json compatible)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False
    head_dim: int | None = None
    hidden_act: str = "silu"
    rope_scaling: dict[str, Any] | None = None
    architectures: list[str] = field(default_factory=list)
    torch_dtype: str = "float16"
    # Kaze extras
    qkv_bias: bool = True
    o_bias: bool = False
    mlp_bias: bool = False
    sliding_window: int | None = None
    qk_norm: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    norm_topk_prob: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelConfig":
        with open(path.rstrip("/") + "/config.json", encoding="utf-8") as f:
            raw = json.load(f)
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs["architectures"] = raw.get("architectures", [])
        if raw.get("head_dim"):
            kwargs["head_dim"] = raw["head_dim"]
        if "attention_bias" in raw:
            kwargs["qkv_bias"] = bool(raw["attention_bias"])
        if "mlp_bias" in raw:
            kwargs["mlp_bias"] = bool(raw["mlp_bias"])
        # transformers 5.x moved the rope settings into "rope_parameters" and no longer
        # writes a top level rope_theta.  Without this a Qwen3-14B/32B config silently
        # falls back to theta=10000 and the logits turn into noise.
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        if isinstance(rope, dict) and rope:
            if kwargs.get("rope_theta") in (None, 10000.0) and rope.get("rope_theta"):
                kwargs["rope_theta"] = float(rope["rope_theta"])
            if kwargs.get("rope_scaling") is None and rope.get("rope_type", "default") != "default":
                kwargs["rope_scaling"] = rope
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
