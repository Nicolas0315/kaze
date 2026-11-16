"""Model configuration (HuggingFace config.json compatible)."""
from __future__ import annotations

import json
import math
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
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_attention_heads",
                     "num_key_value_heads", "max_position_embeddings"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_hidden_layers) is not int or self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be a nonnegative integer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads:
                raise ValueError("hidden_size must be divisible by num_attention_heads without head_dim")
            self.head_dim = self.hidden_size // self.num_attention_heads
        if (type(self.head_dim) is not int or self.head_dim < 2
                or self.head_dim & (self.head_dim - 1)):
            raise ValueError("head_dim must be a power of two (at least 2) for Triton/RoPE")
        for name in ("rope_theta", "rms_norm_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act: {self.hidden_act!r}; expected silu")
        if self.architectures is None:
            self.architectures = []
        supported = {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
                     "MistralForCausalLM", "MixtralForCausalLM", "Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"}
        if not isinstance(self.architectures, list) or any(a not in supported for a in self.architectures):
            raise ValueError(f"unsupported architectures: {self.architectures!r}")
        for name in ("num_experts", "num_experts_per_tok"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if ((self.num_experts == 0) != (self.num_experts_per_tok == 0)
                or self.num_experts_per_tok > self.num_experts):
            raise ValueError("num_experts_per_tok must be within the configured expert count")
        if self.mlp_bias:
            raise ValueError("mlp_bias is not supported by the fused SwiGLU path")
        if self.shared_expert_intermediate_size:
            raise ValueError("shared_expert_intermediate_size is not supported")
        if self.rope_scaling is not None:
            if not isinstance(self.rope_scaling, dict):
                raise ValueError("rope_scaling must be a dictionary")
            if any(isinstance(v, dict) for v in self.rope_scaling.values()):
                raise ValueError("per-layer rope settings are not supported")
            if self.rope_scaling.get("partial_rotary_factor", 1.0) != 1.0:
                raise ValueError("partial rotary embeddings are not supported")
            rtype = self.rope_scaling.get("rope_type") or self.rope_scaling.get("type", "default")
            if rtype not in ("default", "linear", "yarn", "llama3"):
                raise ValueError(f"unsupported rope type: {rtype!r}")
            factor = self.rope_scaling.get("factor", 1.0)
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
                raise ValueError("rope scaling factor must be finite and positive")

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
        kwargs["architectures"] = raw.get("architectures") or []
        if "num_experts" not in raw and "num_local_experts" in raw:
            kwargs["num_experts"] = raw["num_local_experts"]
        if "norm_topk_prob" not in raw and (raw.get("model_type") == "mixtral"
                                               or "MixtralForCausalLM" in kwargs["architectures"]):
            kwargs["norm_topk_prob"] = True
        if raw.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("partial rotary embeddings are not supported")
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
        if raw.get("rope_parameters"):
            kwargs["rope_scaling"] = rope
        if isinstance(rope, dict) and rope:
            if kwargs.get("rope_theta") in (None, 10000.0) and "rope_theta" in rope:
                kwargs["rope_theta"] = float(rope["rope_theta"])
            if kwargs.get("rope_scaling") is None and rope.get("rope_type", "default") != "default":
                kwargs["rope_scaling"] = rope
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
