"""チェックポイントのキーが実装と噛み合っているかを、重みを落とさずに確かめる。

`model.safetensors.index.json`（数十 KB）だけを取ってきて、実装が要求するキーと突き合わせます。
55GB を落とす前に、KeyError 系の失敗を潰せます。

    python3 bench/check_weights.py Qwen/Qwen3.8-27B
"""
from __future__ import annotations

import json
import sys


def normalise(key: str) -> str | None:
    """配布物の VLM 形式を text 側の名前に直す。視覚側と MTP は捨てる。"""
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key.startswith(("model.visual.", "visual.", "mtp.")):
        return None
    return key


def shape_of(key: str) -> str:
    """レイヤ番号を落として「型」に落とす。"""
    parts = key.split(".")
    if len(parts) > 2 and parts[1] == "layers":
        return ".".join(parts[:2] + ["N"] + parts[3:])
    return key


def check(repo: str) -> int:
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    path = hf_hub_download(repo, "model.safetensors.index.json")
    index = json.load(open(path))
    real = {}
    for key in index["weight_map"]:
        name = normalise(key)
        if name is not None:
            real[shape_of(name)] = 1

    # 実装側が要求する型は、同じ構成の極小モデルから取る（重みは乱数でよい）
    cfg = Qwen3_5TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=512, rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0}, partial_rotary_factor=0.5,
        attn_output_gate=True, output_gate_type="swish", attention_bias=False,
        attention_dropout=0.0, hidden_act="silu", full_attention_interval=2,
        tie_word_embeddings=False, linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"] * 2)
    ours = {shape_of(k) for k in AutoModelForCausalLM.from_config(cfg).state_dict()}

    missing = sorted(ours - set(real))
    print(f"{repo}: テキスト側のテンソル型 {len(real)} / 実装が要求する型 {len(ours)}")
    print("足りない型:", missing if missing else "なし")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(check(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B"))
