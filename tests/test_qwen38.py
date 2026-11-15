"""Qwen3.8（Qwen3_5 text）の実装が transformers の参照と一致するかを見るテスト。

重みは乱数で作った小さなモデル（層 4、hidden 64）を使うので、GPU もダウンロードも不要です。
transformers に qwen3_5 が入っていない環境では SKIP します。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from kaze.qwen38 import Qwen38Config, Qwen38Model

BASE = {
    "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 32, "max_position_embeddings": 512, "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000.0}, "partial_rotary_factor": 0.5,
    "attn_output_gate": True, "output_gate_type": "swish", "attention_bias": False,
    "attention_dropout": 0.0, "hidden_act": "silu", "full_attention_interval": 2,
    "tie_word_embeddings": False, "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4,
    "bos_token_id": 1, "eos_token_id": 2,
}

LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]




def check_load_and_generate() -> bool:
    """safetensors に保存 → load_qwen38 → generate が通るか（CLI の経路）。"""
    import json
    import tempfile

    import safetensors.torch as st
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    from kaze.qwen38 import generate, load_qwen38

    raw = dict(BASE, layer_types=LAYER_TYPES, tie_word_embeddings=False)
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_config(Qwen3_5TextConfig(**raw),
                                          dtype=torch.float32).eval()
    with tempfile.TemporaryDirectory() as d:
        st.save_file({k: v.contiguous() for k, v in ref.state_dict().items()},
                     os.path.join(d, "model.safetensors"))
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(dict(raw, model_type="qwen3_5_text"), f)
        model = load_qwen38(d, dtype=torch.float32, device="cpu")
    ids = [5, 12, 33]
    want = ref(torch.tensor(ids)[None, :]).logits[0, -1]
    got = model(torch.tensor(ids)).flatten()
    cos = torch.nn.functional.cosine_similarity(want, got, dim=0).item()
    out = generate(model, ids, max_tokens=4)
    print("load_qwen38: cos %.6f  生成 %d トークン %s" % (cos, len(out), out[:4]))
    return cos > 0.999 and len(out) == 4




def check_cache() -> bool:
    """キャッシュ経路が再計算経路と同じ logits・同じ生成結果になるか。"""
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    from kaze.qwen38 import generate

    raw = dict(BASE, layer_types=LAYER_TYPES)
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_config(Qwen3_5TextConfig(**raw),
                                          dtype=torch.float32).eval()
    sd = {k: v.detach().float() for k, v in ref.state_dict().items()}
    model = Qwen38Model(Qwen38Config(raw), sd, dtype=torch.float32, device="cpu")
    ids = [5, 12, 33, 7, 91]
    cache = model.new_cache()
    a = model(torch.tensor(ids), cache, 0)[0]
    b = model(torch.tensor(ids))[0]
    tok = int(a.argmax())
    c = model(torch.tensor([tok]), cache, cache.len)[0]
    d = model(torch.tensor(ids + [tok]))[0]
    diff = (c - d).abs().max().item()
    same_tokens = generate(model, ids, max_tokens=8, use_cache=True) == \
        generate(model, ids, max_tokens=8, use_cache=False)
    print("cache: 1ステップ差 %.7f  生成一致 %s" % (diff, same_tokens))
    return diff < 1e-4 and same_tokens




def check_quantised() -> bool:
    """int8/int4 に量子化しても参照と近い logits が出るか（GPU）。"""
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    if not torch.cuda.is_available():
        print("quantised: skip (cuda not available)")
        return True
    raw = dict(BASE, layer_types=LAYER_TYPES)
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_config(Qwen3_5TextConfig(**raw),
                                          dtype=torch.float32).eval()
    sd = {k: v.detach().float() for k, v in ref.state_dict().items()}
    ids = torch.tensor([5, 12, 33, 7, 91, 2, 44])
    with torch.no_grad():
        want = ref(ids[None, :]).logits[0, -1].cuda()
    ok = True
    for kind, thr in (("int8", 0.99), ("int4", 0.9)):
        model = Qwen38Model(Qwen38Config(raw), sd, dtype=torch.float16,
                            device="cuda", kind=kind)
        got = model(ids.cuda()).flatten()
        cos = torch.nn.functional.cosine_similarity(want, got, dim=0).item()
        mem = torch.cuda.memory_allocated() / 1e9
        print("quantised %-4s cos %.6f  %.2f GB" % (kind, cos, mem))
        ok = ok and cos > thr
        del model
        torch.cuda.empty_cache()
    return ok


def main() -> int:
    try:
        from transformers import AutoModelForCausalLM, Qwen3_5TextConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except Exception as exc:
        print("skip: transformers not usable", exc)
        return 0
    if "qwen3_5_text" not in CONFIG_MAPPING_NAMES:
        print("skip: this transformers has no qwen3_5_text")
        return 0

    raw = dict(BASE, layer_types=LAYER_TYPES)
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_config(Qwen3_5TextConfig(**raw),
                                          dtype=torch.float32).eval()
    sd = {k: v.detach().float() for k, v in ref.state_dict().items()}
    ids = torch.tensor([5, 12, 33, 7, 91, 2, 44], dtype=torch.long)
    with torch.no_grad():
        want = ref(ids[None, :]).logits[0, -1]

    ours = Qwen38Model(Qwen38Config(raw), sd, dtype=torch.float32, device="cpu")
    got = ours(ids).flatten()
    cos = torch.nn.functional.cosine_similarity(want, got, dim=0).item()
    diff = (want - got).abs().max().item()
    print("hybrid 4層（線形2 + full2）: cos %.6f  maxdiff %.6f" % (cos, diff))
    ok = (cos > 0.999 and check_load_and_generate() and check_cache()
          and check_quantised())
    print("ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())