"""同じ GPU・同じ条件で、llama.cpp（llama-cpp-python）と Kaze を並べて測る。

llama-cpp-python が無い環境では Kaze 側だけ測ります。GGUF は環境変数で指定します。

    KAZE_GGUF=/path/to/model.gguf python3 bench/compare_llamacpp.py
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODELS = os.environ.get("KAZE_MODELS", os.path.join(os.path.dirname(ROOT), "models"))
GGUF = os.environ.get("KAZE_GGUF", "")
PROMPT_TOKENS = 48
GEN_TOKENS = 128


def prompt_text(n: int) -> str:
    """だいたい n トークンになるプロンプト。"""
    base = "The quick brown fox jumps over the lazy dog. "
    return (base * 20)[: n * 4]


def bench_kaze(model_name: str = "Qwen2.5-3B-Instruct", kind: str = "int4") -> dict:
    import torch

    from kaze.llm import LLM

    llm = LLM(os.path.join(MODELS, model_name), kind=kind, lm_head_kind=kind,
              speculative=False)
    prompt = prompt_text(PROMPT_TOKENS)
    llm.generate(prompt, max_tokens=8, temperature=0.0)      # 暖機
    torch.cuda.synchronize()
    # llama.cpp 側と条件を揃えるため、プロンプト評価の分を差し引く
    times = {}
    for n in (16, GEN_TOKENS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompt, max_tokens=n, temperature=0.0)
        torch.cuda.synchronize()
        times[n] = time.perf_counter() - t0
    decode = (GEN_TOKENS - 16) / max(times[GEN_TOKENS] - times[16], 1e-9)
    del llm
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return {"engine": "kaze", "decode_tok_s": decode}


def bench_llamacpp() -> dict:
    try:
        from llama_cpp import Llama
    except Exception as exc:
        return {"engine": "llama.cpp", "error": str(exc)[:120]}
    if not GGUF or not os.path.isfile(GGUF):
        return {"engine": "llama.cpp", "error": "KAZE_GGUF を指定してください"}
    llm = Llama(model_path=GGUF, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    prompt = prompt_text(PROMPT_TOKENS)
    llm(prompt, max_tokens=8)
    # timings はバージョンによって入っていないので、長さを変えた 2 回の差から decode 速度を出す
    times = {}
    for n in (16, GEN_TOKENS):
        t0 = time.perf_counter()
        llm(prompt, max_tokens=n, temperature=0)
        times[n] = time.perf_counter() - t0
    decode = (GEN_TOKENS - 16) / max(times[GEN_TOKENS] - times[16], 1e-9)
    del llm
    import gc

    gc.collect()
    return {"engine": "llama.cpp", "decode_tok_s": decode}


def main() -> None:
    # 引数で GGUF と Kaze のモデルを差し替えられる（サイズごとに A/B する）
    global GGUF
    if len(sys.argv) > 2:
        GGUF = sys.argv[1]
        model = sys.argv[2]
    else:
        model = "Qwen2.5-3B-Instruct"
    # 8GB のカードでは 2 つ同時に載らないので、先に llama.cpp を測って解放する
    for row in (bench_llamacpp(), bench_kaze(model)):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
