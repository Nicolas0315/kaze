"""プロンプト 512 トークン + 1 トークン出力の所要時間（TTFT 相当）を 2 エンジンで測る。

    KAZE_GGUF=/path/to.gguf python3 bench/compare_ttft.py <gguf> <kaze モデル名>

注意: llama.cpp は**プロンプトの前方一致をキャッシュ**するので、毎回プロンプトの先頭を
変えないと 2 回目以降が実質 8 トークンだけの処理になり、比較になりません。
このスクリプトは先頭 8 文字を差し替えてキャッシュを避けています。
"""
import json, os, subprocess, sys, time
sys.path.insert(0, "/home/sahen/File/Prog/kaze")
os.environ.setdefault("KAZE_MODELS", "/home/sahen/File/Prog/models")
PROMPT_LEN = 512

def kaze_ttft(model_name):
    import torch
    from kaze.llm import LLM
    m = LLM(os.path.join(os.environ["KAZE_MODELS"], model_name), kind="int4",
            lm_head_kind="int4", speculative=False)
    base = ("The quick brown fox jumps over the lazy dog. " * 200)[:PROMPT_LEN * 4]
    prompt = base
    m.generate(prompt, max_tokens=2, temperature=0.0)          # 暖機
    torch.cuda.synchronize()
    best = 1e9
    for i in range(3):
        # 前置きを変えてプレフィックス再利用を避ける
        prompt = ("%08d" % i) + base[8:]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = m.generate(prompt, max_tokens=1, temperature=0.0)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    del m
    import gc; gc.collect(); torch.cuda.empty_cache()
    return out.prompt_tokens if hasattr(out, "prompt_tokens") else PROMPT_LEN, best

def llamacpp_ttft(gguf):
    from llama_cpp import Llama
    m = Llama(model_path=gguf, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    prompt = ("The quick brown fox jumps over the lazy dog. " * 200)[:PROMPT_LEN * 4]
    base = prompt
    m(prompt, max_tokens=1)
    best = 1e9
    for i in range(3):
        prompt = ("%08d" % i) + base[8:]        # 先頭を変えて前方一致キャッシュを避ける
        t0 = time.perf_counter(); m(prompt, max_tokens=1, temperature=0); best = min(best, time.perf_counter() - t0)
    del m
    import gc; gc.collect()
    return best

gguf = sys.argv[1]
kaze_model = sys.argv[2]
ll = llamacpp_ttft(gguf)
n, kk = kaze_ttft(kaze_model)
print("prompt %d tokens  1 token output" % PROMPT_LEN)
print("llama.cpp  %.3f s" % ll)
print("kaze       %.3f s  (prefill %.0f tok/s)" % (kk, n / kk))
