"""Single-engine, single-model measurement. Prints one JSON line.

Run as a subprocess so each engine gets a clean GPU (no allocator carry-over).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def measure_kaze(model_path: str, kind: str = "int8", group: int = 128,
                 lm_head_kind: str = "int8", prompt_len: int = 64, gen_len: int = 128,
                 speculative: bool = True) -> dict:
    import torch
    from kaze.llm import LLM
    from kaze.sampler import SamplingParams

    llm = LLM(model_path, kind=kind, group=group, lm_head_kind=lm_head_kind,
              speculative=speculative, start_loop=False, verbose=False)
    try:
        tok = llm.tokenizer
        prompt = "Write a short paragraph about GPUs. " * 64
        ids = llm._encode(prompt, chat=False)[:prompt_len]
        out = {"engine": "kaze", "model": model_path.rstrip("/").split("/")[-1],
               "kind": kind, "bytes": llm.model.weight_bytes()}

        # prefill: 512 fresh tokens each time (no prefix-cache hits), matches llama-bench pp512
        best_prefill = 1e9
        for _ in range(3):
            fresh = list(torch.randint(0, 30000, (512,)).tolist())     # no prefix-cache hits
            seq = llm.engine.add_request(fresh, SamplingParams(max_tokens=1, temperature=0.0))
            t0 = time.perf_counter()
            while seq.state != 3:
                llm.engine.step()
            torch.cuda.synchronize()
            best_prefill = min(best_prefill, time.perf_counter() - t0)
        out["prefill_tok_s"] = 512 / best_prefill
        out["ttft_ms"] = best_prefill * 1e3

        # decode: batch 1, 64-token prompt
        prompt_ids = ids[:64]
        base_len = len(prompt_ids)
        best = None
        for _ in range(3):
            seq = llm.engine.add_request(prompt_ids, SamplingParams(max_tokens=gen_len, temperature=0.0))
            while seq.state != 3:
                llm.engine.step()
            torch.cuda.synchronize()
            if not seq.itls:
                continue
            itl = sum(seq.itls) / len(seq.itls)
            if best is None or itl < best:
                best = itl
                out["decode_tok_s"] = 1.0 / itl
                out["tpot_ms"] = itl * 1e3
                out["n_out"] = len(seq.output)
        streaming = out["bytes"] - llm.model.embed.numel() * 2
        out["stream_mb"] = streaming / 1e6
        if "tpot_ms" in out:
            out["bw_gbs"] = streaming / (out["tpot_ms"] / 1e3) / 1e9
        out["spec_steps"] = llm.engine.stats["spec_steps"]

        # concurrency
        llm.engine.stats["spec_steps"] = 0
        for n in (8, 32):
            t0 = time.perf_counter()
            seqs = [llm.engine.add_request(prompt_ids, SamplingParams(max_tokens=64, temperature=0.0))
                    for _ in range(n)]
            while any(s.state != 3 for s in seqs):
                llm.engine.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            out[f"agg_tok_s_{n}"] = sum(len(s.output) for s in seqs) / dt
        return out
    finally:
        llm.shutdown()


def measure_hf(model_path: str, prompt_len: int = 64, gen_len: int = 128) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda()
    model.eval()
    prompt = "Write a short paragraph about GPUs."
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).cuda()
    out = {"engine": "hf-fp16", "model": model_path.rstrip("/").split("/")[-1],
           "bytes": sum(p.numel() * p.element_size() for p in model.parameters())}
    # prefill only: one forward
    with torch.no_grad():
        for _ in range(2):
            model(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(ids)
        torch.cuda.synchronize()
        out["prefill_tok_s"] = ids.shape[1] / (time.perf_counter() - t0)
    # greedy generation
    with torch.no_grad():
        model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(2):
            t0 = time.perf_counter()
            gen = model.generate(ids, max_new_tokens=gen_len, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            best = min(best, dt / max(gen.shape[1] - ids.shape[1], 1))
    out["tpot_ms"] = best * 1e3
    out["decode_tok_s"] = 1.0 / best
    out["stream_mb"] = out["bytes"] / 1e6
    out["bw_gbs"] = out["bytes"] / best / 1e9
    del model
    return out


if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    args["model_path"] = args.pop("model")
    if args["engine"] == "kaze":
        res = measure_kaze(**{k: v for k, v in args.items() if k != "engine"})
    elif args["engine"] == "hf":
        res = measure_hf(**{k: v for k, v in args.items() if k != "engine"})
    else:
        raise SystemExit(f"unknown engine {args['engine']}")
    print("RESULT " + json.dumps(res))
