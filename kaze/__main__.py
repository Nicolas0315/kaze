"""Command line entry point: kaze serve | generate | bench."""
from __future__ import annotations

import argparse
import time


def _maybe_qwen38(args) -> bool:
    """Qwen3.8（Qwen3_5 text）なら専用の経路で生成して True を返す。"""
    import json
    import os

    path = args.model
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)
    text_cfg = raw.get("text_config", raw)
    if text_cfg.get("model_type") not in ("qwen3_5_text", "qwen3_5"):
        return False

    import time

    from transformers import AutoTokenizer

    from kaze.qwen38 import generate, generate_graph, load_qwen38

    try:
        tok = AutoTokenizer.from_pretrained(path)
        ids = tok(args.prompt).input_ids
        if not ids:      # tokenizer だけ有って語彙が合わない場合は id 直指定に落とす
            raise ValueError("empty encoding")
        decode = lambda t: tok.decode(t, skip_special_tokens=True)
        eos = (tok.eos_token_id,) if tok.eos_token_id is not None else ()
    except Exception:
        # tokenizer が無いディレクトリ（重みだけの配布）では id をそのまま使う
        tok = None
        ids = [int(x) for x in args.prompt.replace(",", " ").split()] or [1]
        decode = lambda t: str(t)
        eos = ()
    model = load_qwen38(path)
    t0 = time.perf_counter()
    out = generate(model, ids, max_tokens=args.max_tokens,                   temperature=args.temperature, eos_token_ids=eos)
    dt = time.perf_counter() - t0
    print(decode(out))
    print(f"--- {len(out)} tokens in {dt:.2f}s ({len(out) / dt:.1f} tok/s)")
    return True


def _maybe_qwen38_serve(args) -> bool:
    """Qwen3.8（Qwen3_5 text）なら専用の HTTP サーバを立てて True を返す。"""
    import json
    import os

    cfg_path = os.path.join(args.model, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("text_config", raw).get("model_type") not in ("qwen3_5_text", "qwen3_5"):
        return False

    import uvicorn
    from transformers import AutoTokenizer

    from kaze.qwen38 import load_qwen38
    from kaze.qwen38_server import create_app

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_qwen38(args.model, kind=getattr(args, "kind", "int8"))
    app = create_app(model, tok, model_name=os.path.basename(args.model.rstrip("/")),
                     graph=getattr(args, "graph", False))
    uvicorn.run(app, host=args.host, port=args.port)
    return True


def main() -> None:
    ap = argparse.ArgumentParser("kaze")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True)
        p.add_argument("--kind", default="int8", choices=["int4", "int8", "fp16"])
        # 128 matches LLM's default and is 13% faster for int4; pass --group 64 for a
        # ~19% KL improvement at that speed cost
        p.add_argument("--group", type=int, default=128)
        p.add_argument("--cache-dtype", default="fp16",
                       choices=["fp16", "fp8v", "fp8"],
                       help="KV cache dtype: fp8v keeps K in fp16 and V in fp8 "
                            "(25%% less memory, no measured quality loss)")
        p.add_argument("--lm-head-kind", default="int8", choices=["int4", "int8", "fp16"])
        p.add_argument("--max-batch", type=int, default=32)
        p.add_argument("--max-batched-tokens", type=int, default=2048)
        p.add_argument("--no-cuda-graph", action="store_true")
        p.add_argument("--no-spec", action="store_true")
        p.add_argument("--spec-k", type=int, default=8)
        p.add_argument("--num-blocks", type=int, default=None)

    s = sub.add_parser("serve")
    common(s)
    s.add_argument("--graph", action="store_true",
                   help="CUDA Graph で decode を固める（Qwen3.8 系のみ）")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    g = sub.add_parser("generate")
    common(g)
    g.add_argument("prompt")
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--chat", action="store_true")
    g.add_argument("--graph", action="store_true",
                   help="CUDA Graph で decode を固める（Qwen3.8 系のみ）")

    b = sub.add_parser("bench")
    common(b)
    b.add_argument("--prompt-len", type=int, default=512)
    b.add_argument("--gen-len", type=int, default=128)
    b.add_argument("--concurrency", type=int, default=1)
    b.add_argument("--repeats", type=int, default=3)

    args = ap.parse_args()
    # Qwen3.8（Qwen3_5 text）は専用の経路で動かす（dense 用のローダを通さない）
    if args.cmd == "generate" and _maybe_qwen38(args):
        return
    if args.cmd == "serve" and _maybe_qwen38_serve(args):
        return

    from kaze.llm import LLM

    llm = LLM(args.model, kind=args.kind, group=args.group, lm_head_kind=args.lm_head_kind,
               cache_dtype=args.cache_dtype,
              max_batch=args.max_batch, max_batched_tokens=args.max_batched_tokens,
              use_cuda_graph=not args.no_cuda_graph, num_blocks=args.num_blocks,
              speculative=not args.no_spec, spec_k=args.spec_k)

    if args.cmd == "generate":
        from kaze.sampler import SamplingParams
        p = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
        t0 = time.time()
        out = llm.generate(args.prompt, p, chat=args.chat)
        dt = time.time() - t0
        print(out.text)
        print(f"--- {len(out.tokens)} tokens in {dt:.2f}s "
              f"(ttft {out.ttft_ms:.0f} ms, tpot {out.tpot_ms:.2f} ms, "
              f"{len(out.tokens) / dt:.1f} tok/s)")
    elif args.cmd == "serve":
        import uvicorn
        from kaze.server import create_app
        app = create_app(llm)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.cmd == "bench":
        from bench.run import run_bench
        run_bench(llm, args)


if __name__ == "__main__":
    main()
