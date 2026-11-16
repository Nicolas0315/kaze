"""Command line entry point: kaze serve | generate | bench."""
from __future__ import annotations

import argparse
import math
import time


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
        # None follows --kind, matching LLM(lm_head_kind=None).  The old default of int8
        # disagreed with the library and cost single-stream users a lot: measured through
        # the server on 0.5B (128-token prose generation) int4 head runs 2,275 tok/s vs
        # 1,380 with an int8 head (+65%), and batch throughput is unchanged (9,076 vs
        # 9,163 at 32 streams).  For the last 0.1 of KL on small models pass
        # --lm-head-kind int8 (KL 0.093 -> 0.069 on 0.5B, -9% decode).
        p.add_argument("--lm-head-kind", default=None,
                       choices=["int4", "int8", "fp16"])
        p.add_argument("--max-batch", type=int, default=64)
        p.add_argument("--max-batched-tokens", type=int, default=2048)
        p.add_argument("--no-cuda-graph", action="store_true")
        p.add_argument("--no-spec", action="store_true")
        # keep this in step with LLM(spec_k=...): the CLI default used to be 8 while the
        # library default was 16, so `kaze serve` silently ran the narrow draft.  Measured
        # on 7B at 512 tokens: 445 tok/s (k=8) vs 1,466 (k=16) -- bench/server_spec.py.
        p.add_argument("--spec-k", type=int, default=16)
        p.add_argument("--num-blocks", type=int, default=None)

    s = sub.add_parser("serve")
    common(s)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    g = sub.add_parser("generate")
    common(g)
    g.add_argument("prompt")
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--chat", action="store_true")

    b = sub.add_parser("bench")
    common(b)
    b.add_argument("--prompt-len", type=int, default=512)
    b.add_argument("--gen-len", type=int, default=128)
    b.add_argument("--concurrency", type=int, default=1)
    b.add_argument("--repeats", type=int, default=3)

    args = ap.parse_args()
    positive = ["group", "max_batch", "max_batched_tokens", "spec_k", "num_blocks"]
    if args.cmd == "generate":
        positive.append("max_tokens")
        if not math.isfinite(args.temperature) or args.temperature < 0:
            ap.error("--temperature must be finite and non-negative")
    elif args.cmd == "bench":
        positive.extend(["prompt_len", "concurrency", "repeats"])
        if args.gen_len < 2:
            ap.error("--gen-len must be at least 2 to measure decode timing")
    elif not 1 <= args.port <= 65535:
        ap.error("--port must be between 1 and 65535")
    for name in positive:
        value = getattr(args, name)
        if value is not None and value <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.group % 8:
        ap.error("--group must be a multiple of 8")
    from kaze.llm import LLM

    llm = LLM(args.model, kind=args.kind, group=args.group, lm_head_kind=args.lm_head_kind,
               cache_dtype=args.cache_dtype,
              max_batch=args.max_batch, max_batched_tokens=args.max_batched_tokens,
              use_cuda_graph=not args.no_cuda_graph, num_blocks=args.num_blocks,
              speculative=not args.no_spec, spec_k=args.spec_k,
              start_loop=args.cmd != "bench")

    try:
        if args.cmd == "generate":
            from kaze.sampler import SamplingParams
            p = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
            t0 = time.time()
            prompt = [{"role": "user", "content": args.prompt}] if args.chat else args.prompt
            out = llm.generate(prompt, p, chat=args.chat)
            dt = time.time() - t0
            ttft = "n/a" if out.ttft_ms is None else f"{out.ttft_ms:.0f} ms"
            tpot = "n/a" if out.tpot_ms is None else f"{out.tpot_ms:.2f} ms"
            print(out.text)
            print(f"--- {len(out.tokens)} tokens in {dt:.2f}s "
                  f"(ttft {ttft}, tpot {tpot}, {len(out.tokens) / dt:.1f} tok/s)")
        elif args.cmd == "serve":
            import uvicorn
            from kaze.server import create_app
            # NOTE: torch.set_num_threads(1) here was tried for the 7B-at-32 problem
            # (bare process 16 ms/step vs 37 ms under the server) and made no difference
            # (875 -> 861 tok/s), so it is not the cause.  See docs/scorecard round 160.
            app = create_app(llm)
            uvicorn.run(app, host=args.host, port=args.port)
        elif args.cmd == "bench":
            from bench.run import run_bench
            run_bench(llm, args)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
