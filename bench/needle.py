"""Needle-in-haystack check for a given context length.

Used to prove a tiling/kernel change did not break long-range retrieval: one unique
passkey is buried in a random haystack and the model must reproduce it.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


FILLER = [
    "The garden was quiet in the early morning light.",
    "A train passed through the valley shortly after noon.",
    "She wrote the numbers down and checked them twice.",
    "The library closes at six on weekdays.",
    "Rain fell steadily over the northern hills.",
    "He repaired the old radio with spare parts.",
]


def build(passkey: str, target_tokens: int, depth: float = 0.5) -> str:
    rng = random.Random(0)
    sentences = []
    while len(sentences) * 12 < target_tokens:
        sentences.append(rng.choice(FILLER))
    pos = int(len(sentences) * depth)
    sentences.insert(pos, "The passkey is " + passkey + ".")
    return (" ".join(sentences)
            + "\nQuestion: What is the passkey? Answer with the number only.\nAnswer:")


def main() -> None:
    ap = argparse.ArgumentParser("needle")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="int4")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--passkey", default="7391")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--cache-dtype", default="fp16")
    ap.add_argument("--num-blocks", type=int, default=None)
    ap.add_argument("--factor", type=float, default=None,
                    help="enable YaRN rope scaling at this factor (needed when ctx >"
                         " the model's native max_position_embeddings, e.g. 8 for 128K on a 32K model)")
    args = ap.parse_args()

    from kaze.llm import LLM
    from kaze.sampler import SamplingParams

    model_path = args.model
    if args.factor:
        # YaRN has to be in the config BEFORE the rope tables are built, so point the
        # loader at a temp dir with symlinks + a patched config.json
        import json, shutil, tempfile
        src = os.path.abspath(args.model.rstrip("/"))
        tmp = tempfile.mkdtemp(prefix="kaze-yarn")
        for f in os.listdir(src):
            if f != "config.json":
                os.symlink(os.path.join(src, f), os.path.join(tmp, f))
        cfg = json.load(open(os.path.join(src, "config.json")))
        cfg["rope_scaling"] = {"rope_type": "yarn", "factor": args.factor,
                               "original_max_position_embeddings":
                               cfg.get("max_position_embeddings"), "finetuned": False}
        json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
        model_path = tmp
        # clean the temp config dir on exit: leftover kaze-yarn* dirs accumulated in
        # /var/tmp across runs and contributed to filling the disk (round 199)
        import atexit, shutil as _sh
        atexit.register(lambda d=tmp: _sh.rmtree(d, ignore_errors=True))

    llm = LLM(model_path, kind=args.kind, group=args.group, lm_head_kind=args.kind,
              cache_dtype=args.cache_dtype, num_blocks=args.num_blocks,
              speculative=False, verbose=False)
    prompt = build(args.passkey, args.ctx)
    out = llm.generate(prompt, SamplingParams(max_tokens=8, temperature=0.0))
    text = out.text.strip()
    verdict = "HIT" if args.passkey in text[:12] else "MISS"
    name = os.path.basename(args.model.rstrip(chr(47)))
    print(name + " ctx~%d: %s -> %r" % (args.ctx, verdict, text[:40]))
    llm.shutdown()


if __name__ == "__main__":
    main()
