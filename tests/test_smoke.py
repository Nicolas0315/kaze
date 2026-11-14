"""Quality smoke test across quantisation settings.

Silent corruption (garbage text) is invisible to shape/NaN checks: round 71 found that
int4 with group=64 -- the CLI default -- produced KL 11.6 because the int8-tensor-core
prefill kernel hardcoded BLOCK_K=128.  This test pins a factual answer across the
parameter combinations a user actually hits.

Run: python3 tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer

from kaze.engine import Engine, SamplingParams
from kaze.loader import load_model

MODELS = os.environ.get("KAZE_MODELS", os.path.expanduser("~/models"))
MODEL = os.path.join(MODELS, "Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
EXPECT = "Paris"

# (kind, group, lm_head_kind) -- the first one is the CLI default for int4
COMBOS = [
    ("int4", 64, "int4"),
    ("int4", 128, "int4"),
    ("int8", 64, "int8"),
    ("mixed", 128, "int4"),
]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    fails = []
    for kind, group, head in COMBOS:
        model = load_model(MODEL, kind=kind, group=group, lm_head_kind=head, verbose=False)
        model.sparse_frac = 0.0
        eng = Engine(model, max_batch=1, use_cuda_graph=False, verbose=False, num_blocks=256)
        ids = tok(PROMPT).input_ids
        seq = eng.add_request(list(ids), SamplingParams(max_tokens=8, temperature=0.0))
        while seq.state != 3:
            eng.step()
        text = tok.decode(seq.output)
        ok = EXPECT.lower() in text.lower()
        print(f"{kind:6s} group={group:3d} head={head:5s} -> {text!r} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append((kind, group, text))
        del eng, model
        torch.cuda.empty_cache()
    print()
    print("all ok" if not fails else f"{len(fails)} combo(s) produced wrong text")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())