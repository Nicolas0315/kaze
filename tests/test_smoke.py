"""Quality smoke test across quantisation settings.

Silent corruption (garbage text) is invisible to shape/NaN checks: round 71 found that
int4 with group=64 produced KL 11.6 because the int8-tensor-core
prefill kernel hardcoded BLOCK_K=128.  This test pins a factual answer across the
parameter combinations a user actually hits.

Run: python3 tests/test_smoke.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer

from kaze.engine import Engine, SamplingParams
from kaze.loader import load_model

MODEL = "/models/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"
EXPECT = "Paris"

# (kind, group, lm_head_kind): only group 128 is supported now (see main() for why).
COMBOS = [
    ("int8", 128, "int8"),  # CLI defaults
    ("int4", 128, "int8"),  # CLI with only --kind int4
    ("int4", 128, "int4"),
    ("mixed", 128, "int4"),
]

# A prompt long enough to take the batched prefill path (m >= 64 rows).  The short
# "The capital of France is" prompt hides the group-64 corruption entirely because it
# runs the batch-1 kernels instead -- that is how round 71's bug survived until it was
# re-checked with a long prompt.
LONG_PROMPT = ("The history of computing begins with mechanical calculators and continues "
               "through punched cards, relays, vacuum tubes and transistors. ") * 6


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
        # the same model through the batched prefill path must not corrupt the output
        long_ids = tok(LONG_PROMPT).input_ids
        seq2 = eng.add_request(list(long_ids),
                               SamplingParams(max_tokens=8, temperature=0.0))
        while seq2.state != 3:
            eng.step()
        long_text = tok.decode(seq2.output)
        print(f"{kind:6s} group={group:3d} long({len(long_ids)} tok) -> {long_text!r}")
        if "!!!!" in long_text or len(long_text.strip()) < 2:
            fails.append((kind, group, long_text))
        del eng, model
        torch.cuda.empty_cache()

    # group 64 as well: the fused activation quantiser used to hardcode 128-wide scale
    # groups, which corrupted any prompt long enough for the batched prefill path
    # (round 189).  Keep a combo so that regression cannot come back silently.
    for kind, group, head in (("int4", 64, "int4"),):
        model = load_model(MODEL, kind=kind, group=group, lm_head_kind=head, verbose=False)
        model.sparse_frac = 0.0
        eng = Engine(model, max_batch=1, use_cuda_graph=False, verbose=False, num_blocks=256)
        long_ids = tok(LONG_PROMPT).input_ids
        seq = eng.add_request(list(long_ids), SamplingParams(max_tokens=8, temperature=0.0))
        while seq.state != 3:
            eng.step()
        text = tok.decode(seq.output)
        ok = "!!!!" not in text and len(text.strip()) >= 2
        print(f"{kind:6s} group={group:3d} long({len(long_ids)} tok) -> {text!r} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append((kind, group, text))
        del eng, model
        torch.cuda.empty_cache()
    print()
    print("all ok" if not fails else f"{len(fails)} combo(s) produced wrong text")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())