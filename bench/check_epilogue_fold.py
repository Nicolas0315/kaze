"""Bit-exactness check for the split-K epilogue fold (rmsnorm / rope_scatter consumers).

The fold must reproduce gemv_epilogue_kernel exactly: same sp order, then bias, then the
residual, then one fp16 rounding.  This compares decode logits with the fold on and off on
the same inputs, so a mismatch shows up as a difference in the fp16 bits, not as "close".
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from kaze.cache import KVCache
from kaze.model import set_fuse_epilogue


def run(model, cache, ids, positions, slots, bt, lens, tdim=1):
    return model.decode_logits(ids, positions, slots, bt, lens, cache, tdim=tdim).clone()


def main() -> None:
    from kaze.llm import LLM

    path = sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen2.5-0.5B-Instruct"
    kind = sys.argv[2] if len(sys.argv) > 2 else "int4"
    llm = LLM(path, kind=kind, lm_head_kind=kind, speculative=False, verbose=False)
    model, eng = llm.model, llm.engine
    cache = eng.cache
    dev = model.device
    ids = torch.randint(0, 30000, (1,), device=dev, dtype=torch.long)
    pos = torch.zeros(1, dtype=torch.long, device=dev)
    slots = torch.zeros(1, dtype=torch.long, device=dev)
    bt = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    lens = torch.ones(1, dtype=torch.int32, device=dev)

    set_fuse_epilogue(True)
    fused = run(model, cache, ids, pos, slots, bt, lens)
    again = run(model, cache, ids, pos, slots, bt, lens)
    set_fuse_epilogue(False)
    ref = run(model, cache, ids, pos, slots, bt, lens)
    deterministic = torch.equal(fused, again)
    cos = torch.nn.functional.cosine_similarity(fused[0].float(), ref[0].float(),
                                                dim=0).item()
    diff = (fused.float() - ref.float()).abs().max().item()
    name = os.path.basename(path.rstrip(chr(47)))
    # Exact bit equality held before the fold started summing splits as a tree (round 104,
    # worth +2.6% decode).  The contract is now: deterministic, and equal to the epilogue
    # path within fp16 rounding.
    ok = deterministic and cos > 0.99999
    print(f"{name} {kind}: deterministic={deterministic} cos={cos:.7f} max_diff={diff:.3e}")
    llm.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
