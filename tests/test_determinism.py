"""Decode determinism across batch shapes (B10).

Two separate properties are pinned here:

1. Padding rows of a partially filled bucket must not touch the cache.  They used to keep
   slot 0 / block table 0, and the allocator hands out block 0 first, so a batch decode
   overwrote a live sequence's first token in every layer on every step.  The fix makes
   every padding row a copy of row 0, which writes byte-identical K/V back into row 0's own
   slot.

2. Greedy decode must not depend on whether a sequence shares its batch.  It does by
   default, because the row bucket selects both the GEMV config family and the kernel
   family (TDIM=1 matvec + fp16-activation gate_up vs TDIM>=16 tl.dot + int8-activation
   gate_up).  Engine.set_deterministic / KAZE_DETERMINISTIC=1 pins one bucket, and then solo
   and batched decode produce identical tokens.

Run: python3 tests/test_determinism.py
"""
from __future__ import annotations

import sys
import unittest

sys.path.insert(0, ".")

import torch

from kaze.engine import Engine
from kaze.loader import load_model
from kaze.sampler import SamplingParams

MODEL = "/models/Qwen2.5-0.5B-Instruct"
PROMPT = "The history of computing begins with"
OTHER = "A short story about a lighthouse keeper:"
GEN = 24


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


_MODEL = None


def _model():
    """The 0.5B int4 model, loaded once: five loads in a row is 5x the VRAM churn."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(MODEL, kind="int4", group=128, lm_head_kind="int4", verbose=False)
        _MODEL.sparse_frac = 0.0
    return _MODEL


def _engine(model, **kw):
    return Engine(model, max_batch=kw.pop("max_batch", 8), use_cuda_graph=False,
                  verbose=False, num_blocks=kw.pop("num_blocks", 256), **kw)


def _decode(eng, ids, gen: int = GEN):
    seq = eng.add_request(list(ids), SamplingParams(max_tokens=gen, temperature=0.0,
                                                    ignore_eos=True))
    while seq.state != 3:
        eng.step()
    return list(seq.output)


def _batch(eng, ids_list, gen: int = GEN):
    seqs = [eng.add_request(list(i), SamplingParams(max_tokens=gen, temperature=0.0,
                                                    ignore_eos=True)) for i in ids_list]
    while any(s.state != 3 for s in seqs):
        eng.step()
    return [list(s.output) for s in seqs]


class PaddingRowsTest(unittest.TestCase):
    """Property 1: a padding row is a copy of row 0 and writes nothing new."""

    def test_padding_rows_duplicate_row0(self):
        model = _model()
        eng = _engine(model)
        tok = _tokenizer()
        a = eng.add_request(list(tok.encode(PROMPT)),
                            SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
        b = eng.add_request(list(tok.encode(OTHER)),
                            SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
        for _ in range(6):
            eng.step()
            if eng.stats["decode_tokens"] >= 2:
                break
        self.assertGreaterEqual(eng.stats["decode_tokens"], 2, "no batched decode step ran")
        st = eng._static(16)                     # 2 sequences -> bucket 16
        self.assertTrue(torch.equal(st["ids"][2:], st["ids"][:1].expand(14)))
        self.assertTrue(torch.equal(st["positions"][2:], st["positions"][:1].expand(14)))
        self.assertTrue(torch.equal(st["slots"][2:], st["slots"][:1].expand(14)))
        self.assertTrue(torch.equal(st["seq_lens"][2:], st["seq_lens"][:1].expand(14)))
        self.assertTrue(torch.equal(st["block_tables"][2:], st["block_tables"][:1].expand(14, -1)))
        self.assertEqual(eng.buckets, [1, 16])
        self.assertTrue(a.output and b.output)

    def test_batched_decode_leaves_a_live_sequences_first_token_alone(self):
        model = _model()
        eng = _engine(model)
        tok = _tokenizer()
        cfg = model.cfg
        kv = (cfg.num_key_value_heads, cfg.head_dim)
        a = eng.add_request(list(tok.encode(PROMPT)),
                            SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
        eng.step()                               # prefill A alone
        slot0 = a.blocks[0] * eng.block_size
        layers = (0, len(model.layers) - 1)
        before = {li: eng.cache.k[li].view(-1, *kv)[slot0].clone() for li in layers}
        vbefore = {li: eng.cache.v[li].view(-1, *kv)[slot0].clone() for li in layers}

        eng.add_request(list(tok.encode(OTHER)),
                        SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
        for _ in range(6):
            if eng.stats["decode_tokens"]:
                break
            eng.step()
        self.assertTrue(eng.stats["decode_tokens"], "no decode step ran")
        for li in layers:
            self.assertTrue(torch.equal(before[li], eng.cache.k[li].view(-1, *kv)[slot0]),
                            f"layer {li}: batched decode rewrote A's first-token K")
            self.assertTrue(torch.equal(vbefore[li], eng.cache.v[li].view(-1, *kv)[slot0]),
                            f"layer {li}: batched decode rewrote A's first-token V")


class BatchShapeTest(unittest.TestCase):
    """Property 2: greedy output does not depend on the batch shape."""

    def test_identical_prompts_in_one_batch_agree(self):
        model = _model()
        eng = _engine(model, max_batch=16)
        tok = _tokenizer()
        ids = tok.encode(PROMPT)
        outs = _batch(eng, [ids] * 4)
        for i, o in enumerate(outs):
            self.assertEqual(o, outs[0], f"row {i} differs from row 0")

    def test_deterministic_mode_makes_solo_equal_batch(self):
        model = _model()
        eng = _engine(model, max_batch=16)
        tok = _tokenizer()
        ids = tok.encode(PROMPT)
        eng.set_deterministic(True)
        self.assertEqual(eng.buckets, [16])
        # Single-row decode now runs the 16-row kernels, so its tokens are allowed to differ
        # from the tuned batch-1 path (measured: rms logit gap 0.010, argmax flips after a
        # few dozen tokens).  What must hold is that the batch shape stops mattering.
        solo = _decode(eng, ids)
        batched = _batch(eng, [ids] * 4)
        for i, o in enumerate(batched):
            self.assertEqual(o, solo, f"row {i} of the batch differs from the solo run")

    def test_deterministic_mode_is_off_by_default(self):
        model = _model()
        eng = _engine(model)
        self.assertFalse(eng.deterministic)
        self.assertEqual(eng.buckets, [1, 16])


if __name__ == "__main__":
    unittest.main(verbosity=2)
