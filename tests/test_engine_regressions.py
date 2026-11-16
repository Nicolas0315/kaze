"""Scheduler/sampling regressions without CUDA: python3 tests/test_engine_regressions.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaze.config import ModelConfig
from kaze.engine import DECODE, DONE, WAITING, Engine
from kaze.sampler import SamplingParams


class CPUModel:
    """Only replace GPU forwards; keep the real engine and block allocator."""
    def __init__(self, logits=None):
        self.device = torch.device("cpu")
        self.cfg = ModelConfig(vocab_size=5, hidden_size=4, intermediate_size=4,
                               num_hidden_layers=1, num_attention_heads=1,
                               num_key_value_heads=1, max_position_embeddings=128)
        self.fixed_logits = logits
        self.prefills = []
        self.buffers = {}

    def get_buffers(self, tdim):
        if tdim not in self.buffers:
            self.buffers[tdim] = SimpleNamespace(
                pos=torch.zeros(tdim, dtype=torch.long), logits=torch.zeros(tdim, 5))
        return self.buffers[tdim]

    def _logits(self, ids):
        if self.fixed_logits is not None:
            return torch.tensor(self.fixed_logits).repeat(len(ids), 1)
        out = torch.full((len(ids), 5), -100.0)
        out.scatter_(1, ((ids + 1) % 5).view(-1, 1), 10.0)
        return out

    def prefill_logits(self, ids, positions, cache, bt, start):
        self.prefills.append(ids.tolist())
        return self._logits(ids[-1:])

    def decode_logits(self, ids, pos, slots, bt, lens, cache, tdim, attn_mode):
        return self._logits(ids)

    def verify_logits(self, ids, positions, slots, cache, bt, qs, sl, tdim, n_real):
        out = self.get_buffers(tdim).logits
        out[:n_real].copy_(self._logits(ids[:n_real]))
        return out


def make_engine(**kwargs):
    model = kwargs.pop("model", CPUModel())
    return Engine(model, num_blocks=kwargs.pop("num_blocks", 32),
                  block_size=kwargs.pop("block_size", 2),
                  max_batch=kwargs.pop("max_batch", 4), use_cuda_graph=False,
                  enable_prefix_cache=False, verbose=False, **kwargs)


class EngineRegressions(unittest.TestCase):
    def test_empty_tokens_are_rejected_before_enqueue(self):
        engine = make_engine()
        for prompt in [[], torch.tensor([], dtype=torch.long)]:
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    engine.add_request(prompt)
        self.assertTrue(engine.is_idle())

    def test_decode_retries_allocation_and_skips_preempted_snapshot_entries(self):
        engine = make_engine(num_blocks=2)
        seqs = [engine.add_request([0, 1], SamplingParams(max_tokens=4)) for _ in range(2)]
        engine.waiting.clear()
        for seq in seqs:
            seq.output.append(2)
            seq.num_computed = 2
            seq.state = DECODE
            self.assertTrue(engine._ensure_blocks(seq, 2))
        engine.running = list(seqs)
        outs = []
        engine._decode_step(outs)
        self.assertEqual([out.request_id for out in outs], [seqs[0].request_id])
        self.assertEqual(seqs[0].num_computed, 3)
        self.assertEqual(seqs[1].state, WAITING)
        self.assertEqual(seqs[1].blocks, [])
        self.assertEqual(engine.stats["preemptions"], 1)
        self.assertEqual(sum(engine.cache.allocator.ref), 2)

    def test_preempted_sequence_recomputes_generated_history(self):
        engine = make_engine()
        seq = engine.add_request([0], SamplingParams(max_tokens=4))
        engine._prefill_step([], 8)
        self.assertEqual(seq.output, [1])
        # Exercise the same freed state produced by _preempt, preserving output.
        engine.running.remove(seq)
        engine._free(seq)
        seq.state = WAITING
        engine.waiting.append(seq)
        engine._prefill_step([], 8)
        self.assertEqual(engine.model.prefills[-1], [0, 1])
        self.assertEqual(seq.output, [1, 2])
        self.assertEqual(seq.num_computed, 2)

    def test_running_sequences_never_exceed_max_batch(self):
        engine = make_engine(max_batch=1)
        seqs = [engine.add_request([0], SamplingParams(max_tokens=2)) for _ in range(3)]
        for _ in range(10):
            engine.step()
            self.assertLessEqual(len(engine.running), 1)
            if engine.is_idle():
                break
        self.assertTrue(all(seq.state == DONE for seq in seqs))
        self.assertEqual([seq.output for seq in seqs], [[1, 2], [1, 2], [1, 2]])

    def test_speculation_preserves_repetition_penalty(self):
        results = []
        for speculative in [False, True]:
            engine = make_engine(model=CPUModel([10.0, 9.0, -100.0, -100.0, -100.0]),
                                 speculative=speculative, spec_k=4, spec_accept_ema=2.0)
            seq = engine.add_request([0, 0, 0], SamplingParams(
                max_tokens=5, temperature=0, repetition_penalty=2))
            engine.waiting.clear()
            seq.state = DECODE
            seq.num_computed = 2
            engine.running.append(seq)
            self.assertTrue(engine._ensure_blocks(seq, 3))
            outs = []
            engine._decode_step(outs)
            results.append([out.tokens[0] for out in outs])
        self.assertEqual(results[0], [1])
        self.assertEqual(results[1], [1])

    def _sample_sequence(self, seed, with_noise=False):
        engine = make_engine()
        seq = engine.add_request([0], SamplingParams(
            max_tokens=24, temperature=1.0, seed=seed))
        noise = engine.add_request([0], SamplingParams(max_tokens=24, temperature=1.0))
        rows = [noise, seq] if with_noise else [seq]
        for _ in range(24):
            engine._sample_and_append(rows, torch.zeros(len(rows), 5), [])
        return seq.output

    def test_request_seed_is_independent_of_other_requests(self):
        self.assertEqual(self._sample_sequence(123), self._sample_sequence(123, True))

    def test_distinct_request_seeds_change_sampling(self):
        self.assertNotEqual(self._sample_sequence(123), self._sample_sequence(321))


if __name__ == "__main__":
    unittest.main(verbosity=2)
