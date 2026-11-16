"""CPU-only scheduler/cache audit: python3 tests/test_scheduler_audit.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaze.cache import BlockAllocator
from kaze.engine import DECODE, DONE, Engine
from kaze.sampler import SamplingParams
from test_engine_regressions import CPUModel


def make_engine(**kwargs):
    return Engine(kwargs.pop("model", CPUModel()),
                  num_blocks=kwargs.pop("num_blocks", 32),
                  block_size=kwargs.pop("block_size", 2),
                  use_cuda_graph=False, verbose=False, **kwargs)


class CacheWriter(CPUModel):
    """Replace GPU forwards, preserving the real cache write side effect."""
    def prefill_logits(self, ids, positions, cache, bt, start, **kwargs):
        self.prefills.append(ids.tolist())
        slots = cache.slots_of(bt.view(-1), start, start + len(ids))
        cache.k_flat[:, slots] = float(len(self.prefills))
        cache.v_flat[:, slots] = float(len(self.prefills))
        return self._logits(ids[-1:])


class BranchReader(CPUModel):
    def verify_logits_multi(self, ids, positions, slots, cache, bt, groups, tdim, n_real):
        self.prefixes = []
        for _lo, _hi, branch, start, _end in groups:
            block = int(bt[branch, start // cache.block_size])
            self.prefixes.append(cache.k[:, block, :start % cache.block_size].clone())
        return self._logits(ids)


class SchedulerAudit(unittest.TestCase):
    def test_engine_sizes_are_positive_integers(self):
        for name in ("max_batch", "max_batched_tokens", "block_size",
                     "num_blocks", "max_model_len", "spec_k"):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        make_engine(**{name: value})

    def test_invalid_token_ids_are_rejected_before_enqueue(self):
        engine = make_engine()
        invalid_prompts = ([-1], [5], [1.5], [True], ["1"], [[1]],
                           torch.tensor([1.0]), torch.tensor([[1]]), torch.tensor(1))
        for prompt in invalid_prompts:
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    engine.add_request(prompt)
                self.assertTrue(engine.is_idle())

    def test_failed_block_growth_rolls_back_partial_allocations(self):
        engine = make_engine(num_blocks=2, enable_prefix_cache=False)
        seq = engine.add_request([0])
        self.assertTrue(engine._ensure_blocks(seq, 1))
        original = list(seq.blocks)
        self.assertFalse(engine._ensure_blocks(seq, 5))
        self.assertEqual(seq.blocks, original)
        self.assertEqual(sum(engine.cache.allocator.ref), 1)
        self.assertEqual(engine.cache.free_blocks, 1)

    def test_prompt_that_fills_cache_can_complete_without_slack_block(self):
        engine = make_engine(num_blocks=1, enable_prefix_cache=False)
        seq = engine.add_request([0, 1], SamplingParams(max_tokens=1, temperature=0))
        outs = engine.step()
        self.assertEqual(seq.state, DONE)
        self.assertEqual([out.tokens for out in outs], [[2]])
        self.assertTrue(engine.is_idle())

    def test_prompt_larger_than_cache_is_rejected_before_enqueue(self):
        engine = make_engine(num_blocks=1)
        with self.assertRaises(ValueError):
            engine.add_request([0, 1, 2])
        self.assertTrue(engine.is_idle())

    def test_single_decode_reports_capacity_exhaustion_instead_of_spinning(self):
        engine = make_engine(num_blocks=1, enable_prefix_cache=False)
        seq = engine.add_request([0, 1], SamplingParams(max_tokens=4, temperature=0))
        engine.waiting.clear()
        seq.output.append(2)
        seq.num_computed = 2
        seq.state = DECODE
        engine.running.append(seq)
        self.assertTrue(engine._ensure_blocks(seq, 2))
        outs = engine.step()
        self.assertEqual(seq.state, DONE)
        self.assertEqual(len(outs), 1)
        self.assertTrue(outs[0].finished)
        self.assertEqual(outs[0].tokens, [])
        self.assertEqual(outs[0].error, "KV cache exhausted")
        self.assertTrue(engine.is_idle())
        self.assertEqual(engine.cache.free_blocks, 1)
        following = engine.add_request([0], SamplingParams(max_tokens=1, temperature=0))
        engine.step()
        self.assertEqual(following.state, DONE)

    def test_cached_prefix_is_not_rewritten_to_recover_final_logits(self):
        engine = make_engine(model=CacheWriter())
        first = engine.add_request([0, 1, 2, 3], SamplingParams(max_tokens=4, temperature=0))
        engine._prefill_step([], 16)
        block = first.blocks[1]
        before = engine.cache.k[:, block].clone()
        second = engine.add_request([0, 1, 2, 3], SamplingParams(max_tokens=4, temperature=0))
        engine._prefill_step([], 16)
        torch.testing.assert_close(engine.cache.k[:, block], before)
        self.assertEqual(first.blocks[0], second.blocks[0])
        self.assertNotEqual(first.blocks[1], second.blocks[1])
        self.assertEqual(first.output, second.output)

    def test_free_releases_draft_model_blocks_too(self):
        engine = make_engine(draft_model=CacheWriter(), use_draft_model=True)
        seq = engine.add_request([0, 1, 2])
        seq.num_computed = 2
        self.assertTrue(engine._draft_tokens_from_model(seq, 1))
        self.assertGreater(sum(engine.draft_cache.allocator.ref), 0)
        engine._free(seq)
        self.assertEqual(sum(engine.draft_cache.allocator.ref), 0)
        self.assertEqual(seq.draft_len, 0)
        self.assertFalse(seq.draft_blocks)

    def test_speculative_branch_copies_the_real_cached_prefix(self):
        model = BranchReader()
        engine = make_engine(model=model, speculative=True, block_size=16)
        engine.spec_chains = 2
        engine.spec_chain_tdim = 4
        seq = engine.add_request([0, 1, 2, 0, 1, 3, 0, 1],
                                 SamplingParams(max_tokens=4, temperature=0))
        engine.waiting.clear()
        seq.num_computed = 7
        seq.state = DECODE
        engine.running.append(seq)
        self.assertTrue(engine._ensure_blocks(seq, 8))
        engine.cache.k[:, seq.blocks[0], :7].fill_(7)
        self.assertTrue(engine._spec_decode_chains(seq, []))
        self.assertEqual(len(model.prefixes), 2)
        for prefix in model.prefixes:
            self.assertTrue(torch.all(prefix == 7))

    def test_generate_propagates_local_capacity_error(self):
        engine = make_engine(num_blocks=1, enable_prefix_cache=False)
        out = engine.generate([0, 1], SamplingParams(max_tokens=4, temperature=0))
        self.assertTrue(out.finished)
        self.assertEqual(out.error, "KV cache exhausted")
        self.assertEqual(out.tokens, [2])

    def test_draft_capacity_shortage_falls_back_without_partial_leak(self):
        engine = make_engine(num_blocks=2, draft_model=CacheWriter(), use_draft_model=True)
        seq = engine.add_request([0, 1, 2])
        seq.num_computed = 2
        self.assertEqual(engine._draft_tokens_from_model(seq, 6), [])
        self.assertEqual(sum(engine.draft_cache.allocator.ref), 0)
        self.assertFalse(seq.draft_blocks)

    def test_penalties_include_tokens_older_than_128_positions(self):
        for penalties in ({"repetition_penalty": 2}, {"presence_penalty": 2}):
            with self.subTest(penalties=penalties):
                model = CPUModel()
                model.cfg.max_position_embeddings = 256
                engine = make_engine(model=model, num_blocks=80)
                seq = engine.add_request([0] + [2] * 129, SamplingParams(
                    max_tokens=1, temperature=0, **penalties))
                engine._sample_and_append([seq], torch.tensor([[10., 9., 0., -100., -100.]]), [])
                self.assertEqual(seq.output, [1])

    def test_submitted_params_are_a_request_snapshot(self):
        engine = make_engine()
        params = SamplingParams(max_tokens=2, temperature=0)
        seq = engine.add_request([0], params)
        params.max_tokens = 5
        for _ in range(5):
            engine.step()
        self.assertEqual(seq.output, [1, 2])

    def test_stop_wins_over_simultaneous_max_tokens(self):
        for options in ({"eos_token_ids": [1]}, {}):
            with self.subTest(options=options):
                engine = make_engine(**options)
                seq = engine.add_request([0], SamplingParams(
                    max_tokens=1, temperature=0, stop_token_ids=[1]))
                outs = engine.step()
                self.assertEqual(seq.finish_reason, "stop")
                self.assertEqual(outs[0].finish_reason, "stop")

    def test_speculation_respects_token_budget_and_counts_emitted_tokens(self):
        engine = make_engine(speculative=True, spec_k=2, spec_accept_ema=2.0,
                             enable_prefix_cache=False)
        seq = engine.add_request([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0],
                                 SamplingParams(max_tokens=2, temperature=0))
        engine.step()
        outs = engine.step()
        self.assertEqual(seq.output, [1, 2])
        self.assertEqual([out.tokens for out in outs], [[2]])
        self.assertTrue(outs[-1].finished)
        self.assertEqual(engine.stats["spec_tokens"], 1)

    def test_speculative_context_limit_matches_regular_decode(self):
        outputs = []
        for speculative in (False, True):
            engine = make_engine(speculative=speculative, spec_k=2, spec_accept_ema=2.0,
                                 max_model_len=13, enable_prefix_cache=False)
            seq = engine.add_request([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0],
                                     SamplingParams(max_tokens=20, temperature=0))
            for _ in range(10):
                engine.step()
                if engine.is_idle():
                    break
            self.assertTrue(engine.is_idle())
            outputs.append(seq.output)
        self.assertEqual(outputs[0], [1, 2, 3])
        self.assertEqual(outputs[1], outputs[0])


class AllocatorAudit(unittest.TestCase):
    def test_rehashing_duplicate_block_keeps_other_block_lookup(self):
        alloc = BlockAllocator(2, 2)
        first, second = alloc.allocate(), alloc.allocate()
        alloc.register_hash(11, first)
        alloc.register_hash(11, second)
        alloc.register_hash(22, first)
        self.assertEqual(alloc.lookup(11), second)
        self.assertEqual(alloc.lookup(22), first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
