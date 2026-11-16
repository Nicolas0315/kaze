"""Sampling boundary regressions: python3 tests/test_sampler_audit.py."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaze.sampler import SamplingParams, sample_rows


class SamplerAuditTests(unittest.TestCase):
    def test_invalid_params_are_rejected_before_inference(self):
        bad = {"max_tokens": [0, -1, 1.5], "temperature": [-1, math.nan, math.inf],
               "top_p": [0, -1, 1.1, math.nan], "top_k": [-1, 1.5],
               "repetition_penalty": [0, -1, math.inf, 1e-40, 1e40],
               "presence_penalty": [-2.1, 2.1, math.nan],
               "seed": [1.5, -(2 ** 63) - 1, 2 ** 64]}
        for key, values in bad.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        SamplingParams(**{key: value})

    def test_clone_has_independent_stop_lists(self):
        original = SamplingParams(stop=["END"], stop_token_ids=[1])
        clone = original.clone()
        clone.stop.append("STOP")
        clone.stop_token_ids.append(2)
        self.assertEqual(original.stop, ["END"])
        self.assertEqual(original.stop_token_ids, [1])

    def test_params_copy_mutable_constructor_inputs(self):
        stops, ids = ["END"], [1]
        params = SamplingParams(stop=stops, stop_token_ids=ids)
        stops.append("STOP")
        ids.append(2)
        self.assertEqual(params.stop, ["END"])
        self.assertEqual(params.stop_token_ids, [1])

    def test_repetition_sampling_preserves_input_logits(self):
        logits = torch.tensor([[10.0, 9.0]])
        p = SamplingParams(repetition_penalty=2.0)
        self.assertEqual(sample_rows(logits, [p], [[0]]).tolist(), [1])
        self.assertEqual(sample_rows(logits, [SamplingParams()]).tolist(), [0])
        torch.testing.assert_close(logits, torch.tensor([[10.0, 9.0]]))

    def test_presence_penalty_changes_greedy_choice(self):
        logits = torch.tensor([[10.0, 9.0]])
        params = SamplingParams(presence_penalty=2.0)
        self.assertEqual(sample_rows(logits, [params], [[0, 0]]).tolist(), [1])

    def test_negative_presence_penalty_rewards_a_seen_token_once(self):
        logits = torch.tensor([[8.0, 9.0]])
        params = SamplingParams(presence_penalty=-2.0)
        self.assertEqual(sample_rows(logits, [params], [[0, 0]]).tolist(), [0])

    def test_top_p_exact_boundary_keeps_minimal_covering_set(self):
        logits = torch.zeros(32, 2)
        params = [SamplingParams(temperature=1.0, top_p=0.5) for _ in range(32)]
        got = sample_rows(logits, params, generator=torch.Generator().manual_seed(7))
        self.assertEqual(got.tolist(), [0] * 32)

    def test_tiny_positive_top_p_keeps_the_highest_probability_token(self):
        for top_p in [1e-46, 1e-300]:
            with self.subTest(top_p=top_p):
                got = sample_rows(torch.tensor([[10.0, 9.0]]),
                                  [SamplingParams(temperature=1.0, top_p=top_p)])
                self.assertEqual(got.tolist(), [0])

    def test_seed_bounds_and_legacy_tuple_stops_remain_valid(self):
        for seed in [-(2 ** 63), 0, 2 ** 64 - 1]:
            params = SamplingParams(seed=seed, stop=("END",), stop_token_ids=(1,))
            self.assertEqual(params.stop, ["END"])
            self.assertEqual(params.stop_token_ids, [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
