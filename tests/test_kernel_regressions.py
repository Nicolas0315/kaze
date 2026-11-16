"""Targeted GPU regressions: python3 tests/test_kernel_regressions.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaze.kernels.fused import rope_scatter


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RopeScatterRegressions(unittest.TestCase):
    def _check_scatter(self, quantize_keys):
        torch.manual_seed(17)
        dim = 128
        qkv = torch.randn(1, 3 * dim, device="cuda", dtype=torch.float16) * 0.4
        angles = torch.arange(dim // 2, device="cuda", dtype=torch.float32).view(1, -1) / 64
        cos, sin = angles.cos(), angles.sin()
        q = torch.zeros(1, dim, device="cuda", dtype=torch.float16)
        keys = torch.zeros(2, dim, device="cuda",
                           dtype=torch.int8 if quantize_keys else torch.float16)
        values = torch.zeros(2, dim, device="cuda", dtype=torch.int8)
        scales = torch.zeros(2, 1, dim // 32, device="cuda", dtype=torch.float16)
        key_scales = torch.zeros_like(scales) if quantize_keys else None
        rope_scatter(qkv, cos, sin, q, keys, values,
                     torch.tensor([1], device="cuda"), torch.tensor([0], device="cuda"),
                     1, 1, dim, 1, k_scale=key_scales, v_scale=scales)
        torch.cuda.synchronize()
        raw_q = qkv[:, :dim].float()
        raw_k = qkv[:, dim:2 * dim].float()
        def rotate(x):
            first, second = x.chunk(2, dim=-1)
            return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
        torch.testing.assert_close(q.float(), rotate(raw_q), atol=0.002, rtol=0.002)
        if quantize_keys:
            decoded_k = (keys[1].float().view(-1, 32) * key_scales[1, 0, :, None]).view(1, -1)
            torch.testing.assert_close(decoded_k, rotate(raw_k), atol=0.02, rtol=0)
        else:
            torch.testing.assert_close(keys[1:].float(), rotate(raw_k), atol=0.002, rtol=0.002)
        decoded_v = (values[1].float().view(-1, 32) * scales[1, 0, :, None]).view(-1)
        torch.testing.assert_close(decoded_v, qkv[0, 2 * dim:].float(), atol=0.02, rtol=0)
        self.assertTrue(torch.all(scales[1] > 0))
        self.assertTrue(torch.all(scales[0] == 0))
        self.assertTrue(torch.all(values[0] == 0))

    def test_int8_values_use_chunk_indices_not_rope_cosines(self):
        self._check_scatter(quantize_keys=False)

    def test_int8_key_and_value_chunks_compile_and_preserve_values(self):
        self._check_scatter(quantize_keys=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
