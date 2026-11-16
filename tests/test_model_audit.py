"""Kernel regressions: CPU by default; KAZE_AUDIT_DEVICE=cuda opts into GPU probes."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ModelAuditRegressions(unittest.TestCase):
    def run_cpu(self, code):
        device = os.environ.get("KAZE_AUDIT_DEVICE", "cpu")
        self.assertIn(device, ("cpu", "cuda"))
        env = dict(os.environ, TRITON_INTERPRET="1" if device == "cpu" else "0",
                   PYTHONDONTWRITEBYTECODE="1")
        code = f"import torch; torch.set_default_device({device!r})\n" + code
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def check_int4_single_row_zero_weight(self, split_k):
        self.run_cpu(f"""
import torch
from kaze.quant import QuantLinear
from kaze.kernels.quant_linear import gemv
n, k = 3, 256
packed = torch.full((n, k // 2), -120, dtype=torch.int8).view(torch.int32)
scales = torch.ones((n, k // 128), dtype=torch.float16)
lin = QuantLinear(packed, scales, None, n, k, 128, 'int4')
x = torch.ones((1, k), dtype=torch.float16)
out = gemv(x, lin, 1, split_k={split_k}, block_n=32, block_k=128)
assert torch.count_nonzero(out) == 0, out.tolist()
""")

    def test_int4_single_row_high_nibbles_subtract_zero_point(self):
        self.check_int4_single_row_zero_weight(split_k=1)

    def test_int4_single_row_splitk_high_nibbles_subtract_zero_point(self):
        self.check_int4_single_row_zero_weight(split_k=2)

    def test_int4_single_row_swiglu_high_nibbles_subtract_zero_point(self):
        self.run_cpu("""
import torch
from kaze.kernels.quant_linear import gemv_int4_swiglu_kernel
n, k = 3, 256
packed = torch.full((2 * n, k // 2), -120, dtype=torch.int8)
scales = torch.ones((2 * n, k // 128), dtype=torch.float16)
x = torch.ones((1, k), dtype=torch.float16)
out = torch.empty((1, n), dtype=torch.float16)
gemv_int4_swiglu_kernel[(1,)](x, packed, scales, out, n, k,
    BLOCK_N=32, BLOCK_K=128, GROUP=128, TDIM=1)
assert torch.count_nonzero(out) == 0, out.tolist()
""")

    def test_quantized_rmsnorm_preserves_scale_storage_guard(self):
        self.run_cpu("""
import torch
from kaze.kernels.norm import rmsnorm
for n in (128, 1536):
    x = torch.ones((2, n), dtype=torch.float16)
    guard = torch.full((2 * (n // 128) + 32,), -99., dtype=torch.float32)
    scales = guard[:2 * (n // 128)].view(2, n // 128)
    q = torch.empty_like(x, dtype=torch.int8)
    rmsnorm(x, torch.ones(n, dtype=x.dtype), 1e-6, out_q=q, out_s=scales)
    assert torch.all(guard[scales.numel():] == -99), (n, guard.tolist())
    torch.testing.assert_close(scales, torch.full_like(scales, 1 / 127), atol=1e-7, rtol=1e-5)
""")

    def test_fp16_fallback_masks_partial_input_tile(self):
        self.run_cpu("""
import torch
from kaze.quant import QuantLinear
from kaze.kernels.quant_linear import gemv
for k, expected in ((192, 192), (96, 96)):
    weight_storage = torch.ones(k + 128, dtype=torch.float16)
    lin = QuantLinear.from_float(weight_storage[:k].view(1, k), None, 'int4', group=128)
    assert lin.kind == 'fp16'
    x_storage = torch.ones(16 * k + 128, dtype=torch.float16)
    x = x_storage[:16 * k].view(16, k)
    out = gemv(x, lin, 16)
    assert torch.all(out == expected), out.flatten().tolist()
""")

    def test_int4_integer_prefill_masks_output_column_tail(self):
        self.run_cpu("""
import torch
from kaze.kernels.quant_linear import gemm_int4_w8a8_kernel, gemm_swiglu_w8a8_kernel
m, n, k = 64, 65, 128
aq = torch.ones((m, k), dtype=torch.int8)
scales_a = torch.ones((m, 1), dtype=torch.float32)
for swiglu in (False, True):
    rows = 2 * n if swiglu else n
    packed = torch.full((rows, k // 2), -120, dtype=torch.int8)
    scales_w = torch.ones((rows, 1), dtype=torch.float16)
    guard = torch.full((m * n + 64,), -99., dtype=torch.float16)
    out = guard[:m * n].view(m, n)
    kw = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP=128, XGROUP=128)
    if swiglu:
        # YQ / YS are the fused int8-output pointers (unused with HAS_OUTQ=False)
        gemm_swiglu_w8a8_kernel[(1, 2)](aq, scales_a, packed, scales_w, out, out, out,
                                        m, n, k, **kw)
    else:
        gemm_int4_w8a8_kernel[(1, 2)](aq, scales_a, packed, scales_w, out, out,
                                    m, n, k, HAS_BIAS=False, **kw)
    assert torch.all(guard[m * n:] == -99), (swiglu, guard[m * n:].tolist())
    assert torch.count_nonzero(out) == 0
""")

    def test_moe_router_load_mask_covers_expert_and_hidden_tails(self):
        self.run_cpu("""
import ast
import inspect
import torch
from kaze.kernels.quant_linear import moe_router_kernel
source = ast.parse(inspect.getsource(moe_router_kernel.fn))
for node in ast.walk(source):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'load':
        if not any(isinstance(name, ast.Name) and name.id == 'R' for name in ast.walk(node.args[0])):
            continue
        mask = next(kw.value for kw in node.keywords if kw.arg == 'mask')
        context = dict(mask_e=torch.arange(4) < 3, h0=128, offs_h=torch.arange(128), H=129)
        actual = eval(compile(ast.Expression(mask), '<router-mask>', 'eval'), context)
        expected = torch.zeros((4, 128), dtype=torch.bool)
        expected[:3, 0] = True
        assert torch.equal(actual, expected), actual.sum().item()
        break
else:
    raise AssertionError('router weight load not found')
""")

    def test_short_attention_supports_non_power_of_two_gqa(self):
        self.run_cpu("""
import torch
from kaze.kernels.attn import attn_decode
q = torch.zeros((1, 24, 64), dtype=torch.float16)
k = torch.zeros((1, 16, 1, 64), dtype=torch.float16)
out = torch.full_like(q, -123)
attn_decode(q, k, torch.ones_like(k), torch.zeros((1, 1), dtype=torch.int32),
            torch.ones(1, dtype=torch.int32), 16, 24, 64, out, mode='short')
assert torch.all(out == 1)
""")

    def test_long_attention_writes_every_head_in_large_gqa_group(self):
        self.run_cpu("""
import torch
from kaze.config import ModelConfig
from kaze.model import DecodeBuffers
from kaze.kernels.attn import attn_decode
cfg = ModelConfig(vocab_size=128, hidden_size=2048, intermediate_size=128,
                  num_hidden_layers=1, num_attention_heads=32,
                  num_key_value_heads=1, head_dim=64)
bufs = DecodeBuffers(cfg, torch.get_default_device(), 16)
q = torch.zeros((1, 32, 64), dtype=torch.float16)
k = torch.zeros((1, 16, 1, 64), dtype=torch.float16)
v = torch.ones_like(k)
out = torch.full_like(q, -123)
attn_decode(q, k, v, torch.zeros((1, 1), dtype=torch.int32),
            torch.ones(1, dtype=torch.int32), 16, 32, 64, out,
            bufs.acc[:1], bufs.ml[:1], mode='long')
assert torch.all(out == 1), out[0, :, 0].tolist()
""")

    def test_moe_router_normalizes_probabilities_not_signed_logits(self):
        self.run_cpu("""
import torch
from kaze.kernels.quant_linear import moe_router
x = torch.zeros((1, 128), dtype=torch.float16)
x[0, 0] = 1
router = torch.zeros((2, 128), dtype=torch.float16)
router[:, 0] = torch.tensor([1., -1.])
for topk, normalize, expected in ((2, True, [0.88079708, 0.11920292]),
                                  (1, False, [0.88079708])):
    indices = torch.empty(topk, dtype=torch.int32)
    weights = torch.empty(topk, dtype=torch.float32)
    moe_router(x, router, topk, 2, normalize, indices, weights)
    torch.testing.assert_close(weights, torch.tensor(expected), atol=1e-6, rtol=1e-6)
""")

    def test_moe_prefill_zero_experts_remain_finite(self):
        self.run_cpu("""
import torch
from types import SimpleNamespace
from kaze.model import KazeModel
from kaze.quant import QuantLinear
model = object.__new__(KazeModel)
model.cfg = SimpleNamespace(num_experts_per_tok=2, num_experts=2,
                            norm_topk_prob=True, hidden_size=128)
x = torch.zeros((1, 128), dtype=torch.float16)
x[0, 0] = 1
router = torch.zeros((2, 128), dtype=torch.float16)
router[:, 0] = torch.tensor([1., -1.])
def weights(n):
    return QuantLinear.from_float(torch.zeros((n, 128), dtype=torch.float16), None, 'fp16')
layer = SimpleNamespace(router=router, moe_i=128, w_gu_e=weights(512), w_down_e=weights(256))
hidden = torch.zeros_like(x)
model._moe_prefill(layer, x, hidden)
assert torch.equal(hidden, torch.zeros_like(x)), hidden.tolist()
""")


if __name__ == "__main__":
    unittest.main(verbosity=2)
