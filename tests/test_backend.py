"""Backend policy and graph gating, exercised without ROCm hardware.

The ROCm policy is reachable on any box through KAZE_BACKEND=rocm (or --backend rocm),
so these tests pin the policy itself and the engine's reaction to it.

Run: python3 tests/test_backend.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaze.backend import (Backend, cache_dtype_policy, describe, detect_device,
                          real_vendor)
from kaze.config import ModelConfig
from kaze.engine import DONE, Engine
from kaze.llm import LLM
from kaze.sampler import SamplingParams


HIP_RUNTIME = getattr(torch.version, "hip", None)


@contextlib.contextmanager
def backend_env(**values):
    """Run with KAZE_BACKEND / KAZE_GRAPHS set; a None value means unset."""
    with patch.dict(os.environ, {k: v for k, v in values.items() if v is not None},
                    clear=False):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
        yield


class CPUModel:
    """Logits-only stand-in; the real engine, cache and scheduler stay in the loop."""
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = ModelConfig(vocab_size=5, hidden_size=4, intermediate_size=4,
                               num_hidden_layers=1, num_attention_heads=1,
                               num_key_value_heads=1, max_position_embeddings=128)
        self.buffers = {}

    def get_buffers(self, tdim):
        if tdim not in self.buffers:
            self.buffers[tdim] = SimpleNamespace(
                pos=torch.zeros(tdim, dtype=torch.long), logits=torch.zeros(tdim, 5))
        return self.buffers[tdim]

    def _logits(self, ids):
        out = torch.full((len(ids), 5), -100.0)
        out.scatter_(1, ((ids + 1) % 5).view(-1, 1), 10.0)
        return out

    def prefill_logits(self, ids, positions, cache, bt, start):
        return self._logits(ids[-1:])

    def decode_logits(self, ids, pos, slots, bt, lens, cache, tdim, attn_mode):
        return self._logits(ids)


def make_engine(**kwargs):
    kwargs.setdefault("backend", None)
    return Engine(CPUModel(), num_blocks=32, block_size=2, max_batch=2,
                  enable_prefix_cache=False, verbose=False, **kwargs)


def run_tokens(engine, prompt=(0, 1), max_tokens=5):
    seq = engine.add_request(list(prompt), SamplingParams(max_tokens=max_tokens,
                                                          temperature=0.0))
    while seq.state != DONE:
        engine.step()
    return list(seq.output)


CUDA_POLICY = Backend("cuda", "fake-sm86", (8, 6), True, True, "cuda test policy")
ROCM_POLICY = Backend("rocm", "fake-gfx1100", (11, 0), False, False, "rocm test policy")


class DetectionTests(unittest.TestCase):
    def test_default_policy_follows_the_bound_device(self):
        with backend_env(KAZE_BACKEND=None, KAZE_GRAPHS=None):
            backend = detect_device("cuda")
        self.assertEqual(backend.vendor, "rocm" if HIP_RUNTIME else "cuda")
        # graph support tracks the probed capability, never a hard-coded True
        self.assertEqual(backend.graphs_ok, backend.capability[0] >= 7)

    def test_rocm_override_off_means_no_graphs(self):
        with backend_env(KAZE_BACKEND="rocm", KAZE_GRAPHS=None):
            backend = detect_device("cuda")
        self.assertEqual(backend.vendor, "rocm")
        self.assertFalse(backend.graphs_ok, "ROCm must default to the eager path")
        self.assertEqual(backend.fp8_ok, backend.capability[0] >= 9,
                         "float8 KV is gfx94x-class only")
        self.assertIn("ROCm", backend.tuning_note)
        if real_vendor("cuda") != "rocm":
            self.assertIn("KAZE_BACKEND=rocm", backend.tuning_note,
                          "a forced policy must advertise itself")

    def test_cuda_override_selects_the_cuda_policy(self):
        with backend_env(KAZE_BACKEND="cuda", KAZE_GRAPHS=None):
            backend = detect_device("cuda")
        self.assertEqual(backend.vendor, "cuda")
        self.assertTrue(backend.fp8_ok, "CUDA keeps float8 KV")
        self.assertEqual(backend.graphs_ok, backend.capability[0] >= 7)
        self.assertIn("CUDA", backend.tuning_note)

    def test_cpu_override_is_graph_and_float8_free(self):
        with backend_env(KAZE_BACKEND="cpu", KAZE_GRAPHS="1"):
            backend = detect_device("cuda")
        self.assertEqual(backend.vendor, "cpu")
        self.assertEqual(backend.capability, (0, 0))
        self.assertFalse(backend.graphs_ok, "KAZE_GRAPHS cannot enable capture on CPU")
        self.assertFalse(backend.fp8_ok)

    def test_cpu_device_needs_no_override(self):
        with backend_env(KAZE_BACKEND=None, KAZE_GRAPHS=None):
            backend = detect_device("cpu")
        self.assertEqual(backend.vendor, "cpu")
        self.assertFalse(backend.graphs_ok)
        self.assertFalse(backend.fp8_ok)

    def test_graph_override_beats_the_vendor_default(self):
        with backend_env(KAZE_BACKEND="rocm", KAZE_GRAPHS="1"):
            self.assertTrue(detect_device("cuda").graphs_ok)
        with backend_env(KAZE_BACKEND="cuda", KAZE_GRAPHS="0"):
            self.assertFalse(detect_device("cuda").graphs_ok)
        with backend_env(KAZE_BACKEND="rocm", KAZE_GRAPHS="off"):
            self.assertFalse(detect_device("cuda").graphs_ok)

    def test_unknown_backend_is_rejected(self):
        with backend_env(KAZE_BACKEND="tpu"):
            with self.assertRaises(ValueError) as caught:
                detect_device("cuda")
        self.assertIn("KAZE_BACKEND", str(caught.exception))

    def test_cache_dtype_policy_downgrades_only_float8(self):
        ampere_rocm = Backend("rocm", "fake-gfx1100", (11, 0), False, False, "")
        mi300 = Backend("rocm", "fake-gfx942", (9, 4), False, True, "")
        self.assertEqual(cache_dtype_policy("fp8", ampere_rocm), "fp16")
        self.assertEqual(cache_dtype_policy("fp8v", ampere_rocm), "fp16")
        self.assertEqual(cache_dtype_policy("fp16", ampere_rocm), "fp16")
        self.assertEqual(cache_dtype_policy("fp8", mi300), "fp8")
        self.assertEqual(cache_dtype_policy("fp8v", mi300), "fp8v")

    def test_describe_reports_vendor_graphs_and_fp8(self):
        text = describe(Backend("rocm", "gfx1100", (11, 0), False, False, ""))
        self.assertIn("backend=rocm", text)
        self.assertIn("graphs=off", text)
        self.assertIn("fp8=unsupported", text)
        self.assertNotIn("sm_00", describe(Backend("cpu", "cpu", (0, 0), False, False, "")))


class EngineGatingTests(unittest.TestCase):
    def test_engine_skips_capture_when_the_backend_forbids_graphs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            engine = make_engine(backend=ROCM_POLICY, use_cuda_graph=True,
                                 speculative=True, spec_k=4)
            engine.capture_graphs()
            engine.capture_spec_graph()
        self.assertFalse(engine.use_cuda_graph)
        self.assertEqual(engine.graphs, {}, "decode graphs must not be captured")
        self.assertEqual(engine._spec_cache, {}, "spec verification must not capture either")
        lines = out.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 1, "log exactly one line about the policy")
        self.assertIn("disabled by backend policy", lines[0])
        self.assertIn("rocm", lines[0])

    def test_graph_backend_keeps_the_requested_capture_switch(self):
        engine = make_engine(backend=CUDA_POLICY, use_cuda_graph=True)
        self.assertTrue(engine.use_cuda_graph)
        self.assertEqual(engine.backend, CUDA_POLICY)

    def test_spec_capture_follows_the_operator_switch(self):
        engine = make_engine(backend=CUDA_POLICY, use_cuda_graph=False,
                             speculative=True, spec_k=4)
        engine.capture_spec_graph()
        self.assertEqual(engine._spec_cache, {},
                         "use_cuda_graph=False must not capture spec graphs")

    def test_gated_engine_matches_the_eager_path(self):
        with contextlib.redirect_stdout(io.StringIO()):
            gated = run_tokens(make_engine(backend=ROCM_POLICY, use_cuda_graph=True))
        eager = run_tokens(make_engine(backend=CUDA_POLICY, use_cuda_graph=False))
        self.assertEqual(gated, eager)
        self.assertTrue(gated, "the eager decode must still emit tokens")

    def test_engine_falls_back_from_float8_cache_with_a_warning(self):
        # catch_warnings instead of assertWarns: under unittest discovery on Python 3.14,
        # assertWarns probes __warningregistry__ on every sys.modules entry, which makes
        # transformers' lazy module import torchvision (absent here) and abort the test.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine = make_engine(backend=ROCM_POLICY, cache_dtype="fp8v",
                                 use_cuda_graph=False)
        messages = [str(item.message) for item in caught]
        self.assertEqual(engine.cache_dtype, "fp16")
        self.assertTrue(any("float8" in message for message in messages), messages)
        self.assertEqual(engine.cache.k.dtype, torch.float16)
        self.assertEqual(engine.cache.v.dtype, torch.float16)

    def test_fp16_cache_is_untouched(self):
        engine = make_engine(backend=ROCM_POLICY, cache_dtype="fp16", use_cuda_graph=False)
        self.assertEqual(engine.cache_dtype, "fp16")


class CliBackendTests(unittest.TestCase):
    def _run_cli(self, argv):
        seen = {}

        def fake_init(instance, *args, start_loop=True, **kwargs):
            seen["env"] = os.environ.get("KAZE_BACKEND")
            seen["kwargs"] = kwargs
            instance.engine = SimpleNamespace(backend=detect_device("cuda"),
                                              use_cuda_graph=False, cache_dtype="fp16")
            instance.shutdown = lambda: None

        def fake_bench(llm, args):
            seen["ran"] = True

        output = io.StringIO()
        with backend_env(KAZE_BACKEND=None, KAZE_GRAPHS=None), \
                patch.object(LLM, "__init__", fake_init), \
                patch.object(sys, "argv", ["kaze", *argv]), \
                patch("bench.run.run_bench", fake_bench), \
                contextlib.redirect_stdout(output):
            from kaze.__main__ import main
            main()
        return output.getvalue(), seen

    def test_backend_flag_sets_the_policy_and_prints_one_line(self):
        output, seen = self._run_cli(
            ["bench", "--model", "fixture", "--backend", "rocm", "--no-cuda-graph"])
        self.assertEqual(seen["env"], "rocm", "--backend must reach detect_device")
        self.assertTrue(seen["ran"])
        self.assertFalse(seen["kwargs"]["use_cuda_graph"])
        lines = [line for line in output.splitlines() if line.startswith("kaze:")]
        self.assertEqual(len(lines), 1, output)
        self.assertIn("backend=rocm", lines[0])
        self.assertIn("graphs=off", lines[0])
        self.assertIn("fp8=", lines[0])
        self.assertIn("cuda_graph=off", lines[0])

    def test_unknown_backend_flag_is_rejected_by_argparse(self):
        with backend_env(KAZE_BACKEND=None, KAZE_GRAPHS=None):
            with patch.object(sys, "argv",
                              ["kaze", "bench", "--model", "fixture", "--backend", "tpu"]):
                from kaze.__main__ import main
                with self.assertRaises(SystemExit):
                    with contextlib.redirect_stderr(io.StringIO()):
                        main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
