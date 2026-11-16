"""CPU-only weight regressions: python3 tests/test_weights_audit.py."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import save_file

from kaze.config import ModelConfig
from kaze.loader import _Reader, _shard_files, load_model
from kaze.quant import (QuantLinear, concat_rows, dequantize_int4,
                        dequantize_int8, quantize_int4, quantize_int8)


BASE = dict(vocab_size=3, hidden_size=8, intermediate_size=8,
            num_hidden_layers=0, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=8)


class ConfigAudit(unittest.TestCase):
    def load_config(self, **updates):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(BASE | updates), encoding="utf-8")
            return ModelConfig.from_pretrained(str(path))

    def test_unknown_activation_is_not_silently_silu(self):
        with self.assertRaisesRegex(ValueError, "hidden_act"):
            ModelConfig(**(BASE | dict(hidden_act="gelu")))

    def test_invalid_dimensions_fail_at_configuration_boundary(self):
        for changes in (dict(num_attention_heads=0), dict(num_key_value_heads=0),
                        dict(num_key_value_heads=3), dict(head_dim=3),
                        dict(hidden_size=9), dict(max_position_embeddings=0),
                        dict(num_hidden_layers=-1), dict(hidden_size=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ModelConfig(**(BASE | changes))

    def test_non_power_of_two_heads_fail_before_triton_compilation(self):
        for head_dim in (6, 96):
            with self.subTest(head_dim=head_dim), self.assertRaisesRegex(ValueError, "head_dim"):
                ModelConfig(**(BASE | dict(head_dim=head_dim)))

    def test_explicit_nonstandard_head_dimension_remains_supported(self):
        cfg = ModelConfig(**(BASE | dict(head_dim=16)))
        self.assertEqual(cfg.q_size, 32)

    def test_nullable_architectures_from_transformers_remain_supported(self):
        self.assertEqual(self.load_config(architectures=None).architectures, [])

    def test_local_expert_count_alias_is_honored(self):
        cfg = self.load_config(num_local_experts=8, num_experts_per_tok=2)
        self.assertEqual(cfg.num_experts, 8)

    def test_mixtral_derives_topk_normalization_when_omitted(self):
        for identity in (dict(model_type="mixtral"), dict(architectures=["MixtralForCausalLM"])):
            with self.subTest(identity=identity):
                cfg = self.load_config(**identity, num_local_experts=8, num_experts_per_tok=2)
                self.assertTrue(cfg.norm_topk_prob)

    def test_explicit_topk_normalization_and_other_models_are_preserved(self):
        for architecture in ("MixtralForCausalLM", "Qwen3MoeForCausalLM"):
            for enabled in (False, True):
                with self.subTest(architecture=architecture, enabled=enabled):
                    cfg = self.load_config(architectures=[architecture], num_local_experts=8,
                                           num_experts_per_tok=2, norm_topk_prob=enabled)
                    self.assertEqual(cfg.norm_topk_prob, enabled)
        cfg = self.load_config(architectures=["Qwen3MoeForCausalLM"], num_local_experts=8,
                               num_experts_per_tok=2)
        self.assertFalse(cfg.norm_topk_prob)

    def test_invalid_expert_routing_counts_fail_early(self):
        for changes in (dict(num_experts=2, num_experts_per_tok=3),
                        dict(num_experts=0, num_experts_per_tok=1)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load_config(**changes)

    def test_nested_or_partial_rope_is_not_silently_full_default(self):
        for changes in (dict(rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 1e6}}),
                        dict(partial_rotary_factor=0.5),
                        dict(rope_parameters={"rope_type": "default", "rope_theta": 0})):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "rope|rotary"):
                self.load_config(**changes)

    def test_unknown_architecture_is_not_loaded_as_llama(self):
        with self.assertRaisesRegex(ValueError, "architect"):
            self.load_config(architectures=["GemmaForCausalLM"])

    def test_supported_dense_architectures_remain_supported(self):
        for name in ("LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "MistralForCausalLM"):
            with self.subTest(name=name):
                self.assertEqual(self.load_config(architectures=[name]).architectures, [name])

    def test_dynamic_ntk_is_not_silently_approximated_as_linear(self):
        with self.assertRaisesRegex(ValueError, "rope"):
            self.load_config(rope_scaling=dict(type="dynamic", factor=2))

    def test_unknown_rope_type_is_not_silently_default(self):
        with self.assertRaisesRegex(ValueError, "rope"):
            self.load_config(rope_scaling=dict(type="longrope", factor=2))

    def test_invalid_rope_and_normalization_values_fail_early(self):
        for changes in (dict(rope_theta=0), dict(rope_theta=float("nan")),
                        dict(rms_norm_eps=0), dict(rope_scaling=dict(type="linear", factor=0))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load_config(**changes)

    def test_new_rope_parameters_override_stale_legacy_settings(self):
        cfg = self.load_config(rope_scaling=dict(type="linear", factor=2),
                               rope_parameters=dict(rope_type="linear", factor=4, rope_theta=1000000))
        self.assertEqual(cfg.rope_scaling["factor"], 4)
        self.assertEqual(cfg.rope_theta, 1000000)

    def test_ignored_mlp_bias_and_shared_experts_fail_early(self):
        for changes in (dict(mlp_bias=True), dict(shared_expert_intermediate_size=8)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load_config(**changes)


class QuantAudit(unittest.TestCase):
    def test_large_weight_validation_bounds_real_temporary_allocations(self):
        real_isfinite = torch.isfinite
        real_abs = torch.Tensor.abs
        max_elements = 4 * 1024 * 1024
        inspected = []

        def bounded_isfinite(tensor, *args, **kwargs):
            if tensor.numel() > max_elements:
                raise torch.OutOfMemoryError("simulated oversized isfinite allocation")
            inspected.append(tensor.numel())
            return real_isfinite(tensor, *args, **kwargs)

        def bounded_abs(tensor, *args, **kwargs):
            if tensor.numel() > max_elements:
                raise torch.OutOfMemoryError("simulated oversized abs allocation")
            return real_abs(tensor, *args, **kwargs)

        weight = torch.ones(4097, 1024)
        with patch.object(torch, "isfinite", bounded_isfinite), patch.object(torch.Tensor, "abs", bounded_abs):
            layer = QuantLinear.from_float(weight, None, kind="fp16")
            self.assertEqual(tuple(layer.packed.shape), (4097, 1024))
            self.assertEqual(layer.packed[-1, -1].item(), 1.)
            weight[-1, -1] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                QuantLinear.from_float(weight, None, kind="fp16")
            weight[-1, -1] = 1e10
            with self.assertRaisesRegex(ValueError, "fp16"):
                QuantLinear.from_float(weight, None, kind="fp16")
        self.assertTrue(inspected)
        self.assertLessEqual(max(inspected), max_elements)

    def test_quantization_rejects_invalid_group_and_shape(self):
        for fn in (quantize_int4, quantize_int8):
            for shape, group in (((2, 8), 0), ((2, 8), -8), ((2, 8), 3),
                                 ((0, 8), 8), ((2, 0), 8), ((8,), 8)):
                with self.subTest(fn=fn.__name__, shape=shape, group=group), self.assertRaises(ValueError):
                    fn(torch.ones(shape), group)

    def test_nonfinite_weights_are_not_silently_integer_cast(self):
        for fn in (quantize_int4, quantize_int8):
            for value in (float("nan"), float("inf")):
                with self.subTest(fn=fn.__name__, value=value), self.assertRaises(ValueError):
                    fn(torch.full((1, 8), value), 8)

    def test_stored_scales_do_not_underflow_to_zero(self):
        for quantize, dequantize in ((quantize_int4, dequantize_int4), (quantize_int8, dequantize_int8)):
            with self.subTest(fn=quantize.__name__):
                # Negative values avoid int4's intentional +7 versus -8 clipping.
                source = torch.full((1, 8), -1e-6)
                packed, scales = quantize(source, 8)
                self.assertTrue(torch.all(scales > 0).item())
                torch.testing.assert_close(dequantize(packed, scales, 8, torch.float32), source,
                                           atol=1e-7, rtol=0.05)

    def test_scales_that_overflow_fp16_raise_instead_of_producing_inf(self):
        for fn in (quantize_int4, quantize_int8):
            with self.subTest(fn=fn.__name__), self.assertRaises(ValueError):
                fn(torch.full((1, 8), 1e10), 8)

    def test_dense_fallback_rejects_fp16_overflow(self):
        with self.assertRaisesRegex(ValueError, "fp16"):
            QuantLinear.from_float(torch.full((2, 9), 1e10), None, group=8)

    def test_float_linear_rejects_fp16_bias_overflow(self):
        with self.assertRaisesRegex(ValueError, "bias"):
            QuantLinear.from_float(torch.ones(2, 8), torch.full((2,), 1e10), group=8)

    def test_float_linear_rejects_typo_in_kind(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            QuantLinear.from_float(torch.ones(2, 8), None, kind="int44", group=8)

    def test_float_linear_rejects_invalid_chunk_bias_and_group(self):
        for kwargs, bias in ((dict(chunk=0), None), (dict(chunk=-1), None),
                             (dict(group=0), None), ({}, torch.ones(3))):
            with self.subTest(kwargs=kwargs, bias=bias), self.assertRaises(ValueError):
                QuantLinear.from_float(torch.ones(2, 128), bias, **kwargs)

    def test_nondivisible_input_width_retains_dense_fallback(self):
        layer = QuantLinear.from_float(torch.ones(2, 9), None, group=8)
        self.assertEqual(layer.kind, "fp16")
        torch.testing.assert_close(layer.dequantize(), torch.ones(2, 9).half())

    def test_dequantization_rejects_broadcast_scale_shape(self):
        for fn, packed in ((dequantize_int4, torch.zeros((2, 2), dtype=torch.int32)),
                           (dequantize_int8, torch.zeros((2, 16), dtype=torch.int8))):
            with self.subTest(fn=fn.__name__), self.assertRaises(ValueError):
                fn(packed, torch.ones(1, 2), 8)

    def test_concat_rejects_mismatched_quantization_groups(self):
        layers = [QuantLinear.from_float(torch.ones(2, 16), None, group=g) for g in (8, 16)]
        with self.assertRaisesRegex(ValueError, "group"):
            concat_rows(layers)

    def test_chunked_quantization_matches_single_chunk(self):
        weight = torch.arange(40, dtype=torch.float32).reshape(5, 8) / 10
        for kind in ("int4", "int8", "fp16"):
            with self.subTest(kind=kind):
                chunked = QuantLinear.from_float(weight, None, kind=kind, group=8, chunk=2)
                whole = QuantLinear.from_float(weight, None, kind=kind, group=8)
                torch.testing.assert_close(chunked.dequantize(), whole.dequantize())


class LoaderAudit(unittest.TestCase):
    def directory(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def fixture(self, path, *, moe=False, mlp_bias=False):
        cfg = BASE | dict(num_hidden_layers=1)
        weights = {"model.embed_tokens.weight": torch.ones(3, 8),
                   "model.norm.weight": torch.ones(8), "lm_head.weight": torch.ones(3, 8)}
        prefix = "model.layers.0."
        weights.update({prefix + name: torch.ones(shape) for name, shape in {
            "input_layernorm.weight": (8,), "post_attention_layernorm.weight": (8,),
            "self_attn.q_proj.weight": (8, 8), "self_attn.k_proj.weight": (4, 8),
            "self_attn.v_proj.weight": (4, 8), "self_attn.o_proj.weight": (8, 8)}.items()})
        if moe:
            cfg.update(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=8)
            weights[prefix + "mlp.router.weight"] = torch.ones(2, 8)
            for expert in range(2):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    weights[prefix + f"mlp.experts.{expert}.{name}.weight"] = torch.full((8, 8), float(expert + 1))
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                weights[prefix + f"mlp.{name}.weight"] = torch.ones(8, 8)
        (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        return weights

    def test_per_expert_weights_load_without_missing_stack_helper(self):
        path = self.directory()
        save_file(self.fixture(path, moe=True), str(path / "model.safetensors"))
        model = load_model(str(path), kind="fp16", device="cpu", verbose=False)
        self.assertEqual(model.layers[0].w_gu_e.n, 32)
        self.assertEqual(model.layers[0].w_down_e.n, 16)
        torch.testing.assert_close(model.layers[0].w_down_e.to_dense()[8:], torch.full((8, 8), 2.).half())

    def test_mixed_per_expert_weights_use_int4_for_both_gate_and_up(self):
        path = self.directory()
        save_file(self.fixture(path, moe=True), str(path / "model.safetensors"))
        model = load_model(str(path), kind="mixed", group=8, device="cpu", verbose=False)
        layer = model.layers[0]
        self.assertEqual(layer.wqkv.kind, "int8")
        self.assertEqual(layer.w_gu_e.kind, "int4")
        self.assertEqual(layer.w_down_e.kind, "int4")
        self.assertEqual(tuple(layer.w_gu_e.dequantize().shape), (32, 8))
        self.assertEqual(tuple(layer.w_down_e.dequantize().shape), (16, 8))

    def test_wrong_weight_shape_is_rejected_before_model_construction(self):
        path = self.directory()
        weights = self.fixture(path)
        weights["model.layers.0.self_attn.k_proj.weight"] = torch.ones(8, 8)
        save_file(weights, str(path / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "k_proj"):
            load_model(str(path), kind="fp16", device="cpu", verbose=False)

    def test_nonfinite_embedding_is_rejected_before_model_construction(self):
        path = self.directory()
        weights = self.fixture(path)
        weights["model.embed_tokens.weight"][0, 0] = float("nan")
        save_file(weights, str(path / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "finite"):
            load_model(str(path), kind="fp16", device="cpu", verbose=False)

    def test_router_expert_count_must_match_configuration(self):
        path = self.directory()
        weights = self.fixture(path, moe=True)
        weights["model.layers.0.mlp.router.weight"] = torch.ones(3, 8)
        save_file(weights, str(path / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "router"):
            load_model(str(path), kind="fp16", device="cpu", verbose=False)

    def test_qk_norm_weights_must_be_a_pair(self):
        path = self.directory()
        weights = self.fixture(path)
        weights["model.layers.0.self_attn.q_norm.weight"] = torch.ones(4)
        save_file(weights, str(path / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "norm"):
            load_model(str(path), kind="fp16", device="cpu", verbose=False)

    def test_unconfigured_mlp_bias_is_not_ignored(self):
        path = self.directory()
        weights = self.fixture(path)
        weights["model.layers.0.mlp.down_proj.bias"] = torch.ones(8)
        save_file(weights, str(path / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "bias"):
            load_model(str(path), kind="fp16", device="cpu", verbose=False)

    def test_bin_tensor_mapping_loads_real_cpu_model(self):
        path = self.directory()
        torch.save(self.fixture(path), path / "pytorch_model.bin")
        model = load_model(str(path), kind="int8", group=8, device="cpu", verbose=False)
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.layers[0].wqkv.n, 16)

    def test_bin_non_tensor_weight_is_rejected(self):
        path = self.directory()
        torch.save({"bad.weight": [1, 2]}, path / "pytorch_model.bin")
        with self.assertRaisesRegex(ValueError, "tensor"):
            _Reader(str(path))

    def test_safetensors_take_precedence_over_legacy_bin_index(self):
        path = self.directory()
        save_file({"weight": torch.ones(2)}, str(path / "model.safetensors"))
        (path / "pytorch_model.bin.index.json").write_text(
            json.dumps({"weight_map": {"weight": "old.bin"}}), encoding="utf-8")
        self.assertEqual(_shard_files(str(path)), [str(path / "model.safetensors")])

    def test_index_path_cannot_escape_model_directory(self):
        path = self.directory()
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "../outside.safetensors"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shard"):
            _shard_files(str(path))

    def test_duplicate_tensor_across_shards_is_rejected(self):
        path = self.directory()
        for index in range(2):
            save_file({"weight": torch.ones(2)}, str(path / f"part-{index}.safetensors"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _Reader(str(path))

    def test_safetensors_header_bounds_are_validated_at_open(self):
        path = self.directory()
        header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}).encode()
        (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
        with self.assertRaises((ValueError, IOError)):
            _Reader(str(path))

    def test_valid_empty_and_scalar_safetensors_are_readable(self):
        path = self.directory()
        save_file({"empty": torch.empty(0, 2), "scalar": torch.tensor(3.)}, str(path / "model.safetensors"))
        reader = _Reader(str(path))
        self.assertEqual(tuple(reader.get("empty").shape), (0, 2))
        self.assertEqual(reader.get("scalar").item(), 3.)


if __name__ == "__main__":
    unittest.main()
