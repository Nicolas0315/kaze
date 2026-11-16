"""CPU-only loader/CLI regressions: python3 tests/test_loader_cli_regressions.py."""
from __future__ import annotations

import contextlib
import io
import json
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from kaze.__main__ import main
from kaze.engine import RequestOutput
from kaze.llm import LLM
from kaze.loader import load_model


class LoaderRegressions(unittest.TestCase):
    def load_fixture(self, *, prefix="model.", head_name="lm_head.weight", tied=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        config = dict(vocab_size=3, hidden_size=2, intermediate_size=2,
                      num_hidden_layers=0, num_attention_heads=1,
                      num_key_value_heads=1, max_position_embeddings=8,
                      tie_word_embeddings=tied)
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        weights = {prefix + "embed_tokens.weight": torch.ones(3, 2),
                   prefix + "norm.weight": torch.ones(2)}
        if head_name is not None:
            weights[head_name] = torch.full((3, 2), 7.0)
        save_file(weights, str(path / "model.safetensors"))
        return load_model(str(path), kind="fp16", device="cpu", verbose=False)

    def test_prefixed_embeddings_use_top_level_output_head(self):
        model = self.load_fixture()
        torch.testing.assert_close(model.lm_head.dequantize(), torch.full((3, 2), 7.0).half())

    def test_prefixed_output_head_remains_supported(self):
        model = self.load_fixture(head_name="model.lm_head.weight")
        torch.testing.assert_close(model.lm_head.dequantize(), torch.full((3, 2), 7.0).half())

    def test_unprefixed_embeddings_and_output_head_remain_supported(self):
        model = self.load_fixture(prefix="")
        torch.testing.assert_close(model.lm_head.dequantize(), torch.full((3, 2), 7.0).half())

    def test_tied_embeddings_supply_an_omitted_output_head(self):
        model = self.load_fixture(head_name=None, tied=True)
        torch.testing.assert_close(model.lm_head.dequantize(), model.embed)

    def test_explicit_output_head_is_used_even_when_embeddings_are_tied(self):
        model = self.load_fixture(tied=True)
        torch.testing.assert_close(model.lm_head.dequantize(), torch.full((3, 2), 7.0).half())

    def test_untied_model_requires_an_output_head(self):
        with self.assertRaises(KeyError):
            self.load_fixture(head_name=None)


class CPUGenerationEngine:
    """Return one deterministic completion instead of executing GPU kernels."""
    def __init__(self):
        self.requests = queue.Queue()
        self.prompt_ids = None

    def add_request(self, prompt_ids, params):
        self.prompt_ids = list(prompt_ids)
        self.requests.put("cpu-request")
        return SimpleNamespace(request_id="cpu-request")

    def is_idle(self):
        return self.requests.empty()

    def step(self):
        request_id = self.requests.get_nowait()
        return [RequestOutput(request_id, [4], "answer", True, "length",
                              ttft_ms=1.0, tpot_ms=2.0)]


class CLIRegressions(unittest.TestCase):
    def run_generate(self, *, chat=False, template=True):
        backend = Tokenizer(WordLevel(
            {"[UNK]": 0, "user": 1, "hello": 2, "assistant": 3, "answer": 4},
            unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        if template:
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['role'] }} "
                "{{ message['content'] }} {% endfor %}"
                "{% if add_generation_prompt %}assistant{% endif %}")
        instances = []

        def init_cpu(instance, *args, **kwargs):
            instance.tokenizer = tokenizer
            instance.engine = CPUGenerationEngine()
            instance._lock = threading.Lock()
            instance._streams = {}
            instance._results = {}
            instance._prof = {"step_s": 0.0, "dispatch_s": 0.0, "steps": 0}
            instance._stop = threading.Event()
            instance._worker_error = None
            instance._thread = threading.Thread(target=instance._loop, daemon=True)
            instances.append(instance)
            instance._thread.start()

        argv = ["kaze", "generate", "--model", "cpu-fixture", "hello", "--max-tokens", "1"]
        if chat:
            argv.append("--chat")
        output = io.StringIO()
        try:
            with patch.object(LLM, "__init__", init_cpu), patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(output):
                    main()
        finally:
            for instance in instances:
                instance.shutdown()
        return output.getvalue(), instances[0].engine.prompt_ids

    def test_generate_chat_wraps_prompt_as_a_user_message(self):
        output, ids = self.run_generate(chat=True)
        self.assertIn("answer\n", output)
        self.assertEqual(ids, [1, 2, 3])

    def test_generate_plain_text_keeps_the_original_prompt(self):
        output, ids = self.run_generate()
        self.assertIn("answer\n", output)
        self.assertEqual(ids, [2])

    def test_generate_chat_without_a_template_uses_the_transcript_fallback(self):
        output, ids = self.run_generate(chat=True, template=False)
        self.assertIn("answer\n", output)
        # The fallback contains the user role, the prompt, and an assistant cue.
        self.assertIn(2, ids)
        self.assertGreater(len(ids), 1)


if __name__ == "__main__":
    unittest.main()
