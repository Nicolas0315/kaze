"""API, text decoding and lifecycle regressions: python3 tests/test_api_audit.py."""
from __future__ import annotations

import json
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from kaze.engine import RequestOutput
from kaze.llm import LLM
from kaze.sampler import SamplingParams
from kaze.server import create_app


class Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t) for t in tokens)


class ByteTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return {(): "", (1,): "A\ufffd", (1, 2): "Aあ"}[tuple(tokens)]


class ScriptedLLM:
    def __init__(self, chunks, tokenizer=None):
        self.chunks = chunks
        self.tokenizer = tokenizer or Tokenizer()
        self.params = None

    def _encode(self, prompt, chat):
        return [65]

    def stream(self, ids, params):
        self.params = params
        q = queue.Queue()
        for i, chunk in enumerate(self.chunks):
            done = i == len(self.chunks) - 1
            q.put(RequestOutput("r", chunk, "", done, "stop" if done else None))
        return q


class IdleEngine:
    def __init__(self):
        self.submitted = queue.Queue()
        self.count = 0

    def add_request(self, ids, params):
        self.count += 1
        self.submitted.put(self.count)
        return SimpleNamespace(request_id=str(self.count))

    def is_idle(self):
        return True


def make_llm(engine):
    llm = LLM.__new__(LLM)
    llm.engine = engine
    llm.tokenizer = Tokenizer()
    llm._streams = {}
    llm._results = {}
    llm._lock = threading.Lock()
    llm._stop = threading.Event()
    llm._worker_error = None
    llm._prof = {"step_s": 0.0, "dispatch_s": 0.0, "steps": 0}
    llm._thread = None
    return llm


class TextAuditTests(unittest.TestCase):
    def request(self, llm, *, chat, stream, **settings):
        with TestClient(create_app(llm)) as client:
            body = {"messages": [{"role": "user", "content": "x"}]} if chat else {"prompt": "x"}
            path = "/v1/chat/completions" if chat else "/v1/completions"
            return client.post(path, json={**body, "stream": stream, **settings})

    def response_text(self, response, *, chat, stream):
        if not stream:
            choice = response.json()["choices"][0]
            return choice["message"]["content"] if chat else choice["text"]
        parts = []
        for line in response.text.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choice = json.loads(line[6:])["choices"][0]
            parts.append(choice["delta"].get("content", "") if chat else choice["text"])
        return "".join(parts)

    def test_stop_text_is_removed_from_both_http_modes(self):
        chunks = [[ord(c) for c in part] for part in ["hello<", "EN", "D>trailing"]]
        for chat in [False, True]:
            for stream in [False, True]:
                with self.subTest(chat=chat, stream=stream):
                    response = self.request(ScriptedLLM(chunks), chat=chat, stream=stream, stop="<END>")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(self.response_text(response, chat=chat, stream=stream), "hello")

    def test_partial_unicode_is_not_emitted_before_bytes_complete(self):
        for chat in [False, True]:
            with self.subTest(chat=chat):
                llm = ScriptedLLM([[1], [2]], ByteTokenizer())
                response = self.request(llm, chat=chat, stream=True)
                self.assertEqual(self.response_text(response, chat=chat, stream=True), "Aあ")

    def test_tokenizer_cleanup_does_not_rewrite_already_streamed_spaces(self):
        from tokenizers import Tokenizer as BackendTokenizer, decoders, models
        from transformers import PreTrainedTokenizerFast
        backend = BackendTokenizer(models.WordLevel(
            {"Hello": 0, " ": 1, ",": 2, "[UNK]": 3}, unk_token="[UNK]"))
        backend.decoder = decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                                            clean_up_tokenization_spaces=True, unk_token="[UNK]")
        for stream in [False, True]:
            with self.subTest(stream=stream):
                response = self.request(ScriptedLLM([[0], [1], [2]], tokenizer),
                                        chat=False, stream=stream)
                self.assertEqual(self.response_text(response, chat=False, stream=stream), "Hello ,")

    def test_partial_stop_prefix_is_flushed_on_tokenless_completion(self):
        chunks = [[ord(c) for c in "hello<EN"], []]
        response = self.request(ScriptedLLM(chunks), chat=False, stream=True, stop="<END>")
        self.assertEqual(self.response_text(response, chat=False, stream=True), "hello<EN")

    def test_library_stream_text_matches_nonstream_decoding(self):
        source = ScriptedLLM([[1], [2]], ByteTokenizer())
        llm = make_llm(None)
        llm.tokenizer = source.tokenizer
        llm.stream = source.stream
        chunks = list(llm.stream_text([65]))
        self.assertEqual("".join(text for text, _ in chunks), "Aあ")
        self.assertTrue(chunks[-1][1])

    def test_library_stream_text_does_not_emit_stop_prefix(self):
        source = ScriptedLLM([[ord(c) for c in s] for s in ["a<", "END>"]])
        llm = make_llm(None)
        llm.stream = source.stream
        chunks = list(llm.stream_text([65], SamplingParams(stop=["<END>"])))
        self.assertEqual("".join(text for text, _ in chunks), "a")
        self.assertTrue(chunks[-1][1])

    def test_library_generate_removes_stop_text_but_keeps_token_usage(self):
        pending = queue.Queue()
        tokens = [ord(c) for c in "hello<END>"]
        def add_request(ids, params):
            pending.put([RequestOutput("r", tokens, "", True, "stop")])
            return SimpleNamespace(request_id="r")
        engine = SimpleNamespace(add_request=add_request, is_idle=pending.empty, step=pending.get_nowait)
        llm = make_llm(engine)
        llm._thread = threading.Thread(target=llm._loop, daemon=True)
        llm._thread.start()
        try:
            result = llm.generate([65], SamplingParams(stop=["<END>"]))
            self.assertEqual(result.text, "hello")
            self.assertEqual(result.tokens, tokens)
        finally:
            llm.shutdown()

    def test_completion_sampling_fields_are_not_silently_ignored(self):
        llm = ScriptedLLM([[65]])
        response = self.request(llm, chat=False, stream=False, seed=17, top_k=3,
                                repetition_penalty=1.5, presence_penalty=0.5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual((llm.params.seed, llm.params.top_k, llm.params.repetition_penalty,
                          llm.params.presence_penalty), (17, 3, 1.5, 0.5))

    def test_invalid_seed_and_presence_are_http_validation_errors(self):
        for chat in [False, True]:
            for settings in [{"seed": 2 ** 64}, {"seed": -(2 ** 63) - 1},
                             {"presence_penalty": 3}, {"presence_penalty": "NaN"},
                             {"repetition_penalty": 1e-40}, {"repetition_penalty": 1e40}]:
                with self.subTest(chat=chat, settings=settings):
                    response = self.request(ScriptedLLM([[65]]), chat=chat, stream=False, **settings)
                    self.assertEqual(response.status_code, 422)

    def test_engine_rejected_prompt_returns_422(self):
        llm = ScriptedLLM([[65]])
        def reject(ids, params):
            raise ValueError("Prompt exceeds cache capacity")
        llm.stream = reject
        response = self.request(llm, chat=False, stream=False)
        self.assertEqual(response.status_code, 422)


class LifecycleAuditTests(unittest.TestCase):
    def test_unstarted_worker_rejects_submissions_immediately(self):
        llm = make_llm(IdleEngine())
        with self.assertRaises(RuntimeError):
            llm.stream([65], SamplingParams())
        self.assertEqual(llm.engine.count, 0)

    def test_shutdown_releases_stream_and_generate_waiters(self):
        llm = make_llm(IdleEngine())
        llm._thread = threading.Thread(target=llm._loop, daemon=True)
        llm._thread.start()
        stream = llm.stream([65], SamplingParams())
        result = queue.Queue()
        def generate():
            try:
                result.put(llm.generate([65]))
            except Exception as exc:
                result.put(exc)
        caller = threading.Thread(target=generate, daemon=True)
        caller.start()
        llm.engine.submitted.get(timeout=1)
        llm.engine.submitted.get(timeout=1)
        try:
            llm.shutdown()
            caller.join(timeout=0.5)
            self.assertFalse(caller.is_alive(), "shutdown left generate waiting forever")
            self.assertIsInstance(result.get_nowait(), RuntimeError)
            out = stream.get(timeout=0.5)
            self.assertTrue(out.finished)
            self.assertTrue(out.error)
            self.assertEqual(llm._streams, {})
            self.assertEqual(llm._results, {})
        finally:
            for state in list(llm._results.values()):
                state["done"].set()
            llm.shutdown()
            caller.join(timeout=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
