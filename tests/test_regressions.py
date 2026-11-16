"""CPU-only regressions: python3 tests/test_regressions.py."""
from __future__ import annotations

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


class ImmediateEngine:
    """Complete during submission, without loading a GPU model.

    add_request waits for dispatch to expose the registration race reliably.
    A correct registration lock delays dispatch until submission returns.
    """
    def __init__(self):
        self.ready = threading.Event()
        self.dispatched = threading.Event()
        self.processed = False

    def add_request(self, prompt_ids, params):
        self.ready.set()
        self.dispatched.wait(0.1)
        return SimpleNamespace(request_id="request-1")

    def is_idle(self):
        if self.processed:
            self.dispatched.set()
        return not self.ready.is_set()

    def step(self):
        self.ready.clear()
        self.processed = True
        return [RequestOutput("request-1", [65], "A", True, "length")]


def make_llm(engine):
    llm = LLM.__new__(LLM)
    llm.engine = engine
    llm.tokenizer = Tokenizer()
    llm._lock = threading.Lock()
    llm._streams = {}
    llm._results = {}
    llm._prof = {"step_s": 0.0, "dispatch_s": 0.0, "steps": 0}
    llm._stop = threading.Event()
    llm._worker_error = None
    llm._thread = None
    return llm


class LLMRegressions(unittest.TestCase):
    def start_llm(self):
        llm = make_llm(ImmediateEngine())
        llm._thread = threading.Thread(target=llm._loop, daemon=True)
        llm._thread.start()
        self.addCleanup(llm.shutdown)
        return llm

    def test_stream_keeps_completion_produced_during_submission(self):
        llm = self.start_llm()
        stream = llm.stream([1], SamplingParams(max_tokens=1))
        try:
            out = stream.get(timeout=1)
        except queue.Empty:
            self.fail("completion was dispatched before its stream was registered")
        self.assertEqual(out.tokens, [65])
        self.assertTrue(out.finished)

    def test_generate_keeps_completion_produced_during_submission(self):
        llm = self.start_llm()
        results = []
        caller = threading.Thread(
            target=lambda: results.append(llm.generate([1], SamplingParams(max_tokens=1))),
            daemon=True)
        caller.start()
        caller.join(timeout=1)
        if caller.is_alive():
            # Release the blocked test caller after recording the lost completion.
            with llm._lock:
                for state in llm._results.values():
                    state["done"].set()
            caller.join(timeout=1)
            self.fail("generate waited forever after losing its completion")
        self.assertEqual(results[0].text, "A")
        self.assertEqual(results[0].finish_reason, "length")
        self.assertEqual(llm._results, {})

    def test_finished_stream_is_released(self):
        llm = self.start_llm()
        stream = queue.Queue()
        llm._streams["request-1"] = stream
        llm.engine.ready.set()
        self.assertTrue(stream.get(timeout=1).finished)
        self.assertTrue(llm.engine.dispatched.wait(1))
        self.assertEqual(llm._streams, {})

    def test_stream_text_reports_tokenless_completion(self):
        llm = make_llm(None)
        stream = queue.Queue()
        stream.put(RequestOutput("r", [65], "", False))
        stream.put(RequestOutput("r", [], "", True, "stop"))
        llm.stream = lambda ids, params: stream
        self.assertEqual(list(llm.stream_text([1])), [("A", False), ("", True)])

    def test_stream_text_does_not_duplicate_final_marker(self):
        llm = make_llm(None)
        stream = queue.Queue()
        stream.put(RequestOutput("r", [65], "", True, "length"))
        llm.stream = lambda ids, params: stream
        self.assertEqual(list(llm.stream_text([1])), [("A", True)])


class ServingLLM:
    tokenizer = Tokenizer()

    def __init__(self):
        self.submitted = []

    def _encode(self, prompt, chat):
        return [1]

    def stream(self, ids, params):
        self.submitted.append(params)
        stream = queue.Queue()
        stream.put(RequestOutput("r", [65], "A", True, "length"))
        return stream


class ServerRegressions(unittest.TestCase):
    def setUp(self):
        self.llm = ServingLLM()
        self.client = TestClient(create_app(self.llm))
        self.addCleanup(self.client.close)

    def test_invalid_generation_settings_never_reach_engine(self):
        invalid = {
            "max_tokens": [0, -1],
            "temperature": [-0.1, "NaN", "Infinity"],
            "top_p": [0, -0.1, 1.1, "NaN"],
        }
        for path, base in [("/v1/completions", {"prompt": "hello"}),
                           ("/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hello"}]})]:
            for key, values in invalid.items():
                for value in values:
                    with self.subTest(path=path, key=key, value=value):
                        response = self.client.post(path, json={**base, key: value})
                        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.llm.submitted, [])

    def test_invalid_chat_settings_never_reach_engine(self):
        for setting in [{"max_completion_tokens": 0}, {"max_completion_tokens": -1},
                        {"top_k": -1}, {"repetition_penalty": 0},
                        {"repetition_penalty": -1}, {"repetition_penalty": "NaN"}]:
            with self.subTest(setting=setting):
                response = self.client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "hello"}], **setting})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.llm.submitted, [])

    def test_empty_messages_are_rejected(self):
        response = self.client.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.llm.submitted, [])

    def test_empty_prompt_list_is_rejected(self):
        response = self.client.post("/v1/completions", json={"prompt": []})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.llm.submitted, [])

    def test_prompt_batch_is_not_silently_truncated(self):
        response = self.client.post("/v1/completions", json={"prompt": ["one", "two"]})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.llm.submitted, [])

    def test_empty_tokenized_prompts_never_reach_engine(self):
        self.llm._encode = lambda prompt, chat: []
        for path, body in [("/v1/completions", {"prompt": ""}),
                           ("/v1/chat/completions",
                            {"messages": [{"role": "user", "content": ""}]})]:
            with self.subTest(path=path):
                response = self.client.post(path, json=body)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.llm.submitted, [])

    def test_single_prompt_list_stays_supported(self):
        response = self.client.post("/v1/completions", json={"prompt": ["one"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["text"], "A")

    def test_chat_completion_token_alias_and_greedy_remain_supported(self):
        response = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 3, "temperature": 0, "top_k": 0})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.llm.submitted[0].max_tokens, 3)
        self.assertEqual(self.llm.submitted[0].temperature, 0)

    def test_streaming_response_stays_supported(self):
        for path, body in [("/v1/completions", {"prompt": "hello"}),
                           ("/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hello"}]})]:
            with self.subTest(path=path):
                response = self.client.post(path, json={**body, "stream": True})
                self.assertEqual(response.status_code, 200)
                self.assertIn("data: [DONE]", response.text)
                self.assertIn("A", response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
