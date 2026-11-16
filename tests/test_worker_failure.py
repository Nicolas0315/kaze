"""CPU-only worker failure regressions: python3 tests/test_worker_failure.py."""
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
    def __call__(self, prompt, **kwargs):
        return {"input_ids": [1]}

    def apply_chat_template(self, prompt, **kwargs):
        return [1]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class FailingEngine:
    """A real-loop test double with event-controlled step failure; no GPU calls."""
    def __init__(self):
        self.submitted = queue.Queue()
        self.ready = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.count = 0
        self.first_output = False

    def add_request(self, ids, params):
        self.count += 1
        request_id = f"r-{self.count}"
        self.submitted.put(request_id)
        self.ready.set()
        return SimpleNamespace(request_id=request_id)

    def is_idle(self):
        return not self.ready.is_set()

    def step(self):
        self.calls += 1
        if self.first_output and self.calls == 1:
            return [RequestOutput("r-1", [65], "A", False)]
        self.entered.set()
        if not self.release.wait(2):
            raise AssertionError("test did not release step")
        raise RuntimeError("private-worker-diagnostic")


class WorkerFailureTests(unittest.TestCase):
    def setUp(self):
        self.llm = LLM.__new__(LLM)
        self.engine = self.llm.engine = FailingEngine()
        self.llm.tokenizer = Tokenizer()
        self.llm._lock = threading.Lock()
        self.llm._streams = {}
        self.llm._results = {}
        self.llm._worker_error = None
        self.llm._prof = {"step_s": 0.0, "dispatch_s": 0.0, "steps": 0}
        self.llm._stop = threading.Event()
        self.llm._thread = threading.Thread(target=self.llm._loop, daemon=True)
        self.llm._thread.start()
        self.addCleanup(self.cleanup_worker)

    def cleanup_worker(self):
        self.engine.release.set()
        self.llm.shutdown()
        # Also release callers when running RED against the original broken loop.
        with self.llm._lock:
            for request_id, stream in self.llm._streams.items():
                stream.put(RequestOutput(request_id, [], "", True, "stop"))
            for state in self.llm._results.values():
                state["done"].set()

    def start_call(self, call):
        result = queue.Queue()

        def run():
            try:
                result.put((None, call()))
            except BaseException as error:
                result.put((error, None))

        caller = threading.Thread(target=run, daemon=True)
        caller.start()
        return result

    def finish_call(self, result):
        try:
            return result.get(timeout=1)
        except queue.Empty:
            self.fail("caller remained blocked after worker failure")

    def fail_worker(self):
        self.assertTrue(self.engine.entered.wait(1), "worker never entered step")
        self.engine.release.set()
        self.llm._thread.join(timeout=1)
        self.assertFalse(self.llm._thread.is_alive(), "failed worker kept running")

    def submitted(self):
        try:
            return self.engine.submitted.get(timeout=1)
        except queue.Empty:
            self.fail("request was never submitted")

    def client(self):
        client = TestClient(create_app(self.llm))
        self.addCleanup(client.close)
        return client

    def test_failure_notifies_all_pending_streams_and_releases_registry(self):
        streams = [self.llm.stream([1], SamplingParams()) for _ in range(3)]
        self.fail_worker()
        for stream in streams:
            try:
                out = stream.get(timeout=0.2)
            except queue.Empty:
                self.fail("pending stream never received a failure output")
            self.assertTrue(out.finished)
            self.assertTrue(getattr(out, "error", None))
            self.assertEqual(out.tokens, [])
            self.assertIsNone(out.finish_reason)
            self.assertNotIn("private-worker-diagnostic", out.error)
        self.assertEqual(self.llm._streams, {})
        self.assertIsInstance(self.llm._worker_error, RuntimeError)
        self.assertEqual(self.engine.calls, 1)

    def test_generate_waiters_raise_and_release_results(self):
        results = [self.start_call(lambda: self.llm.generate([1])) for _ in range(2)]
        self.submitted()
        self.submitted()
        self.fail_worker()
        for result in results:
            error, value = self.finish_call(result)
            self.assertIsInstance(error, RuntimeError)
            self.assertIsNone(value)
        self.assertEqual(self.llm._results, {})
        self.assertEqual(self.engine.calls, 1)

    def test_stream_text_raises_instead_of_reporting_normal_completion(self):
        result = self.start_call(lambda: list(self.llm.stream_text([1])))
        self.submitted()
        self.fail_worker()
        error, value = self.finish_call(result)
        self.assertIsInstance(error, RuntimeError)
        self.assertIsNone(value)

    def test_new_submissions_fail_without_touching_failed_engine(self):
        self.llm.stream([1], SamplingParams())
        self.fail_worker()
        count = self.engine.count
        with self.assertRaises(RuntimeError):
            self.llm.stream([1], SamplingParams())
        error, _ = self.finish_call(self.start_call(lambda: self.llm.generate([1])))
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(self.engine.count, count)
        self.assertEqual(self.engine.calls, 1)

    def test_health_checks_live_worker_and_failure(self):
        client = self.client()
        self.assertEqual(client.get("/health").status_code, 200)
        self.llm.stream([1], SamplingParams())
        self.fail_worker()
        response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-worker-diagnostic", response.text)

    def test_health_rejects_stopped_and_unstarted_workers(self):
        client = self.client()
        self.llm.shutdown()
        self.assertEqual(client.get("/health").status_code, 503)
        self.llm._thread = None
        self.assertEqual(client.get("/health").status_code, 503)

    def test_nonstream_http_waiter_returns_503_without_diagnostics(self):
        client = self.client()
        result = self.start_call(lambda: client.post("/v1/completions", json={"prompt": "x"}))
        self.submitted()
        self.fail_worker()
        error, response = self.finish_call(result)
        self.assertIsNone(error)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-worker-diagnostic", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("choices", response.json())

    def test_chat_sse_waiter_emits_error_and_done_not_success(self):
        self.engine.first_output = True
        client = self.client()
        result = self.start_call(lambda: client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "x"}], "stream": True}))
        self.submitted()
        self.fail_worker()
        error, response = self.finish_call(result)
        self.assertIsNone(error)
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error\n", response.text)
        self.assertIn("\"error\"", response.text)
        self.assertEqual(response.text.count("data: [DONE]"), 1)
        self.assertIn('"content": "A"', response.text)
        self.assertNotIn('"finish_reason": "stop"', response.text)
        self.assertNotIn('"finish_reason": "length"', response.text)
        self.assertLess(response.text.index('"content": "A"'), response.text.index("event: error"))
        self.assertNotIn("private-worker-diagnostic", response.text)
        self.assertNotIn("Traceback", response.text)

    def test_http_rejects_failed_worker_before_streaming_starts(self):
        client = self.client()
        self.llm.stream([1], SamplingParams())
        self.fail_worker()
        count = self.engine.count
        for path, body in [("/v1/completions", {"prompt": "x"}),
                           ("/v1/chat/completions", {
                               "messages": [{"role": "user", "content": "x"}]})]:
            for stream in [False, True]:
                with self.subTest(path=path, stream=stream):
                    # Bounded in RED, where q.get would otherwise hang TestClient.
                    result = self.start_call(lambda: client.post(path, json={**body, "stream": stream}))
                    error, response = self.finish_call(result)
                    self.assertIsNone(error)
                    self.assertEqual(response.status_code, 503)
                    self.assertNotIn("private-worker-diagnostic", response.text)
                    self.assertNotIn("text/event-stream", response.headers["content-type"])
        self.assertEqual(self.engine.count, count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
