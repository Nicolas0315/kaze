"""CPU-only entry point/benchmark-runner regressions."""
from __future__ import annotations

import contextlib
import io
import importlib.util
import shutil
import json
import os
import queue
import runpy
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import run as benchmark
from bench import verify
from kaze.__main__ import main
from kaze.engine import RequestOutput
from kaze.llm import LLM


class CPUGenerationEngine:
    """CPU completion boundary; LLM generation, dispatch and shutdown stay real."""
    def __init__(self, *, fail=False, timing=True):
        self.pending = queue.Queue()
        self.fail = fail
        self.timing = timing

    def add_request(self, ids, params):
        self.pending.put("cpu-request")
        return SimpleNamespace(request_id="cpu-request")

    def is_idle(self):
        return self.pending.empty()

    def step(self):
        request_id = self.pending.get_nowait()
        if self.fail:
            raise RuntimeError("fixture compute failed")
        return [RequestOutput(request_id, [1], "answer", True, "length",
                              ttft_ms=1.0 if self.timing else None, tpot_ms=None)]


class Tokenizer:
    def __call__(self, prompt, **kwargs):
        return {"input_ids": [1]}

    def decode(self, tokens, **kwargs):
        return "answer"


class EntryAuditTests(unittest.TestCase):
    def cpu_cli(self, argv, *, fail=False, timing=True, command=None):
        instances = []

        def init_cpu(instance, *args, start_loop=True, **kwargs):
            instance.tokenizer = Tokenizer()
            instance.engine = CPUGenerationEngine(fail=fail, timing=timing)
            instance._lock = threading.Lock()
            instance._streams = {}
            instance._results = {}
            instance._prof = {"step_s": 0.0, "dispatch_s": 0.0, "steps": 0}
            instance._stop = threading.Event()
            instance._worker_error = None
            instance._thread = None
            instances.append(instance)
            self.addCleanup(instance.shutdown)
            if start_loop:
                instance._thread = threading.Thread(target=instance._loop, daemon=True)
                instance._thread.start()

        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(LLM, "__init__", init_cpu))
            stack.enter_context(patch.object(sys, "argv", ["kaze", *argv]))
            stack.enter_context(contextlib.redirect_stdout(output))
            if command is not None:
                stack.enter_context(patch("bench.run.run_bench", command))
            try:
                main()
            except RuntimeError:
                self.assertTrue(instances[0]._stop.is_set(), "CLI must shut down on errors")
                raise
        return output.getvalue(), instances[0]

    def test_single_token_generation_formats_missing_tpot(self):
        output, _ = self.cpu_cli(["generate", "--model", "fixture", "hello", "--max-tokens", "1"])
        self.assertIn("answer\n", output)
        self.assertIn("tpot n/a", output)

    def test_generation_formats_missing_ttft(self):
        output, _ = self.cpu_cli(["generate", "--model", "fixture", "hello"], timing=False)
        self.assertIn("ttft n/a", output)

    def test_generate_stops_the_real_worker_on_success(self):
        _, llm = self.cpu_cli(["generate", "--model", "fixture", "hello"])
        self.assertTrue(llm._stop.is_set())
        self.assertFalse(llm._thread.is_alive())

    def test_generate_stops_the_real_worker_on_error(self):
        with self.assertRaisesRegex(RuntimeError, "worker"):
            self.cpu_cli(["generate", "--model", "fixture", "hello"], fail=True)

    def test_bench_owns_engine_without_a_background_worker(self):
        def check_worker(llm, args):
            self.assertIsNone(llm._thread, "direct engine stepping must have one owner")
        _, llm = self.cpu_cli(["bench", "--model", "fixture"], command=check_worker)
        self.assertTrue(llm._stop.is_set())

    def test_bench_shutdown_runs_after_measurement_error(self):
        def fail(llm, args):
            raise RuntimeError("measurement failed")
        with self.assertRaisesRegex(RuntimeError, "measurement failed"):
            self.cpu_cli(["bench", "--model", "fixture"], command=fail)

    def test_invalid_numbers_fail_before_model_loading(self):
        common = [("--group", "0"), ("--group", "-8"), ("--group", "7"),
                  ("--max-batch", "0"), ("--max-batched-tokens", "-1"),
                  ("--spec-k", "0"), ("--num-blocks", "0")]
        cases = [("generate", ["hello", option, value]) for option, value in common]
        cases += [("generate", ["hello", "--max-tokens", "0"]),
                  ("generate", ["hello", "--temperature", "-1"]),
                  ("generate", ["hello", "--temperature", "nan"]),
                  ("generate", ["hello", "--temperature", "inf"]),
                  ("serve", ["--port", "0"]), ("serve", ["--port", "65536"])]
        cases += [("bench", [option, value]) for option, value in
                  [("--prompt-len", "0"), ("--gen-len", "1"),
                   ("--concurrency", "0"), ("--repeats", "0")]]
        for cmd, extra in cases:
            with self.subTest(cmd=cmd, extra=extra):
                argv = ["kaze", cmd, "--model", "fixture", *extra]
                with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                    with patch.object(LLM, "__init__", side_effect=AssertionError("loaded invalid arguments")):
                        with self.assertRaises(SystemExit) as error:
                            main()
                self.assertEqual(error.exception.code, 2)


class BenchmarkAuditTests(unittest.TestCase):
    def test_run_bench_uses_requested_arguments_and_stable_output_path(self):
        calls = {}
        def prefill(llm, lengths=(128, 512, 2048), repeats=3):
            calls["prefill"] = (lengths, repeats)
            return {"prefill_32_tok_s": 100.0}
        def decode(llm, prompt_len=64, gen_len=128, repeats=3):
            calls["decode"] = (prompt_len, gen_len, repeats)
            return {"decode_tok_s": 200.0}
        def concurrency(llm, levels=(1, 2, 4, 8, 16, 32), prompt_len=128, gen_len=64):
            calls["concurrency"] = (levels, prompt_len, gen_len)
            return {2: {"agg_tok_s": 300.0, "per_stream_tok_s": 150.0, "wall_s": 1.0}}
        llm = SimpleNamespace(model=SimpleNamespace(weight_bytes=lambda: 1024))
        args = SimpleNamespace(prompt_len=32, gen_len=8, repeats=2, concurrency=2)
        captured = {}
        real_open = open
        with tempfile.TemporaryDirectory() as directory:
            def output_file(path, *args, **kwargs):
                captured["path"] = Path(path).resolve()
                return real_open(Path(directory) / "result.json", *args, **kwargs)
            cwd = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(benchmark, "bench_prefill", prefill), \
                     patch.object(benchmark, "bench_decode", decode), \
                     patch.object(benchmark, "bench_concurrency", concurrency), \
                     patch("builtins.open", output_file), contextlib.redirect_stdout(io.StringIO()):
                    benchmark.run_bench(llm, args)
            finally:
                os.chdir(cwd)
            result = json.loads((Path(directory) / "result.json").read_text())
        self.assertEqual(calls["prefill"], ((32,), 2))
        self.assertEqual(calls["decode"], (32, 8, 2))
        self.assertEqual(calls["concurrency"], ((2,), 32, 8))
        self.assertEqual(captured["path"], Path(benchmark.__file__).resolve().with_name("results.json"))
        self.assertEqual(result["model_bytes"], 1024)

    def test_benchmark_rejects_invalid_dimensions_before_engine_use(self):
        for fn, kwargs in [(benchmark.bench_prefill, {"repeats": 0}),
                           (benchmark.bench_prefill, {"lengths": (0,)}),
                           (benchmark.bench_decode, {"gen_len": 1}),
                           (benchmark.bench_decode, {"repeats": 0}),
                           (benchmark.bench_concurrency, {"levels": (0,)})]:
            with self.subTest(fn=fn.__name__, kwargs=kwargs), self.assertRaises(ValueError):
                fn(None, **kwargs)


class VerifyAuditTests(unittest.TestCase):
    def invoke(self, response=None):
        calls = []
        def child(cmd, **kwargs):
            calls.append((cmd, kwargs))
            name = Path(cmd[1]).name
            defaults = {"test_kaze.py": "10/10 passed\n", "test_stress.py": "12/12 trials clean\n",
                        "test_smoke.py": "all ok\n", "test_server.py": "all ok\n",
                        "compare.py": 'RESULT {"decode_tok_s": 100000, "prefill_tok_s": 100000}\n',
                        "drafter_worker.py": '{"tok_s": 100000}\n'}
            rc, out = 0, defaults.get(name, "Ran 1 test in 0.001s\n\nOK\n")
            if response is not None:
                rc, out = response(name, rc, out)
            return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")
        output = io.StringIO()
        with patch.object(verify.subprocess, "run", child), contextlib.redirect_stdout(output):
            rc = verify.main()
        return rc, output.getvalue(), calls

    def test_successful_children_and_metrics_pass(self):
        rc, output, _ = self.invoke()
        self.assertEqual(rc, 0)
        self.assertIn("all ok", output)

    def test_decode_child_failure_overrides_good_metrics(self):
        rc, _, _ = self.invoke(lambda name, rc, out: (7 if name == "compare.py" else rc, out))
        self.assertEqual(rc, 1)

    def test_drafter_child_failure_overrides_good_metrics(self):
        rc, _, _ = self.invoke(lambda name, rc, out: (7 if name == "drafter_worker.py" else rc, out))
        self.assertEqual(rc, 1)

    def test_absent_success_summary_fails(self):
        for target in ["test_kaze.py", "test_stress.py", "test_smoke.py", "test_server.py"]:
            with self.subTest(target=target):
                rc, _, _ = self.invoke(lambda name, rc, out: (rc, "" if name == target else out))
                self.assertEqual(rc, 1)

    def test_failed_summary_with_zero_exit_status_fails(self):
        for target, text in [("test_kaze.py", "9/10 passed\n"),
                             ("test_stress.py", "11/12 trials clean\n"),
                             ("test_smoke.py", "1 combo(s) produced wrong text\n"),
                             ("test_server.py", "failed: stream\n")]:
            with self.subTest(target=target):
                rc, _, _ = self.invoke(lambda name, rc, out: (rc, text if name == target else out))
                self.assertEqual(rc, 1)

    def test_malformed_metrics_fail_without_aborting_remaining_checks(self):
        for text in ["{broken JSON\n", '{"tok_s": null}\n', '{"tok_s": "fast"}\n',
                     '{"tok_s": Infinity}\n']:
            with self.subTest(text=text):
                rc, _, calls = self.invoke(lambda name, rc, out: (rc, text if name == "drafter_worker.py" else out))
                self.assertEqual(rc, 1)
                self.assertEqual(sum(Path(cmd[1]).name == "drafter_worker.py" for cmd, _ in calls), 4)

    def test_metrics_are_parsed_from_result_line_not_log_braces(self):
        rc, _, _ = self.invoke(lambda name, rc, out: (rc, "config {debug}\n" + out + "finished {ok}\n"
                                                    if name == "compare.py" else out))
        self.assertEqual(rc, 0)

    def test_unittest_regressions_are_included(self):
        _, _, calls = self.invoke()
        scripts = {Path(cmd[1]).name for cmd, _ in calls}
        self.assertIn("test_entry_audit.py", scripts)
        self.assertIn("test_loader_cli_regressions.py", scripts)
        self.assertIn("test_worker_failure.py", scripts)
        self.assertIn("test_model_audit.py", scripts)

    def test_runner_paths_follow_the_module_location(self):
        with tempfile.TemporaryDirectory() as directory:
            bench_dir = Path(directory) / "relocated" / "bench"
            bench_dir.mkdir(parents=True)
            target = bench_dir / "verify.py"
            shutil.copyfile(verify.__file__, target)
            spec = importlib.util.spec_from_file_location("relocated_verify", target)
            relocated = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(relocated)
            self.assertEqual(Path(relocated.ROOT), bench_dir.parent)
            self.assertEqual(Path(relocated.MODELS), bench_dir.parent.parent / "models")

    def test_unittest_failure_or_missing_summary_fails(self):
        for status, text in [(1, "Ran 1 test in 0.001s\nFAILED (failures=1)\n"), (0, "")]:
            with self.subTest(status=status, text=text):
                rc, _, _ = self.invoke(lambda name, rc, out:
                                       (status, text) if name == "test_entry_audit.py" else (rc, out))
                self.assertEqual(rc, 1)


class ComparisonAuditTests(unittest.TestCase):
    def test_llamacpp_nonzero_child_does_not_report_throughput(self):
        from bench import compare_llamacpp
        with tempfile.NamedTemporaryFile() as model:
            result = subprocess.CompletedProcess([], 7, stdout='[ {"avg_ts": 1000} ]', stderr="failed")
            with patch.object(compare_llamacpp.subprocess, "run", return_value=result):
                out = compare_llamacpp.run(model.name)
        self.assertIsNone(out["pp_tok_s"])
        self.assertIn("pp_err", out)

    def test_llamacpp_missing_or_nonfinite_metrics_are_failures(self):
        from bench import compare_llamacpp
        for text in ["[]", '[{"avg_ts": null}]', '[{"avg_ts": Infinity}]', '[{"avg_ts": -1}]']:
            with self.subTest(text=text), tempfile.NamedTemporaryFile() as model:
                result = subprocess.CompletedProcess([], 0, stdout=text, stderr="")
                with patch.object(compare_llamacpp.subprocess, "run", return_value=result):
                    out = compare_llamacpp.run(model.name)
                self.assertIsNone(out["pp_tok_s"])
                self.assertIn("pp_err", out)

    def test_llamacpp_script_sets_failure_exit_status(self):
        from bench import compare_llamacpp
        with tempfile.NamedTemporaryFile() as model:
            result = subprocess.CompletedProcess([], 7, stdout="[]", stderr="failed")
            with patch.object(subprocess, "run", return_value=result), \
                 patch.object(sys, "argv", ["compare_llamacpp", model.name]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runpy.run_path(compare_llamacpp.__file__, run_name="__main__")
            self.assertEqual(error.exception.code, 1)

    def run_drafters(self, *, status=0, metric=None):
        output = io.StringIO()
        def child(cmd, **kwargs):
            data = {"tok_s": 1000.0, "tokens_per_step": 1.0} if metric is None else metric
            return subprocess.CompletedProcess(cmd, status, stdout=json.dumps(data), stderr="")
        code = 0
        with tempfile.TemporaryDirectory() as directory:
            real_open = open
            def output_file(path, *args, **kwargs):
                return real_open(Path(directory) / "drafters.json", *args, **kwargs)
            with patch.object(subprocess, "run", child), patch("builtins.open", output_file), \
                 patch.object(sys, "path", list(sys.path)), contextlib.redirect_stdout(output):
                try:
                    runpy.run_path(str(Path(verify.__file__).with_name("compare_drafters.py")), run_name="__main__")
                except SystemExit as error:
                    code = error.code
            wrote_results = (Path(directory) / "drafters.json").exists()
        return code, output.getvalue(), wrote_results

    def test_drafter_comparison_rejects_failed_child_with_valid_json(self):
        code, output, wrote = self.run_drafters(status=7)
        self.assertEqual(code, 1)
        self.assertIn("FAILED", output)
        self.assertFalse(wrote)

    def test_drafter_comparison_handles_missing_metrics(self):
        code, output, wrote = self.run_drafters(metric={"tok_s": None})
        self.assertEqual(code, 1)
        self.assertIn("FAILED", output)
        self.assertFalse(wrote)

    def test_complete_drafter_comparison_keeps_the_table(self):
        code, output, wrote = self.run_drafters()
        self.assertEqual(code, 0)
        self.assertIn("| Task | plain |", output)
        self.assertTrue(wrote)


class AdditionalComparisonAuditTests(unittest.TestCase):
    def test_kaze_measurement_shuts_down_after_encoding_error(self):
        from bench import compare
        instances = []
        def init_cpu(instance, *args, **kwargs):
            def fail_encode(prompt, **kwargs):
                raise RuntimeError("fixture encoding failed")
            instance.tokenizer = fail_encode
            instance._stop = threading.Event()
            instance._lock = threading.Lock()
            instance._streams = {}
            instance._results = {}
            instance._thread = None
            instances.append(instance)
            self.addCleanup(instance.shutdown)
        with patch.object(LLM, "__init__", init_cpu):
            with self.assertRaisesRegex(RuntimeError, "fixture encoding failed"):
                compare.measure_kaze("fixture")
        self.assertTrue(instances[0]._stop.is_set(), "measurement errors must release the LLM")

    def vllm_measurement(self):
        from bench import compare_vllm
        import torch
        class CPUVLLM:
            def __init__(self, **kwargs):
                self.calls = 0
            def generate(self, prompts, params):
                self.calls += 1
                if len(prompts) == 1:
                    count = {1: 1, 2: 1, 3: 5, 4: 3}[self.calls]
                    counts = [count]
                else:
                    counts = range(1, len(prompts) + 1)
                return [SimpleNamespace(prompt_token_ids=[0] * 4,
                                        outputs=[SimpleNamespace(token_ids=[1] * count)])
                        for count in counts]
        module = SimpleNamespace(LLM=CPUVLLM, SamplingParams=SimpleNamespace)
        with patch.dict(sys.modules, {"vllm": module}), patch.dict(os.environ), \
             patch.object(torch.cuda, "synchronize", return_value=None), \
             patch.object(compare_vllm.time, "perf_counter", side_effect=[0, 1, 2, 5, 6, 11, 12, 14, 15, 19]):
            return compare_vllm.main("fixture")

    def test_vllm_decode_pairs_best_timing_with_its_own_token_count(self):
        result = self.vllm_measurement()
        self.assertEqual(result["n_out"], 5)
        self.assertEqual(result["tpot_ms"], 500.0)
        self.assertEqual(result["decode_tok_s"], 2.0)

    def test_vllm_aggregate_counts_actual_outputs_not_requested_maximum(self):
        result = self.vllm_measurement()
        self.assertEqual(result["agg_tok_s_8"], 18.0)
        self.assertEqual(result["agg_tok_s_32"], 132.0)

    def test_make_table_reads_sibling_results_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            bench_dir = Path(directory) / "bench"
            bench_dir.mkdir()
            script = bench_dir / "make_table.py"
            shutil.copyfile(Path(verify.__file__).with_name("make_table.py"), script)
            row = {"model": "Qwen2.5-0.5B-Instruct", "engine": "kaze", "bytes": 1024,
                   "decode_tok_s": 500.0, "prefill_tok_s": 10000.0}
            (bench_dir / "compare_results.jsonl").write_text(json.dumps(row) + "\n")
            output = io.StringIO()
            cwd = os.getcwd()
            try:
                unrelated = Path(directory) / "unrelated"
                unrelated.mkdir()
                os.chdir(unrelated)
                with contextlib.redirect_stdout(output):
                    runpy.run_path(str(script), run_name="__main__")
            finally:
                os.chdir(cwd)
        self.assertIn("| Qwen2.5-0.5B | kaze |", output.getvalue())
        self.assertIn("500.0", output.getvalue())


class LlamaServerAuditTests(unittest.TestCase):
    def invoke(self, *, health=(200,), completion=None, post_status=200, wait_timeout=False):
        from bench import compare_llama_server
        import requests
        state = {"ready": False, "health": 0, "wait": 0, "killed": False}
        def wait(**kwargs):
            state["wait"] += 1
            if wait_timeout and state["wait"] == 1:
                raise subprocess.TimeoutExpired("fixture", 20)
            return 0
        proc = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=wait,
                               kill=lambda: state.update(killed=True))
        def response(status, data):
            def check():
                if status >= 400:
                    raise requests.HTTPError("fixture HTTP failure")
            return SimpleNamespace(status_code=status, raise_for_status=check, json=lambda: data)
        def get(*args, **kwargs):
            index = min(state["health"], len(health) - 1)
            status = health[index]
            state["health"] += 1
            state["ready"] = status == 200
            return response(status, {})
        def post(*args, **kwargs):
            self.assertTrue(state["ready"], "completion started before successful health check")
            return response(post_status, {"tokens_predicted": 4} if completion is None else completion)
        with patch.object(compare_llama_server.subprocess, "Popen", return_value=proc), \
             patch.object(compare_llama_server.requests, "get", get), \
             patch.object(compare_llama_server.requests, "post", post), \
             patch.object(compare_llama_server.time, "sleep", return_value=None):
            result = compare_llama_server.run("fixture.gguf", n=1, gen_len=4)
        return result, state

    def test_server_waits_for_successful_health_status(self):
        result, state = self.invoke(health=(503, 200))
        self.assertEqual(state["health"], 2)
        self.assertGreater(result["agg_tok_s"], 0)

    def test_server_missing_token_count_is_not_fabricated(self):
        with self.assertRaises(ValueError):
            self.invoke(completion={})

    def test_server_http_error_is_not_counted_as_tokens(self):
        import requests
        with self.assertRaises(requests.HTTPError):
            self.invoke(post_status=500)

    def test_server_killed_child_is_reaped(self):
        _, state = self.invoke(wait_timeout=True)
        self.assertTrue(state["killed"])
        self.assertEqual(state["wait"], 2)


if __name__ == "__main__":
    unittest.main()
