"""High level API: model + tokenizer + engine running in its own thread.

The engine thread owns the GPU.  Requests arrive from asyncio handlers, tokens
leave through a thread-safe queue, so an HTTP server can stream while the engine
keeps batching.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Iterator


from kaze.engine import Engine, RequestOutput
from kaze.loader import load_model
from kaze.sampler import SamplingParams
from kaze.text import decode_output


WORKER_ERROR = "Inference worker is unavailable"


@dataclass
class GenResult:
    text: str
    tokens: list[int]
    ttft_ms: float | None
    tpot_ms: float | None
    finish_reason: str | None


class LLM:
    def __init__(self, model_path: str, *, kind: str = "int8", group: int = 128,
                 lm_head_kind: str | None = None, max_batch: int = 64,
                 max_batched_tokens: int = 2048, block_size: int = 16,
                 gpu_memory_utilization: float = 0.9, use_cuda_graph: bool = True,
                 enable_prefix_cache: bool = True, device: str = "cuda",
                 tokenizer_path: str | None = None, num_blocks: int | None = None,
                 speculative: bool = True, spec_k: int = 16,
                 start_loop: bool = True, verbose: bool = True,
                 cache_dtype: str = "fp16"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        t0 = time.time()
        self.model = load_model(model_path, kind=kind, group=group, lm_head_kind=lm_head_kind,
                                device=device, verbose=verbose,
                                tokenizer_dir=tokenizer_path)
        if verbose:
            print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
        self.engine = Engine(self.model, tokenizer=self.tokenizer, block_size=block_size,
                             max_batch=max_batch, max_batched_tokens=max_batched_tokens,
                             gpu_memory_utilization=gpu_memory_utilization,
                             use_cuda_graph=use_cuda_graph,
                             enable_prefix_cache=enable_prefix_cache, num_blocks=num_blocks,
                             speculative=speculative, spec_k=spec_k, verbose=verbose,
                             cache_dtype=cache_dtype)
        self.engine.capture_graphs()
        self.engine.capture_spec_graph()
        self._streams: dict[str, queue.Queue] = {}
        # requests that only need the final output (non-streaming HTTP)
        self._quiet: set[str] = set()
        self._quiet_acc: dict[str, list[int]] = {}
        self._lock = threading.Lock()
        # non-streaming requests do not need a wake-up per token: the loop stores the
        # tokens and only signals when the request is done (one hand-off instead of
        # one per token, which was costing 3.4x of the engine's throughput under load)
        self._results: dict[str, dict] = {}
        # where does a served step spend its time?  step() vs the token dispatch
        self._prof = {"step_s": 0.0, "dispatch_s": 0.0, "steps": 0,
                      "req_n": 0, "req_wait_s": 0.0, "req_run_s": 0.0, "req_tail_s": 0.0}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_error: Exception | None = None
        if start_loop:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- thread
    def _loop(self) -> None:
        p = self._prof
        try:
            while not self._stop.is_set():
                if self.engine.is_idle():
                    time.sleep(0.002)
                    continue
                t0 = time.perf_counter()
                outs = self.engine.step()
                t1 = time.perf_counter()
                quiet = self.__dict__.get("_quiet")
                for out in outs:
                    with self._lock:
                        q = self._streams.get(out.request_id)
                        st = self._results.get(out.request_id)
                        # A "quiet" stream (non-streaming HTTP request) only needs the
                        # final output.  Waking 32 waiting threads on every token was
                        # costing 14 ms per 7B step at 32 streams: measured 841 tok/s
                        # over HTTP against 1,364 in-process with the same engine step.
                        # The tokens are accumulated here and delivered once, so the
                        # handoff cost drops from one wakeup per token to one per request.
                        # __dict__ lookups keep this safe for partially built instances
                        # (the audit fakes construct LLM with __new__).
                        if quiet and out.request_id in quiet:
                            acc = self.__dict__.setdefault(
                                "_quiet_acc", {}).setdefault(out.request_id, [])
                            if out.tokens:
                                acc.extend(out.tokens)
                            if not out.finished:
                                out = None
                            else:
                                out = replace(out, tokens=list(acc))
                                self.__dict__.get("_quiet_acc", {}).pop(out.request_id, None)
                        if out is not None and out.finished:
                            self._streams.pop(out.request_id, None)
                            if quiet:
                                quiet.discard(out.request_id)
                        if out is not None and q is not None:
                            q.put(out)
                        elif out is not None and st is not None:
                            if out.tokens:
                                st["tokens"].extend(out.tokens)
                            if st["ttft"] is None and out.ttft_ms is not None:
                                st["ttft"] = out.ttft_ms
                            st["tpot"] = out.tpot_ms
                            if out.finished:
                                st["reason"] = out.finish_reason
                                st["error"] = out.error
                                st["done"].set()
                p["step_s"] += t1 - t0
                p["dispatch_s"] += time.perf_counter() - t1
                p["steps"] += 1
        except Exception as error:
            # GPU state may be damaged: wake every caller, then leave the worker
            # failed rather than retrying step() or accepting more engine work.
            with self._lock:
                self._worker_error = error
                self._notify_pending_error()

    def _notify_pending_error(self) -> None:
        """Release all callers; the registration/dispatch lock must be held."""
        for request_id, q in self._streams.items():
            q.put(RequestOutput(request_id, [], "", True, error=WORKER_ERROR))
        self._streams.clear()
        for st in self._results.values():
            if not st["done"].is_set():
                st["error"] = WORKER_ERROR
                st["done"].set()
        self._results.clear()

    def is_healthy(self) -> bool:
        return (self._worker_error is None
                and not self._stop.is_set()
                and self._thread is not None and self._thread.is_alive())

    def _check_worker(self) -> None:
        # Called under the registration lock, also held by the failure path.
        error = self._worker_error
        if (error is not None or self._stop.is_set()
                or self._thread is None or not self._thread.is_alive()):
            raise RuntimeError(WORKER_ERROR) from error

    def metrics(self) -> dict:
        m = self.engine.metrics()
        p = self._prof
        n = max(1, p["steps"])
        m.update(steps=p["steps"], step_ms=round(p["step_s"] / n * 1e3, 3),
                 dispatch_ms=round(p["dispatch_s"] / n * 1e3, 3),
                 loop_ms=round((p["step_s"] + p["dispatch_s"]) / n * 1e3, 3),
                 # cumulative seconds, so a load test can take a clean delta instead of
                 # multiplying by a lifetime average (which mixes in warmup and prefill)
                 engine_s=round(p["step_s"] + p["dispatch_s"], 6))
        # request-level split: how long a call waited for its first token and how long the
        # tail (last token -> response) took.  Populated by the HTTP layer.
        reqs = max(1, p.get("req_n", 0))
        if p.get("req_n"):
            m.update(req_n=p["req_n"],
                     req_wait_ms=round(p["req_wait_s"] / reqs * 1e3, 3),
                     req_run_ms=round(p["req_run_s"] / reqs * 1e3, 3),
                     req_tail_ms=round(p["req_tail_s"] / reqs * 1e3, 3))
        return m

    def shutdown(self) -> None:
        with self._lock:
            self._stop.set()
            self._notify_pending_error()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------- api
    def stream(self, prompt_ids: list[int], params: SamplingParams) -> queue.Queue:
        """Register a stream and return its queue (one output per step)."""
        return self._stream(prompt_ids, params, quiet=False)

    def stream_final_only(self, prompt_ids: list[int],
                          params: SamplingParams) -> queue.Queue:
        """Stream that only delivers the final output (non-streaming HTTP requests).

        The per-token wakeups are pure overhead for a caller that waits for the end: at
        32 streams they cost 14 ms per 7B step (841 vs 1,364 tok/s with the same engine
        step).  Kept as a separate method so the plain `stream` signature stays stable
        for embedders and the audit fakes.
        """
        return self._stream(prompt_ids, params, quiet=True)

    def _stream(self, prompt_ids: list[int], params: SamplingParams,
                quiet: bool) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        # Dispatch uses this lock too: no output may arrive before registration.
        with self._lock:
            self._check_worker()
            seq = self.engine.add_request(prompt_ids, params)
            self._streams[seq.request_id] = q
            if quiet:
                self.__dict__.setdefault("_quiet", set()).add(seq.request_id)
                self.__dict__.setdefault("_quiet_acc", {})
        return q

    def generate(self, prompt: str | list[int], params: SamplingParams | None = None,
                 chat: bool = False) -> GenResult:
        params = (params or SamplingParams()).clone()
        ids = self._encode(prompt, chat)
        st = {"tokens": [], "ttft": None, "tpot": None, "reason": None, "error": None,
              "done": threading.Event()}
        with self._lock:
            self._check_worker()
            seq = self.engine.add_request(ids, params)
            self._results[seq.request_id] = st
        st["done"].wait()
        with self._lock:
            self._results.pop(seq.request_id, None)
        if st["error"]:
            raise RuntimeError(st["error"])
        tokens = st["tokens"]
        text = decode_output(self.tokenizer, tokens, params.stop)
        return GenResult(text=text, tokens=tokens, ttft_ms=st["ttft"], tpot_ms=st["tpot"],
                         finish_reason=st["reason"])

    def chat(self, messages: list[dict], params: SamplingParams | None = None) -> GenResult:
        return self.generate(messages, params, chat=True)

    def _encode(self, prompt, chat: bool) -> list[int]:
        if chat:
            try:      # Qwen3: keep the benchmark/generation out of thinking mode
                ids = self.tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, return_tensors=None,
                    enable_thinking=False)
            except (TypeError, ValueError):
                try:
                    ids = self.tokenizer.apply_chat_template(
                        prompt, add_generation_prompt=True, return_tensors=None)
                except (TypeError, ValueError):
                    # tokenizers from older .bin-only releases (Capybara) carry no chat
                    # template; fall back to a plain role-prefixed transcript instead of
                    # failing the request
                    text = "".join(
                        f"### {m['role'].capitalize()}: {m['content']}" + chr(10)
                        for m in prompt)
                    ids = self.tokenizer(text + "### Assistant:",
                                         return_tensors=None)["input_ids"]
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            return list(ids)
        if isinstance(prompt, list):
            return list(prompt)
        return list(self.tokenizer(prompt, return_tensors=None)["input_ids"])

    def stream_text(self, prompt, params: SamplingParams | None = None,
                    chat: bool = False) -> Iterator[tuple[str, bool]]:
        params = (params or SamplingParams()).clone()
        ids = self._encode(prompt, chat)
        q = self.stream(ids, params)
        tokens: list[int] = []
        sent = 0
        while True:
            out: RequestOutput = q.get()
            if out.error:
                raise RuntimeError(out.error)
            if out.tokens:
                tokens.extend(out.tokens)
            text = decode_output(self.tokenizer, tokens, params.stop, finished=out.finished)
            delta = text[sent:]
            sent = len(text)
            if delta or out.finished:
                yield delta, out.finished
            if out.finished:
                break

