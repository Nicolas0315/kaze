"""High level API: model + tokenizer + engine running in its own thread.

The engine thread owns the GPU.  Requests arrive from asyncio handlers, tokens
leave through a thread-safe queue, so an HTTP server can stream while the engine
keeps batching.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import torch

from kaze.config import ModelConfig
from kaze.engine import Engine, RequestOutput
from kaze.loader import load_model
from kaze.model import KazeModel
from kaze.sampler import SamplingParams


@dataclass
class GenResult:
    text: str
    tokens: list[int]
    ttft_ms: float | None
    tpot_ms: float | None
    finish_reason: str | None


class LLM:
    def __init__(self, model_path: str, *, kind: str = "int8", group: int = 128,
                 lm_head_kind: str | None = None, max_batch: int = 32,
                 max_batched_tokens: int = 2048, block_size: int = 16,
                 gpu_memory_utilization: float = 0.9, use_cuda_graph: bool = True,
                 enable_prefix_cache: bool = True, device: str = "cuda",
                 tokenizer_path: str | None = None, num_blocks: int | None = None,
                 speculative: bool = True, spec_k: int = 8,
                 start_loop: bool = True, verbose: bool = True,
                 cache_dtype: str = "fp16"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        t0 = time.time()
        self.model = load_model(model_path, kind=kind, group=group, lm_head_kind=lm_head_kind,
                                device=device, verbose=verbose)
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
        self._lock = threading.Lock()
        # non-streaming requests do not need a wake-up per token: the loop stores the
        # tokens and only signals when the request is done (one hand-off instead of
        # one per token, which was costing 3.4x of the engine's throughput under load)
        self._results: dict[str, dict] = {}
        # where does a served step spend its time?  step() vs the token dispatch
        self._prof = {"step_s": 0.0, "dispatch_s": 0.0, "steps": 0}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start_loop:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- thread
    def _loop(self) -> None:
        p = self._prof
        while not self._stop.is_set():
            if self.engine.is_idle():
                time.sleep(0.002)
                continue
            t0 = time.perf_counter()
            outs = self.engine.step()
            t1 = time.perf_counter()
            for out in outs:
                with self._lock:
                    q = self._streams.get(out.request_id)
                    st = self._results.get(out.request_id)
                if q is not None:
                    q.put(out)
                elif st is not None:
                    if out.tokens:
                        st["tokens"].extend(out.tokens)
                    if st["ttft"] is None and out.ttft_ms is not None:
                        st["ttft"] = out.ttft_ms
                    st["tpot"] = out.tpot_ms
                    if out.finished:
                        st["reason"] = out.finish_reason
                        st["done"].set()
            p["step_s"] += t1 - t0
            p["dispatch_s"] += time.perf_counter() - t1
            p["steps"] += 1

    def metrics(self) -> dict:
        m = self.engine.metrics()
        p = self._prof
        n = max(1, p["steps"])
        m.update(steps=p["steps"], step_ms=round(p["step_s"] / n * 1e3, 3),
                 dispatch_ms=round(p["dispatch_s"] / n * 1e3, 3),
                 loop_ms=round((p["step_s"] + p["dispatch_s"]) / n * 1e3, 3))
        return m

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------- api
    def stream(self, prompt_ids: list[int], params: SamplingParams) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        seq = self.engine.add_request(prompt_ids, params)
        with self._lock:
            self._streams[seq.request_id] = q
        return q

    def generate(self, prompt: str | list[int], params: SamplingParams | None = None,
                 chat: bool = False) -> GenResult:
        params = params or SamplingParams()
        ids = self._encode(prompt, chat)
        seq = self.engine.add_request(ids, params)
        st = {"tokens": [], "ttft": None, "tpot": None, "reason": None,
              "done": threading.Event()}
        with self._lock:
            self._results[seq.request_id] = st
        st["done"].wait()
        with self._lock:
            self._results.pop(seq.request_id, None)
        tokens = st["tokens"]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
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
        params = params or SamplingParams()
        ids = self._encode(prompt, chat)
        q = self.stream(ids, params)
        tokens: list[int] = []
        sent = 0
        while True:
            out: RequestOutput = q.get()
            if out.tokens:
                tokens.extend(out.tokens)
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                delta = text[sent:]
                sent = len(text)
                if delta:
                    yield delta, out.finished
                elif out.finished:
                    yield "", True
            if out.finished:
                break

