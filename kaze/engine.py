"""Continuous-batching engine: scheduler, paged KV, CUDA-graph decode.

One step is one model pass:

  decode  - every running sequence contributes exactly one token; the batch is
            padded to a fixed bucket size so the whole pass replays from a CUDA
            graph (no Python, no launch overhead).
  prefill - a chunk of prompt tokens for waiting sequences, budgeted by
            max_batched_tokens so a long prompt cannot starve decoding.

Memory is managed in blocks; when the cache is full the scheduler preempts the
newest decoding sequence (its blocks are freed, its tokens are recomputed later,
and any still-cached prefix is reused by the prefix cache).
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch

from kaze.cache import KVCache, blocks_for
from kaze.config import ModelConfig
from kaze.model import KazeModel
from kaze.sampler import SamplingParams, sample_rows

WAITING, PREFILL, DECODE, DONE = 0, 1, 2, 3


@dataclass
class RequestOutput:
    request_id: str
    tokens: list[int]
    text: str
    finished: bool
    finish_reason: str | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None


LOOKUP_MIN = 2
LOOKUP_MAX = 6
DRAFT_FANOUT = 4


class Sequence:
    _counter = 0

    def __init__(self, prompt: list[int], params: SamplingParams, request_id: str | None = None):
        Sequence._counter += 1
        self.seq_id = Sequence._counter
        self.request_id = request_id or f"cmpl-{uuid.uuid4().hex[:12]}"
        self.prompt = list(prompt)
        self.output: list[int] = []
        self._all: list[int] = list(self.prompt)
        self.params = params
        self.blocks: list[int] = []
        self.hashes: list[int] = []
        self.num_computed = 0
        self.state = WAITING
        self.arrival = time.perf_counter()
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.itls: list[float] = []
        self.finish_reason: str | None = None
        self.text_so_far = ""
        self.bt: torch.Tensor | None = None
        # draft-model speculation: the drafter keeps its own KV cache, so we remember
        # how many positions of this sequence are already in it (positions beyond it are
        # recomputed because a rejected draft leaves stale entries there)
        self.draft_blocks: list[int] | None = None
        self.draft_len: int = 0
        # suffix index for prompt-lookup drafting: one table per n-gram order, so a
        # long repeat (JSONL lines, code blocks) can be matched with its full context
        self.grams: list[tuple[int, dict]] = [(o, {}) for o in range(LOOKUP_MIN, LOOKUP_MAX + 1)]


    def _index(self, i: int) -> None:
        """Record every suffix ending at i, keyed by its start position."""
        toks = self.tokens
        for order, table in self.grams:
            if i >= order:
                bucket = table.get(tuple(toks[i - order:i]))
                if bucket is None:
                    table[tuple(toks[i - order:i])] = [i - order]
                else:
                    bucket.append(i - order)
                    if len(bucket) > DRAFT_FANOUT:
                        bucket.pop(0)

    def build_index(self) -> None:
        for i in range(LOOKUP_MIN, len(self.tokens)):
            self._index(i)

    def draft(self, k: int) -> list[int]:
        """Prompt-lookup draft: copy the continuation of the longest matching suffix.

        Trying the longest suffix first is what makes edit-style workloads (JSONL,
        code blocks, repeated schemas) accept many tokens per verification: the
        longer the context that matches, the more reliable the copy is.
        """
        toks = self.tokens
        n = len(toks)
        for order, table in reversed(self.grams):
            if n < order + 1:
                continue
            bucket = table.get(tuple(toks[-order:]))
            if not bucket:
                continue
            start = bucket[-1] + order
            if start >= n:
                continue
            return toks[start:min(start + k, n)]
        return []

    def draft_with_counter(self, k: int, tok_int, int_token, nl_tok) -> list[int]:
        """Copy draft whose first field is advanced by the record-to-record step.

        The counter is taken from the field immediately after the record boundary
        (a newline), not from "the last integer anywhere": in JSONL rows the note
        field also holds numbers, and picking those made the draft useless.
        """
        toks = self.tokens
        size = len(toks)
        if size < 8:
            return []
        # record boundary: where does the current record start?
        cur_start = None
        for i in range(size - 1, max(0, size - 64) - 1, -1):
            if toks[i] == nl_tok:
                cur_start = i + 1
                break
        if cur_start is None:
            return self.draft(k)
        for order, table in reversed(self.grams):
            if size < order + 1:
                continue
            bucket = table.get(tuple(toks[-order:]))
            if not bucket:
                continue
            start = bucket[-1] + order
            if start >= size:
                continue
            # the same field of the previous record
            src_start = None
            for i in range(min(start, size) - 1, max(0, start - 64) - 1, -1):
                if toks[i] == nl_tok:
                    src_start = i + 1
                    break
            if src_start is None:
                continue
            cur = next((v for v in (tok_int(t) for t in toks[cur_start:min(cur_start + 8, size)])
                        if v is not None), None)
            old = next((v for v in (tok_int(t) for t in toks[src_start:min(src_start + 8, size)])
                        if v is not None), None)
            cont = toks[start:min(start + k, size)]
            if not cont or cur is None or old is None or cur == old:
                return cont
            delta = cur - old
            for j, t in enumerate(cont[:12]):
                val = tok_int(t)
                if val is not None:
                    tid = int_token(val + delta)
                    if tid is not None:
                        out = list(cont)
                        out[j] = tid
                        return out
                    return cont
            return cont
        return self.draft(k)

    def draft_record(self, k: int, tok_int, int_token, nl_of, tok) -> list[int]:
        """Text-level record extrapolation.

        Token boundaries do not line up with line boundaries (one token can carry a
        closing brace, a newline and an opening brace), so the record is rebuilt in
        text: take the last two complete lines, advance every number in the last one
        by the step seen between them, and re-tokenise.  Only the tokens that follow
        the partial line already emitted are returned.
        """
        if tok is None:
            return []
        toks = self.tokens
        if len(toks) < 16:
            return []
        text = tok.decode(toks[-384:])
        lines = text.split(chr(10))
        if len(lines) < 3:
            return []
        cur, prev, prev2 = lines[-1], lines[-2], lines[-3]
        if not prev.strip():
            return []
        a = re.findall(r"[0-9]+", prev)
        b = re.findall(r"[0-9]+", prev2)
        if len(a) != len(b):
            step = [1] * len(a)
        else:
            step = [int(x) - int(y) for x, y in zip(a, b)]
        it = iter(zip(a, step))
        def bump(m):
            val, st = next(it)
            return str(int(val) + (st if st != 0 else 1))
        new_line = re.sub(r"[0-9]+", bump, prev)
        ids = tok.encode(new_line, add_special_tokens=False)
        cur_ids = tok.encode(cur, add_special_tokens=False) if cur else []
        if cur_ids and ids[:len(cur_ids)] != cur_ids:
            return []
        return ids[len(cur_ids):][:k]

    def draft_chains(self, k: int, n_chains: int) -> list[list[int]]:
        """Up to n_chains distinct continuations, longest context first.

        Different occurrences of the same suffix are genuinely different futures, so
        verifying them side by side catches the branch the model actually takes.
        """
        toks = self.tokens
        size = len(toks)
        out: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for order, table in reversed(self.grams):
            if size < order + 1:
                continue
            bucket = table.get(tuple(toks[-order:]))
            if not bucket:
                continue
            for pos in reversed(bucket):
                start = pos + order
                if start >= size:
                    continue
                cont = toks[start:min(start + k, size)]
                key = tuple(cont)
                if not cont or key in seen:
                    continue
                seen.add(key)
                out.append(cont)
                if len(out) >= n_chains:
                    return out
        return out

    # ------------------------------------------------------------------ misc
    @property
    def tokens(self) -> list[int]:
        # hot path: the decode step reads this once per sequence and _index() reads it
        # per appended token, so rebuilding prompt + output every time dominated the
        # host side.  Keep one list and repair only the missing tail: a full rebuild is
        # O(context), which cost ~9 ms per step at 64K when a spec path appended to
        # output without touching the cache.
        have = len(self._all) - len(self.prompt)
        if have < 0 or have != len(self.output):
            if have < 0:
                self._all = self.prompt + list(self.output)
            else:
                self._all.extend(self.output[have:])
        return self._all

    @property
    def num_tokens(self) -> int:
        return len(self.prompt) + len(self.output)


class Engine:
    def __init__(self, model: KazeModel, *, block_size: int = 16, max_batch: int = 32,
                 max_model_len: int | None = None, max_batched_tokens: int = 2048,
                 num_blocks: int | None = None, gpu_memory_utilization: float = 0.9,
                 use_cuda_graph: bool = True, enable_prefix_cache: bool = True,
                 eos_token_ids: Iterable[int] = (), tokenizer=None, seed: int = 0,
                 speculative: bool = False, spec_k: int = 5, verbose: bool = True,
                 use_draft_model: bool = False,
                 spec_accept_ema: float = 1.0, cache_dtype: str = "fp16",
                 draft_model: "KazeModel | None" = None):
        self.model = model
        self.cfg: ModelConfig = model.cfg
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_batch = max_batch
        # with rope scaling the usable context is the extended one, not the native one
        cap = self.cfg.max_position_embeddings
        _sc = self.cfg.rope_scaling or {}
        if (_sc.get("rope_type") or _sc.get("type")) in ("yarn", "linear", "dynamic"):
            cap = int(cap * float(_sc.get("factor", 1.0)))
            if max_model_len is not None and max_model_len >= cap * 0.95:
                # measured on 0.5B at 128K: with factor=4 (= native*4) the model answers
                # with an empty string, while factor=8 recalls the needle.  Running right
                # at the extrapolation boundary loses long-range recall.
                import warnings
                warnings.warn(
                    f"max_model_len {max_model_len} is at the rope-scaling limit "
                    f"({cap}); set a larger rope_scaling factor (e.g. {int(_sc.get('factor', 1.0)) * 2}) "
                    "if long-range recall matters", stacklevel=2)
        self.max_model_len = min(max_model_len or cap, cap)
        self.max_batched_tokens = max_batched_tokens
        self.use_cuda_graph = use_cuda_graph
        self.enable_prefix_cache = enable_prefix_cache
        self.speculative = speculative
        self.spec_k = spec_k
        self._spec_cache: dict[int, dict] = {}
        self.eos_token_ids = set(int(t) for t in eos_token_ids)
        if tokenizer is not None:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                self.eos_token_ids.add(int(eos))
        self.generator = torch.Generator(device=model.device)
        self.generator.manual_seed(seed)
        # one slot of slack: a request may arrive with max_model_len prompt tokens plus
        # the start of the answer, and the old floor() sizing overflowed at exactly
        # 128K (index 8192 out of bounds for the block table)
        self.max_blocks_per_seq = blocks_for(self.max_model_len + block_size, block_size) + 2

        if num_blocks is None:
            num_blocks = self._blocks_from_memory(gpu_memory_utilization)
        _f16 = torch.float16
        # int8 KV was implemented and measured (KL 0.0225 vs fp16, same speed, 2x
        # capacity) but it breaks long-range needle recall (40K: "10" vs "7429") and
        # the kernel path became unstable once K/V were split into separate flags
        # (Triton pointer-type error, then an illegal access under randomised load).
        # Rejected: a half-working cache mode is worse than not offering it.  The
        # kernel branches stay for a future retry; only fp16 and fp8 are exposed.
        _kinds = {
            "fp16": (_f16, _f16),
            "fp8": (torch.float8_e5m2, torch.float8_e5m2),
            # K stays fp16 (quantising the keys is what breaks long-range recall) while
            # V goes to fp8: 25% less KV memory for long contexts
            "fp8v": (_f16, torch.float8_e5m2),
        }
        _kd, _vd = _kinds.get(cache_dtype, (_f16, _f16))
        self.cache = KVCache(self.cfg, num_blocks, block_size, model.device,
                             dtypes=(_kd, _vd))
        # a second, independent cache for the draft model: it has its own head layout, so
        # it cannot share blocks with the target
        self.draft_model = draft_model
        self.use_draft_model = use_draft_model
        self.draft_cache = None
        if draft_model is not None:
            self.draft_cache = KVCache(draft_model.cfg, num_blocks, block_size,
                                       draft_model.device)
        # the row bucket has to cover max_batch, otherwise sequences beyond the bucket
        # silently wait a step (which also inflates any "aggregate tok/s" measurement)
        cap = 16
        while cap < max_batch:
            cap *= 2
        self.buckets = [b for b in (16, 32, 64, 128, 256) if b <= cap] or [16]
        self.graphs: dict[tuple[int, str], torch.cuda.CUDAGraph] = {}
        self.static: dict[int, dict[str, torch.Tensor]] = {}
        self.waiting: list[Sequence] = []
        self.prefill_queue: list[Sequence] = []
        self.running: list[Sequence] = []
        # counter-rewritten copy drafts: measured neutral on JSONL edit and 20% worse
        # on code edit (the "most recent integer" heuristic picks the wrong field), so
        # it stays off until the delta can be tied to the record boundary
        # record-aware drafts: copy the previous record with its first integer +1.
        # Earlier versions looked for "the last integer in the context" and rewrote
        # the wrong field; this one only fires at a record boundary and only touches
        # the first number of the copied record.
        self.counter_drafts = tokenizer is not None
        self._nl: int | None = None
        self._tok_int_cache: dict = {}
        self._tok_nl_cache: dict = {}
        self._int_tok_cache: dict = {}
        self.spec_tdim = 16 if spec_k <= 16 else (32 if spec_k <= 32 else 64)


        self.spec_plan = sorted({(k, 16 if k <= 16 else (32 if k <= 32 else 64))
                                 for k in (min(8, spec_k), spec_k, min(64, spec_k * 2))})
        # start pessimistic: a fresh workload is usually prose, and the first
        # wide verification on text that never repeats is pure overhead
        self._accept_ema = spec_accept_ema
        self.spec_max_batch = 2       # beyond this, batched decode wins without drafts
        self.attn_mode_override = None   # "short" / "long", for A/B measurements
        self.attn_block_n = 128          # 256 measured neutral, kept as a knob
        self._spec_probe = 0
        self.stats = {"steps": 0, "prefill_tokens": 0, "decode_tokens": 0,
                      "preemptions": 0, "prefix_hits": 0,
                      "spec_steps": 0, "spec_tokens": 0}
        self._step_lock = threading.Lock()
        if verbose:
            print(f"  kv cache: {num_blocks} blocks x {block_size} tokens x "
                  f"{self.cfg.num_hidden_layers} layers = "
                  f"{num_blocks * self.cache.bytes_per_block() / 1e9:.2f} GB", flush=True)

    # --------------------------------------------------------------- helpers
    def _blocks_from_memory(self, util: float) -> int:
        free, total = torch.cuda.mem_get_info()
        # keep a wider margin: the loader's transient buffers and fragmentation are not
        # visible to mem_get_info yet (a 7B int4 model OOMed at 256 MB of slack)
        usable = int(free * util) - (768 << 20)
        return max(8, usable // self.cache_bytes_per_block())

    def cache_bytes_per_block(self) -> int:
        return 2 * self.cfg.num_hidden_layers * self.block_size * self.cfg.kv_size * 2

    def _hash(self, parent: int, tokens: list[int]) -> int:
        return hash((parent, tuple(tokens)))

    # ------------------------------------------------------------- scheduling
    def add_request(self, prompt: list[int] | torch.Tensor, params: SamplingParams | None = None,
                    request_id: str | None = None) -> Sequence:
        ids = prompt.tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
        seq = Sequence(ids[: self.max_model_len], params or SamplingParams(), request_id)
        if self.speculative:
            seq.build_index()
        self.waiting.append(seq)
        return seq

    def _ensure_blocks(self, seq: Sequence, upto: int, reserve: int = 0) -> bool:
        need = blocks_for(upto, self.block_size) + reserve
        while len(seq.blocks) < need:
            try:
                blk = self.cache.allocator.allocate()
            except MemoryError:
                return False
            if seq.bt is None:
                seq.bt = torch.zeros(self.max_blocks_per_seq, dtype=torch.int32,
                                     device=self.model.device)
            seq.bt[len(seq.blocks)] = blk
            seq.blocks.append(blk)
            seq.hashes.append(0)
        return True

    def _match_prefix(self, seq: Sequence) -> None:
        if not self.enable_prefix_cache:
            return
        parent = 0
        n_full = len(seq.prompt) // self.block_size
        matched = 0
        for i in range(n_full):
            toks = seq.prompt[i * self.block_size:(i + 1) * self.block_size]
            h = self._hash(parent, toks)
            blk = self.cache.allocator.lookup(h)
            if blk is None:
                break
            self.cache.allocator.retain(blk)
            if seq.bt is None:
                seq.bt = torch.zeros(self.max_blocks_per_seq, dtype=torch.int32,
                                     device=self.model.device)
            seq.bt[len(seq.blocks)] = blk
            seq.blocks.append(blk)
            seq.hashes.append(h)
            parent = h
            matched += 1
        seq.num_computed = matched * self.block_size
        self.stats["prefix_hits"] += matched

    def _register_prefix(self, seq: Sequence, new_start: int, new_end: int) -> None:
        if not self.enable_prefix_cache:
            return
        bs = self.block_size
        first = (new_start // bs) * bs
        for start in range(first, new_end, bs):
            end = start + bs
            if end > new_end:
                break
            i = start // bs
            if i >= len(seq.blocks) or seq.hashes[i] != 0:
                continue
            parent = seq.hashes[i - 1] if i > 0 else 0
            toks = seq.tokens[start:end]
            h = self._hash(parent, toks)
            seq.hashes[i] = h
            self.cache.allocator.register_hash(h, seq.blocks[i])

    def _free(self, seq: Sequence) -> None:
        for blk in seq.blocks:
            self.cache.allocator.release(blk)
        if seq.bt is not None:
            seq.bt.zero_()
        seq.blocks.clear()
        seq.hashes.clear()
        seq.num_computed = 0

    def _preempt(self) -> bool:
        if len(self.running) <= 1:
            return False
        victim = self.running.pop()
        self._free(victim)
        victim.state = WAITING
        self.waiting.insert(0, victim)
        self.stats["preemptions"] += 1
        return True

    # ------------------------------------------------------------------ step
    def step(self) -> list[RequestOutput]:
        # The engine owns the GPU and is not re-entrant: one owner steps it.
        with self._step_lock:
            return self._step()

    def _step(self) -> list[RequestOutput]:
        outs: list[RequestOutput] = []
        budget = self.max_batched_tokens
        # decode first: keeps inter-token latency stable under heavy prefill
        self._decode_step(outs)
        self._prefill_step(outs, budget)
        self.stats["steps"] += 1
        return outs

    def _tok_int(self, tid: int):
        """token id -> int if the token is exactly a decimal number (cached)."""
        cache = self._tok_int_cache
        if tid in cache:
            return cache[tid]
        val = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode([tid])
            if text.isascii() and text.isdigit() and len(text) <= 6:
                val = int(text)
        cache[tid] = val
        return val

    def _nl_tok(self) -> int:
        if self._nl is None:
            self._nl = -1
            if self.tokenizer is not None:
                ids = self.tokenizer.encode(chr(10), add_special_tokens=False)
                if len(ids) == 1:
                    self._nl = ids[0]
        return self._nl

    def _tok_nl(self, tid: int) -> bool:
        """does this token's text contain a newline? (tokenizers merge '}\n')"""
        cache = self._tok_nl_cache
        val = cache.get(tid)
        if val is None:
            val = self.tokenizer is not None and chr(10) in self.tokenizer.decode([tid])
            cache[tid] = val
        return val

    def _int_token(self, value: int):
        """int -> token id if the number is a single token (cached, encode on demand)."""
        cache = self._int_tok_cache
        if value in cache:
            return cache[value]
        tid = None
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(str(value), add_special_tokens=False)
            if len(ids) == 1 and self._tok_int(ids[0]) is not None:
                tid = ids[0]
        cache[value] = tid
        return tid

    def _spec_static(self, tdim: int, k: int) -> dict:
        st = self._spec_cache.get((tdim, k))
        if st is None:
            dev = self.model.device
            st = {
                "ids": torch.zeros(k, dtype=torch.long, device=dev),
                "slots": torch.zeros(k, dtype=torch.long, device=dev),
                "bt": torch.zeros(self.max_blocks_per_seq, dtype=torch.int32, device=dev),
                "qs": torch.zeros(1, dtype=torch.int32, device=dev),
                "sl": torch.zeros(1, dtype=torch.int32, device=dev),
                "out": self.model.get_buffers(tdim).logits,
                "graph": None,
            }
            self._spec_cache[(tdim, k)] = st
        return st

    def capture_spec_graph(self) -> None:
        """Capture the verification pass (fixed K tokens) for each bucket."""
        if not self.speculative:
            return
        for k, tdim in self.spec_plan:
            st = self._spec_static(tdim, k)
            if st["graph"] is not None:
                continue
            bufs = self.model.get_buffers(tdim)
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.model.verify_logits(st["ids"], bufs.pos, st["slots"], self.cache,
                                             st["bt"], st["qs"], st["sl"], tdim=tdim,
                                             n_real=k)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.model.verify_logits(st["ids"], bufs.pos, st["slots"], self.cache,
                                         st["bt"], st["qs"], st["sl"], tdim=tdim,
                                         n_real=k)
            st["graph"] = g
            g.replay()
        torch.cuda.synchronize()

    def _draft_graph_state(self):
        """Capture a one-row decode for the draft model so drafting costs a replay."""
        if getattr(self, "_dgs", None) is not None:
            return self._dgs
        dm, dc = self.draft_model, self.draft_cache
        tdim = 16
        dm.get_buffers(tdim)
        dev = dm.device
        st = {
            "ids": torch.zeros(tdim, dtype=torch.long, device=dev),
            "pos": torch.zeros(tdim, dtype=torch.long, device=dev),
            "slots": torch.zeros(tdim, dtype=torch.long, device=dev),
            "bt": torch.zeros((1, self.max_blocks_per_seq), dtype=torch.int32, device=dev),
            "qs": torch.zeros(1, dtype=torch.int32, device=dev),
            "sl": torch.zeros(1, dtype=torch.int32, device=dev),
            "out": torch.zeros((tdim, dm.cfg.vocab_size), dtype=torch.float16, device=dev),
            "graph": None,
        }
        if self.use_cuda_graph:
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    st["out"].copy_(dm.decode_logits(
                        st["ids"][:1], st["pos"][:1], st["slots"][:1], st["bt"],
                        st["sl"], dc, tdim=tdim))
                st["graph"] = g
            except Exception:
                st["graph"] = None
        self._dgs = st
        return st

    def _draft_one(self, seq, token: int, at: int, blocks, bt) -> int:
        """One greedy draft step for `token` at position `at`; returns the next token."""
        dm, dc = self.draft_model, self.draft_cache
        st = self._draft_graph_state()
        slot = blocks[at // self.block_size] * self.block_size + at % self.block_size
        st["ids"].zero_()
        st["ids"][0] = token
        st["pos"].zero_()
        st["pos"][0] = at
        st["slots"].zero_()
        st["slots"][0] = slot
        nb_cols = min(bt.shape[1], st["bt"].shape[1])
        st["bt"].zero_()
        st["bt"][:, :nb_cols].copy_(bt.view(1, -1)[:, :nb_cols])
        st["qs"][0] = 0
        st["sl"][0] = at + 1
        if st["graph"] is not None:
            st["graph"].replay()
            lg = st["out"][0]
        else:
            dev = dm.device
            lg = dm.decode_logits(torch.tensor([token], dtype=torch.long, device=dev),
                                  torch.tensor([at], dtype=torch.long, device=dev),
                                  torch.tensor([slot], dtype=torch.long, device=dev),
                                  bt.view(1, -1),
                                  torch.tensor([at + 1], dtype=torch.long, device=dev),
                                  dc)[0]
        return int(lg.argmax().item())
    def _draft_tokens_from_model(self, seq: Sequence, k: int) -> list[int]:
        """Draft k tokens with the small model, keeping its own KV cache in sync.

        Positions at or beyond seq.draft_len are recomputed: an accepted-draft step
        leaves the drafter's cache holding a token that the target rejected.
        """
        dm, dc = self.draft_model, self.draft_cache
        if dm is None or dc is None or k <= 0:
            return []
        P = seq.num_computed
        if P <= 0 or P >= len(seq.tokens):
            return []
        if seq.draft_blocks is None:
            seq.draft_blocks = []
        need_blocks = (P + k) // self.block_size + 2
        while len(seq.draft_blocks) < need_blocks:
            seq.draft_blocks.append(dc.allocator.allocate())
        bt = dc.block_table_tensor(seq.draft_blocks, len(seq.draft_blocks))
        if bt.dim() == 1:
            bt = bt.view(1, -1)
        dev = dm.device
        start = seq.draft_len
        if start >= P:
            start = P - 1          # need the logits for the last computed position
        chunk = torch.tensor(seq.tokens[start:P], dtype=torch.long, device=dev)
        pos = torch.arange(start, P, dtype=torch.long, device=dev)
        lg = dm.prefill_logits(chunk, pos, dc, bt, start, last_only=True)
        nxt = int(lg[0].argmax().item())
        out = [nxt]
        for i in range(k - 1):
            at = P + i
            slot = (seq.draft_blocks[at // self.block_size] * self.block_size
                    + at % self.block_size)
            lg = dm.decode_logits(torch.tensor([out[-1]], dtype=torch.long, device=dev),
                                  torch.tensor([at], dtype=torch.long, device=dev),
                                  torch.tensor([slot], dtype=torch.long, device=dev),
                                  bt.view(1, -1),
                                  torch.tensor([at + 1], dtype=torch.long, device=dev), dc)
            out.append(int(lg[0].argmax().item()))
        seq.draft_len = P + k
        return out

    def _spec_decode(self, seq: Sequence, outs: list[RequestOutput]) -> bool:
        """Verify an n-gram draft in one pass; False means fall back to decode."""
        # the width of the verification should follow how much the drafter actually
        # has to offer, not how well the last steps went
        max_k = self.spec_plan[-1][0]
        if self.counter_drafts:
            # two proposals: the extrapolated record and the plain longest-suffix copy.
            # The longer one is worth verifying (it can accept at least as many tokens).
            rec = seq.draft_record(max_k - 1, self._tok_int, self._int_token, self._tok_nl,
                                   self.tokenizer)
            cop = seq.draft(max_k - 1)
            long_draft = rec if len(rec) >= len(cop) else cop
        else:
            long_draft = seq.draft(max_k - 1)
        if self.draft_model is not None and self.use_draft_model:
            # cross-model drafting: the small model proposes, the target verifies.
            # EXPERIMENTAL: correct but currently far slower than prompt-lookup
            # (13 vs 166 tok/s on a 3B target with a 0.5B drafter); the per-step draft
            # cost dominates.  Off unless use_draft_model=True.
            md = self._draft_tokens_from_model(seq, max_k - 1)
            if len(md) > len(long_draft):
                long_draft = md
        want = 1 + len(long_draft)
        # keep the wide verification only for drafts that can actually fill it:
        # a 32-row verify that accepts 6 tokens costs twice a 16-row one
        if want > 24:
            K, tdim = max(24, min(max_k, 32)), max(32, min(max_k, 32))
            if K > self.spec_plan[-1][0]:
                K, tdim = self.spec_plan[-1]
        elif want > 8:
            K, tdim = min(self.spec_plan[-2][0], 16), 16
        else:
            K, tdim = min(self.spec_plan[0][0], 8), 16
        P = seq.num_computed
        if P <= 0 or P >= len(seq.tokens) or P + K > self.max_model_len:
            return False
        draft = long_draft[: K - 1]
        if not draft:
            return False
        n = 1 + len(draft)
        if not self._ensure_blocks(seq, P + K, reserve=1):
            return False
        st = self._spec_static(tdim, K)
        dev = self.model.device
        # always fill exactly K rows: the verification graph is captured for K, and
        # padding rows must not write KV into slot 0 (that silently corrupted the
        # prefix before).  Extra rows are ignored when accepting.
        rows = [seq.tokens[P]] + draft + [draft[-1]] * (K - n)
        st["ids"].zero_()
        st["ids"][:K].copy_(torch.tensor(rows, dtype=torch.long, device=dev))
        st["slots"][:K].copy_(self.cache.slots_of(seq.bt, P, P + K))
        st["slots"][K:].zero_()
        st["bt"].copy_(seq.bt)
        st["qs"][0] = P
        st["sl"][0] = P + K
        bufs = self.model.get_buffers(tdim)
        bufs.pos[:K].copy_(torch.arange(P, P + K, dtype=torch.long, device=dev))
        replay = self.use_cuda_graph and st["graph"] is not None
        if replay:
            st["graph"].replay()
        else:
            self.model.verify_logits(st["ids"], bufs.pos, st["slots"], self.cache,
                                     st["bt"], st["qs"], st["sl"], tdim=tdim, n_real=K)
        preds = st["out"][:n].argmax(-1).tolist()
        new_tokens: list[int] = []
        for i in range(len(draft)):
            if preds[i] == draft[i]:
                new_tokens.append(draft[i])
            else:
                new_tokens.append(preds[i])
                break
        else:
            new_tokens.append(preds[len(draft)])
        m = len(new_tokens)
        seq.num_computed = P + n if m == n else P + m
        now = time.perf_counter()
        if seq.last_token_time is not None:
            seq.itls.append(now - seq.last_token_time)
        seq.last_token_time = now
        if seq.first_token_time is None:
            seq.first_token_time = now
        reason = None
        for tok in new_tokens:
            seq.output.append(tok)
            seq._index(len(seq.tokens) - 1)
            reason = self._finish_reason(seq, tok)
            outs.append(RequestOutput(
                request_id=seq.request_id, tokens=[tok], text="",
                finished=reason is not None, finish_reason=reason,
                ttft_ms=(seq.first_token_time - seq.arrival) * 1e3,
                tpot_ms=None if not seq.itls else seq.itls[-1] * 1e3))
            if reason is not None:
                seq.state = DONE
                seq.finish_reason = reason
                break
        if seq.state == DONE:
            if seq in self.running:
                self.running.remove(seq)
            self._free(seq)
        self.stats["spec_steps"] += 1
        self.stats["spec_tokens"] += m
        self._accept_ema = 0.5 * self._accept_ema + 0.5 * m
        return True

    def _spec_decode_chains(self, seq: Sequence, outs: list[RequestOutput]) -> bool:
        """Verify several draft branches at once using shadow KV blocks.

        Each branch gets a private copy of the partially filled block the next token
        lands in, so the branches cannot clobber each other's KV.  One packed forward
        scores every branch; the longest accepted run wins and its block is copied back.
        """
        n_chains = getattr(self, "spec_chains", 0)
        if n_chains < 2:
            return False
        bs = self.block_size
        P = seq.num_computed
        tdim = getattr(self, "spec_chain_tdim", 32)
        k_chain = tdim // n_chains
        if k_chain < 2 or P <= 0 or P >= len(seq.tokens) or P + k_chain > self.max_model_len:
            return False
        off = P % bs
        if off + k_chain > bs:
            return False
        chains = seq.draft_chains(k_chain - 1, n_chains)
        if len(chains) < 2:
            return False
        if not self._ensure_blocks(seq, P + k_chain, reserve=1):
            return False
        bp = P // bs
        real = seq.blocks[bp]
        alloc = self.cache.allocator
        scratch = []
        try:
            for _ in chains:
                scratch.append(alloc.allocate())
        except MemoryError:
            for x in scratch:
                alloc.release(x)
            return False
        layers = self.cfg.num_hidden_layers
        try:
            if off:
                src = self.cache.k[:, real, :off]
                dst_idx = torch.tensor(scratch, dtype=torch.long, device=self.cache.device)
                dst = self.cache.k[:, dst_idx, :off]
                dst.copy_(src.unsqueeze(1).expand_as(dst))
                src_v = self.cache.v[:, real, :off]
                dst_v = self.cache.v[:, dst_idx, :off]
                dst_v.copy_(src_v.unsqueeze(1).expand_as(dst_v))

            ids_l, pos_l, slot_l, groups = [], [], [], []
            for c, cont in enumerate(chains):
                toks = [seq.tokens[P]] + cont
                lo = len(ids_l)
                for j, tok in enumerate(toks):
                    p = P + j
                    ids_l.append(tok)
                    pos_l.append(p)
                    slot_l.append(scratch[c] * bs + off + j)
                groups.append((lo, len(ids_l), c, P, P + k_chain))
            n = len(ids_l)
            vdim = 8 if n <= 8 else (16 if n <= 16 else (32 if n <= 32 else 64))
            if n > vdim:
                return False
            dev = self.model.device
            ids_t = torch.zeros(vdim, dtype=torch.long, device=dev)
            pos_t = torch.zeros(vdim, dtype=torch.long, device=dev)
            slot_t = torch.zeros(vdim, dtype=torch.long, device=dev)
            ids_t[:n] = torch.tensor(ids_l, dtype=torch.long, device=dev)
            pos_t[:n] = torch.tensor(pos_l, dtype=torch.long, device=dev)
            slot_t[:n] = torch.tensor(slot_l, dtype=torch.long, device=dev)
            bt = torch.zeros((len(chains), self.max_blocks_per_seq), dtype=torch.int32, device=dev)
            for c in range(len(chains)):
                bt[c, :len(seq.bt)].copy_(seq.bt)
                bt[c, bp] = scratch[c]
            logits = self.model.verify_logits_multi(ids_t, pos_t, slot_t, self.cache, bt,
                                                    groups, tdim=vdim, n_real=n)
            best_m, best_c, best_tokens = 0, 0, []
            for c, cont in enumerate(chains):
                lo, hi, _b, _qs, _sl = groups[c]
                preds = logits[lo:lo + 1 + len(cont)].argmax(-1).tolist()
                toks = []
                for j in range(len(cont)):
                    if preds[j] == cont[j]:
                        toks.append(cont[j])
                    else:
                        toks.append(preds[j])
                        break
                else:
                    toks.append(preds[len(cont)])
                if len(toks) > best_m:
                    best_m, best_c, best_tokens = len(toks), c, toks
            if best_m == 0:
                return False
            # 勝った枝の KV を本物のブロックへ戻す
            win = scratch[best_c]
            self.cache.k[:, real, off:off + best_m].copy_(
                self.cache.k[:, win, off:off + best_m])
            self.cache.v[:, real, off:off + best_m].copy_(
                self.cache.v[:, win, off:off + best_m])
            seq.num_computed = P + best_m
            now = time.perf_counter()
            if seq.last_token_time is not None:
                seq.itls.append(now - seq.last_token_time)
            seq.last_token_time = now
            if seq.first_token_time is None:
                seq.first_token_time = now
            for tok in best_tokens:
                seq.output.append(tok)
                seq._index(len(seq.tokens) - 1)
                reason = self._finish_reason(seq, tok)
                outs.append(RequestOutput(
                    request_id=seq.request_id, tokens=[tok], text="",
                    finished=reason is not None, finish_reason=reason,
                    ttft_ms=(seq.first_token_time - seq.arrival) * 1e3,
                    tpot_ms=None if not seq.itls else seq.itls[-1] * 1e3))
                if reason is not None:
                    seq.state = DONE
                    seq.finish_reason = reason
                    break
            if seq.state == DONE:
                if seq in self.running:
                    self.running.remove(seq)
                self._free(seq)
            self.stats["spec_steps"] += 1
            self.stats["spec_tokens"] += best_m
            self._accept_ema = 0.5 * self._accept_ema + 0.5 * best_m
            return True
        finally:
            for x in scratch:
                alloc.release(x)

    def _spec_decode_multi(self, seqs: list[Sequence], outs: list[RequestOutput]) -> bool:
        """Speculative decode for a batch: every sequence drafts, one forward verifies.

        Rows are packed per sequence (no gaps) so the KV scatter can use a single
        row count; each sequence's attention only sees its own draft window.
        """
        K = min(self.spec_k, self.spec_tdim)
        bs = self.block_size
        plans = []
        for s in seqs:
            P = s.num_computed
            if P <= 0 or P >= len(s.tokens) or P + K > self.max_model_len:
                continue
            draft = s.draft(K - 1)
            if not draft:
                continue
            if not self._ensure_blocks(s, P + K, reserve=1):
                continue
            plans.append((s, P, draft))
        if len(plans) < 2:
            return False

        dev = self.model.device
        ids_l: list[int] = []
        pos_l: list[int] = []
        slot_l: list[int] = []
        lens_l: list[int] = []
        groups = []
        rows_per = []
        for (s, P, draft) in plans:
            n_real = 1 + len(draft)
            seq_ids = [s.tokens[P]] + draft + [draft[-1]] * (K - n_real)
            lo = len(ids_l)
            for j, tok in enumerate(seq_ids):
                p = P + j
                ids_l.append(tok)
                pos_l.append(p)
                slot_l.append(s.blocks[p // bs] * bs + p % bs)
                lens_l.append(P + 1 + j)
            rows_per.append((n_real, lo))
            groups.append((lo, len(ids_l), 0, P, P + K))
        n = len(ids_l)
        bench = getattr(self, "spec_buckets", (8, 16, 32, 48, 64, 96, 128))
        tdim = next((t for t in bench if t >= n), bench[-1])
        if n > tdim:
            return False

        if max(slot_l) >= self.cache.k_flat[0].numel() or max(pos_l) >= self.model.cos.shape[0]:
            raise RuntimeError(f"spec slots/pos out of range: max slot {max(slot_l)} "
                               f"(kv {self.cache.k_flat[0].numel()}), max pos {max(pos_l)} "
                               f"(rope {self.model.cos.shape[0]})")
        if len(slot_l) != len(ids_l) or len(pos_l) != len(ids_l):
            raise RuntimeError("spec row lists disagree")
        # the kernels read the (positions, slots) tensors for every padded row too,
        # so they must be allocated at the full bucket width
        ids_t = torch.zeros(tdim, dtype=torch.long, device=dev)
        pos_t = torch.zeros(tdim, dtype=torch.long, device=dev)
        slot_t = torch.zeros(tdim, dtype=torch.long, device=dev)
        ids_t[:n] = torch.tensor(ids_l, dtype=torch.long, device=dev)
        pos_t[:n] = torch.tensor(pos_l, dtype=torch.long, device=dev)
        slot_t[:n] = torch.tensor(slot_l, dtype=torch.long, device=dev)
        # the attention kernel indexes the row by token position, so the tensor has
        # to span the same block-table width the decode path uses
        bt = torch.zeros((len(plans), self.max_blocks_per_seq), dtype=torch.int32, device=dev)
        for i, (s, P, draft) in enumerate(plans):
            bt[i, :len(s.bt)].copy_(s.bt)
        groups = [(lo, hi, i, q_start, seq_len)
                  for i, (lo, hi, _unused, q_start, seq_len) in enumerate(groups)]

        logits = self.model.verify_logits_multi(ids_t, pos_t, slot_t, self.cache, bt,
                                                groups, tdim=tdim, n_real=n)

        now = time.perf_counter()
        for i, (s, P, draft) in enumerate(plans):
            lo, hi, _b, _qs, _sl = groups[i]
            n_real = rows_per[i][0]
            preds = logits[lo:lo + n_real].argmax(-1).tolist()
            new_tokens: list[int] = []
            for j in range(len(draft)):
                if preds[j] == draft[j]:
                    new_tokens.append(draft[j])
                else:
                    new_tokens.append(preds[j])
                    break
            else:
                new_tokens.append(preds[len(draft)])
            m = len(new_tokens)
            n_rows = n_real
            s.num_computed = P + n_rows if m == n_rows else P + m
            if s.last_token_time is not None:
                s.itls.append(now - s.last_token_time)
            s.last_token_time = now
            if s.first_token_time is None:
                s.first_token_time = now
            for tok in new_tokens:
                s.output.append(tok)
                s._index(len(s.tokens) - 1)
                reason = self._finish_reason(s, tok)
                outs.append(RequestOutput(
                    request_id=s.request_id, tokens=[tok], text="",
                    finished=reason is not None, finish_reason=reason,
                    ttft_ms=(s.first_token_time - s.arrival) * 1e3,
                    tpot_ms=None if not s.itls else s.itls[-1] * 1e3))
                if reason is not None:
                    s.state = DONE
                    s.finish_reason = reason
                    break
            if s.state == DONE:
                if s in self.running:
                    self.running.remove(s)
                self._free(s)
            self.stats["spec_steps"] += 1
            self.stats["spec_tokens"] += m
        return True

    def _decode_step(self, outs: list[RequestOutput]) -> None:
        seqs = [s for s in self.running if s.state == DECODE]
        if not seqs:
            return
        self._spec_probe += 1
        # a drafter that stops hitting is pure overhead: fall back to plain decode and
        # only re-probe now and then so the choice follows the workload
        want_spec = self._accept_ema >= 1.4 or self._spec_probe % 16 == 0
        # verification scales with the row count while plain batched decode does not,
        # so speculation only pays at low concurrency (measured: 5.3x at batch 1,
        # 1.1x at 4, 0.74x -- a regression -- at 8)
        if self.speculative and seqs and want_spec and len(seqs) <= self.spec_max_batch:
            if len(seqs) == 1 and seqs[0].params.temperature <= 0:
                if self._spec_decode_chains(seqs[0], outs):
                    return
                if self._spec_decode(seqs[0], outs):
                    return
            elif len(seqs) > 1 and all(s.params.temperature <= 0 for s in seqs):
                if self._spec_decode_multi(seqs, outs):
                    return
        room = []
        for s in seqs:
            if not self._ensure_blocks(s, s.num_computed + 1):
                if not self._preempt():
                    continue
            room.append(s)
        seqs = room
        if not seqs:
            return
        b = len(seqs)
        tdim = next((t for t in self.buckets if t >= b), self.buckets[-1])
        st = self._static(tdim)
        bs = self.block_size
        ids = st["ids"]
        pos = st["positions"]
        slots = st["slots"]
        bt = st["block_tables"]
        lens = st["seq_lens"]
        n_real = min(b, tdim)
        # build the rows as python lists and upload once per tensor: at 32 streams the
        # per-row .copy_() loop cost more than the kernels it was feeding
        use = seqs[:n_real]
        id_l = [s.tokens[s.num_computed] for s in use]
        pos_l = [s.num_computed for s in use]
        slot_l = [s.blocks[s.num_computed // bs] * bs + s.num_computed % bs for s in use]
        dev = self.model.device
        ids[:n_real].copy_(torch.tensor(id_l, dtype=torch.long, device=dev))
        pos[:n_real].copy_(torch.tensor(pos_l, dtype=torch.long, device=dev))
        slots[:n_real].copy_(torch.tensor(slot_l, dtype=torch.long, device=dev))
        lens[:n_real].fill_(0)
        lens[:n_real].copy_(torch.tensor([s.num_computed + 1 for s in use],
                                         dtype=torch.int32, device=dev))
        bt[:n_real].copy_(torch.stack([s.bt for s in use]))
        if n_real < tdim:
            ids[n_real:].zero_()
            pos[n_real:].zero_()
            slots[n_real:].zero_()
            lens[n_real:].fill_(1)
            bt[n_real:].zero_()
        max_len = max((s.num_computed for s in seqs), default=1)
        mode = self.attn_mode_override or ("short" if max_len <= 1024 else "long")
        if getattr(self.cache, "q8", False):
            # only the direct (single-pass) kernel knows how to apply the int8 scales
            mode = "short"
        graph = self.graphs.get((tdim, mode))
        if self.use_cuda_graph and graph is not None:
            graph.replay()
            logits = st["out"][:n_real].clone()
        else:
            logits = self.model.decode_logits(ids, pos, slots, bt, lens, self.cache,
                                              tdim=tdim, attn_mode=mode).clone()[:n_real]
        for s in seqs[:n_real]:
            s.num_computed += 1      # the token fed this step now has its KV cached
        self._sample_and_append(seqs[:n_real], logits, outs)
        self.stats["decode_tokens"] += n_real

    def _prefill_step(self, outs: list[RequestOutput], budget: int) -> None:
        queue = self.prefill_queue
        while budget > 0:
            if queue:
                seq = queue[0]
            elif self.waiting:
                seq = self.waiting.pop(0)
                if not seq.blocks:
                    self._match_prefix(seq)
                if seq.num_computed >= len(seq.prompt):
                    # fully cached (prefix cache hit covered the whole prompt)
                    self._finish_prefill(seq, None, outs)
                    continue
                if not self._ensure_blocks(seq, seq.num_computed + min(
                        len(seq.prompt) - seq.num_computed, budget), reserve=1):
                    self.waiting.insert(0, seq)
                    return
                queue.append(seq)
            else:
                return
            remaining = len(seq.prompt) - seq.num_computed
            if remaining <= 0:
                self._finish_prefill(seq, None, outs)
                queue.pop(0)
                continue
            chunk = min(remaining, budget)
            if not self._ensure_blocks(seq, seq.num_computed + chunk, reserve=1):
                if queue and queue[0] is seq:
                    queue.pop(0)
                self.waiting.insert(0, seq)
                return
            start = seq.num_computed
            end = start + chunk
            ids = torch.tensor(seq.prompt[start:end], device=self.model.device)
            positions = torch.arange(start, end, device=self.model.device)
            logits = self.model.prefill_logits(ids, positions, self.cache, seq.bt, start)
            seq.num_computed = end
            self._register_prefix(seq, start, end)
            self.stats["prefill_tokens"] += chunk
            budget -= chunk
            if seq.num_computed >= len(seq.prompt):
                queue.pop(0) if queue and queue[0] is seq else None
                self._finish_prefill(seq, logits, outs)
            else:
                break   # continue this sequence next step

    def _finish_prefill(self, seq: Sequence, logits: torch.Tensor | None,
                        outs: list[RequestOutput]) -> None:
        if seq.state == DECODE:
            return          # idempotent: never finish the same sequence twice
        if logits is None:
            # whole prompt was already cached: recompute the last token to get logits
            start = len(seq.prompt) - 1
            logits = self.model.prefill_logits(
                torch.tensor([seq.prompt[-1]], device=self.model.device),
                torch.arange(start, start + 1, device=self.model.device),
                self.cache, seq.bt, start)
        seq.state = DECODE
        self.running.append(seq)
        self._sample_and_append([seq], logits[:1], outs, first=True)

    # --------------------------------------------------------------- sampling
    def _sample_and_append(self, seqs: list[Sequence], logits: torch.Tensor,
                           outs: list[RequestOutput], first: bool = False) -> None:
        params = [s.params for s in seqs]
        history = [s.tokens[-128:] for s in seqs]
        # one host sync for the whole batch instead of one per sequence
        toks = sample_rows(logits, params, history, self.generator).tolist()
        now = time.perf_counter()
        finished: list[Sequence] = []
        for s, tok in zip(seqs, toks):
            if s.first_token_time is None:
                s.first_token_time = now
            else:
                s.itls.append(now - s.last_token_time)
            s.last_token_time = now
            s.output.append(tok)
            s._all.append(tok)
            s._index(len(s._all) - 1)
            reason = self._finish_reason(s, tok)
            if reason is not None:
                s.state = DONE
                s.finish_reason = reason
                finished.append(s)
            if self.tokenizer is not None and s.params.stop:
                s.text_so_far = self._decode_text(s)
            outs.append(RequestOutput(
                request_id=s.request_id, tokens=[tok], text=s.text_so_far,
                finished=reason is not None, finish_reason=reason,
                ttft_ms=(s.first_token_time - s.arrival) * 1e3,
                tpot_ms=None if not s.itls else s.itls[-1] * 1e3))
        for s in finished:
            if s in self.running:
                self.running.remove(s)
            self._free(s)

    def _decode_text(self, seq: Sequence) -> str:
        return self.tokenizer.decode(seq.output, skip_special_tokens=True)

    def _finish_reason(self, seq: Sequence, token: int) -> str | None:
        p = seq.params
        if len(seq.output) >= p.max_tokens:
            return "length"
        if token in self.eos_token_ids and not p.ignore_eos:
            return "stop"
        if p.stop_token_ids and token in p.stop_token_ids:
            return "stop"
        if p.stop and self.tokenizer is not None:
            text = self._decode_text(seq)
            for s in p.stop:
                if s and s in text:
                    return "stop"
        if seq.num_computed >= self.max_model_len:
            return "length"
        return None

    # ----------------------------------------------------------------- graphs
    def _static(self, tdim: int) -> dict[str, torch.Tensor]:
        st = self.static.get(tdim)
        if st is None:
            dev = self.model.device
            st = {
                "ids": torch.zeros(tdim, dtype=torch.long, device=dev),
                "positions": torch.zeros(tdim, dtype=torch.long, device=dev),
                "slots": torch.zeros(tdim, dtype=torch.long, device=dev),
                "block_tables": torch.zeros((tdim, self.max_blocks_per_seq),
                                            dtype=torch.int32, device=dev),
                "seq_lens": torch.ones(tdim, dtype=torch.int32, device=dev),
            }
            bufs = self.model.get_buffers(tdim)
            st["out"] = bufs.logits
            self.static[tdim] = st
        return st

    def capture_graphs(self) -> None:
        if not self.use_cuda_graph:
            return
        for tdim in self.buckets:
            for mode in ("short", "long"):
                if (tdim, mode) in self.graphs:
                    continue
                st = self._static(tdim)
                bufs = self.model.get_buffers(tdim)
                torch.cuda.synchronize()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self.model.decode_logits(st["ids"], st["positions"], st["slots"],
                                                 st["block_tables"], st["seq_lens"],
                                                 self.cache, tdim=tdim, bufs=bufs,
                                                 attn_mode=mode)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.model.decode_logits(st["ids"], st["positions"], st["slots"],
                                             st["block_tables"], st["seq_lens"],
                                             self.cache, tdim=tdim, bufs=bufs,
                                             attn_mode=mode)
                self.graphs[(tdim, mode)] = g
                g.replay()
        torch.cuda.synchronize()

    # --------------------------------------------------------------- driving
    def is_idle(self) -> bool:
        return not (self.waiting or self.prefill_queue or self.running)

    def generate(self, prompt: list[int] | torch.Tensor, params: SamplingParams | None = None,
                 on_token: Callable[[RequestOutput], None] | None = None) -> RequestOutput:
        seq = self.add_request(prompt, params)
        while seq.state != DONE:
            outs = self.step()
            if on_token is not None:
                for o in outs:
                    if o.request_id == seq.request_id:
                        on_token(o)
        return RequestOutput(request_id=seq.request_id, tokens=seq.output,
                             text=seq.text_so_far, finished=True,
                             finish_reason=seq.finish_reason,
                             ttft_ms=(seq.first_token_time - seq.arrival) * 1e3 if seq.first_token_time else None,
                             tpot_ms=sum(seq.itls) / len(seq.itls) * 1e3 if seq.itls else None)

    def run_until_idle(self, on_token: Callable[[RequestOutput], None] | None = None) -> None:
        while not self.is_idle():
            outs = self.step()
            if on_token is not None:
                for o in outs:
                    on_token(o)

    def kv_usage(self) -> float:
        return self.cache.used_blocks / self.cache.num_blocks

    def metrics(self) -> dict[str, float]:
        return dict(self.stats, kv_usage=self.kv_usage(),
                    running=len(self.running), waiting=len(self.waiting))
