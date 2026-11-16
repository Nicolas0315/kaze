"""Token sampling: greedy, temperature, top-k, top-p, repetition penalty."""
from __future__ import annotations

from dataclasses import dataclass, field

import math
from dataclasses import replace

import torch


MIN_REPETITION_PENALTY = 1e-6
MAX_REPETITION_PENALTY = 1e6


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        for name, minimum in (("max_tokens", 1), ("top_k", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("temperature", "top_p", "repetition_penalty", "presence_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not MIN_REPETITION_PENALTY <= self.repetition_penalty <= MAX_REPETITION_PENALTY:
            raise ValueError("repetition_penalty must be in [1e-6, 1e6]")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)
                                     or not -(2 ** 63) <= self.seed < 2 ** 64):
            raise ValueError("seed must be a supported 64-bit integer")
        if not isinstance(self.stop, (list, tuple)) or any(
                not isinstance(s, str) or not s for s in self.stop):
            raise ValueError("stop must contain non-empty strings")
        if not isinstance(self.stop_token_ids, (list, tuple)) or any(
                isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in self.stop_token_ids):
            raise ValueError("stop_token_ids must contain non-negative integers")
        self.stop = list(self.stop)
        self.stop_token_ids = list(self.stop_token_ids)

    def clone(self) -> "SamplingParams":
        return replace(self)


def apply_repetition_penalty(logits: torch.Tensor, history: list[list[int]],
                             penalty: float) -> None:
    if penalty == 1.0:
        return
    for row, toks in enumerate(history):
        if not toks:
            continue
        idx = torch.tensor(sorted(set(toks)), dtype=torch.long, device=logits.device)
        vals = logits[row].index_select(0, idx)
        vals = torch.where(vals > 0, vals / penalty, vals * penalty)
        logits[row].index_copy_(0, idx, vals)


@torch.no_grad()
def sample_rows(logits: torch.Tensor, params: list[SamplingParams],
                history: list[list[int]] | None = None,
                generator: torch.Generator | None = None,
                generators: list[torch.Generator | None] | None = None) -> torch.Tensor:
    """logits [B, V] -> token ids [B]."""
    b = logits.shape[0]
    # fast path: every row greedy and no repetition penalty -> one argmax on the
    # half-precision logits.  The generic path converts [B, V] to fp32 and gathers
    # the greedy rows again, which cost 3.8 ms of a 7.7 ms batched step.
    if (history is None or all(p.repetition_penalty == 1.0 and p.presence_penalty == 0
                               for p in params)) and \
            all(p.temperature <= 0 for p in params):
        return logits.argmax(dim=-1)
    out = torch.empty(b, dtype=torch.long, device=logits.device)
    logits = logits.to(dtype=torch.float32, copy=True)
    if history is not None:
        for row, p in enumerate(params):
            if p.repetition_penalty != 1.0:
                apply_repetition_penalty(logits[row:row + 1], [history[row]], p.repetition_penalty)
            if p.presence_penalty and history[row]:
                idx = torch.tensor(sorted(set(history[row])), dtype=torch.long, device=logits.device)
                logits[row, idx] -= p.presence_penalty
    greedy_rows, sample_rows_idx = [], []
    for i, p in enumerate(params):
        (greedy_rows if p.temperature <= 0 else sample_rows_idx).append(i)
    if greedy_rows:
        idx = torch.tensor(greedy_rows, device=logits.device)
        out[idx] = logits.index_select(0, idx).argmax(dim=-1)
    for i in sample_rows_idx:
        p = params[i]
        row = logits[i] / max(p.temperature, 1e-6)
        if p.top_k and p.top_k > 0 and p.top_k < row.numel():
            kth = torch.topk(row, p.top_k).values[-1]
            row = torch.where(row < kth, torch.full_like(row, float("-inf")), row)
        if p.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(row, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            # keep the smallest set whose cumulative probability covers top_p
            mask = cum - probs >= p.top_p
            mask[0] = False  # top_p can round to zero in the tensor dtype
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            row = torch.full_like(row, float("-inf")).scatter(0, sorted_idx, sorted_logits)
        probs = torch.softmax(row, dim=-1)
        row_generator = generators[i] if generators is not None else None
        out[i] = torch.multinomial(probs, 1, generator=(
            row_generator if row_generator is not None else generator))
    return out
