"""Token sampling: greedy, temperature, top-k, top-p, repetition penalty."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


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

    def clone(self) -> "SamplingParams":
        return SamplingParams(**{k: getattr(self, k) for k in self.__dataclass_fields__})


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
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V] -> token ids [B]."""
    b = logits.shape[0]
    # fast path: every row greedy and no repetition penalty -> one argmax on the
    # half-precision logits.  The generic path converts [B, V] to fp32 and gathers
    # the greedy rows again, which cost 3.8 ms of a 7.7 ms batched step.
    if (history is None or all(p.repetition_penalty == 1.0 for p in params)) and \
            all(p.temperature <= 0 for p in params):
        return logits.argmax(dim=-1)
    out = torch.empty(b, dtype=torch.long, device=logits.device)
    logits = logits.float()
    if history is not None:
        for row, p in enumerate(params):
            if p.repetition_penalty != 1.0:
                apply_repetition_penalty(logits[row:row + 1], [history[row]], p.repetition_penalty)
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
            mask = cum - probs > p.top_p
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            row = torch.full_like(row, float("-inf")).scatter(0, sorted_idx, sorted_logits)
        probs = torch.softmax(row, dim=-1)
        out[i] = torch.multinomial(probs, 1, generator=generator)
    return out
