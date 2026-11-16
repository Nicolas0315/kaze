"""Consistent decoded output for library and HTTP generation."""
from __future__ import annotations

from collections.abc import Sequence


def decode_output(tokenizer, tokens: list[int], stops: Sequence[str] = (),
                  *, finished: bool = True) -> str:
    """Hide stop markers and withhold incomplete text until it is stable.

    Tokenizer prefixes may end in a replacement character while a UTF-8
    character is incomplete. Stop strings may also span several tokens.
    Neither prefix is safe to send before subsequent tokens resolve it.
    """
    decode_kwargs = {"skip_special_tokens": True}
    if getattr(tokenizer, "clean_up_tokenization_spaces", False):
        # Cleanup can revise an already-emitted space before punctuation.
        decode_kwargs["clean_up_tokenization_spaces"] = False
    text = tokenizer.decode(tokens, **decode_kwargs)
    end = len(text)
    for stop in stops:
        if stop:
            index = text.find(stop)
            if index >= 0:
                end = min(end, index)
    if end < len(text):
        return text[:end]
    if finished:
        return text
    text = text.rstrip("\ufffd")
    withheld = 0
    for stop in stops:
        for size in range(min(len(text), len(stop) - 1), 0, -1):
            if text.endswith(stop[:size]):
                withheld = max(withheld, size)
                break
    return text[:-withheld] if withheld else text
