"""LLM ラッパーの扱いやすさと、固まらないことの確認。

ここで見ているのは推論の正しさではなく、呼び出し側の都合です。
ループが死んだときに永久に待つ、といった失敗を作らないためのテストです。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from kaze.llm import LLM
from kaze.sampler import SamplingParams

MODELS = os.environ.get("KAZE_MODELS", os.path.expanduser("~/models"))
MODEL = os.path.join(MODELS, "Qwen2.5-0.5B-Instruct")

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            fn()
            results.append((name, True, ""))
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, repr(exc)))
            print(f"  FAIL {name}: {exc!r}")
        return fn
    return deco


def main() -> int:
    llm = LLM(MODEL, kind="int4", lm_head_kind="int8", verbose=False)

    @check("stream_text(sampling kwargs)")
    def _() -> None:
        text = "".join(t for t, _ in llm.stream_text("The capital of France is", max_tokens=8))
        assert text.strip(), "empty output"

    @check("generate(params=...)")
    def _() -> None:
        r = llm.generate("2 + 2 =", params=SamplingParams(max_tokens=4, temperature=0.0))
        assert r.tokens, "no tokens"

    @check("unknown sampling argument raises")
    def _() -> None:
        try:
            list(llm.stream_text("x", max_tokenz=5))
        except TypeError:
            return
        raise AssertionError("unknown kwarg was accepted")

    llm.shutdown()
    del llm
    torch.cuda.empty_cache()

    @check("start_loop=False fails fast instead of hanging")
    def _() -> None:
        llm2 = LLM(MODEL, kind="int8", start_loop=False, verbose=False)
        t0 = time.perf_counter()
        try:
            llm2.generate("hi", max_tokens=2)
        except RuntimeError:
            assert time.perf_counter() - t0 < 5.0, "took too long"
            return
        finally:
            llm2.shutdown()
        raise AssertionError("generate() returned without a loop")

    @check("engine loop failure reaches the caller")
    def _() -> None:
        llm3 = LLM(MODEL, kind="int8", verbose=False)

        def boom(*a, **k):
            raise RuntimeError("simulated kernel failure")

        llm3.engine.step = boom
        try:
            llm3.generate("hello", max_tokens=4)
        except RuntimeError:
            return
        finally:
            llm3.shutdown()
        raise AssertionError("a dead loop was not reported")

    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())