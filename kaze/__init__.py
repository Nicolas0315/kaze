"""Kaze - a bandwidth-roofline LLM inference engine.

Design goals
------------
* Prefill and decode both saturate HBM bandwidth (measured, reported).
* Real serving: continuous batching, paged KV cache, prefix caching,
  chunked prefill, CUDA-graph decode, OpenAI-compatible API.
* Small enough to read: one Triton kernel per operation.
"""
import os as _os

# Triton 3.9 は既定で fp fusion を有効にするが、未設定だと毎回警告を出す。
# 挙動は変えずに（有効のまま）静かにする。
_os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

from kaze.config import ModelConfig

__version__ = "0.1.0"
__all__ = ["Engine", "ModelConfig", "SamplingParams", "RequestOutput"]


def __getattr__(name):
    if name in ("Engine", "SamplingParams", "RequestOutput"):
        from kaze import engine

        return getattr(engine, name)
    raise AttributeError(name)
