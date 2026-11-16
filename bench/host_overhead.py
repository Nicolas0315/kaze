"""Where does the batch-1 decode step time go: kernels, or host bookkeeping?

Compares the engine-measured inter-token latency with a bare CUDA-graph replay of the
same decode graph (the replay is what the step would cost if the host were free).
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main() -> None:
    from kaze.llm import LLM
    from kaze.sampler import SamplingParams

    llm = LLM("/models/Qwen2.5-0.5B-Instruct", kind="int4",
              lm_head_kind="int4", speculative=False, verbose=False)
    eng = llm.engine
    ids = list(torch.randint(0, 30000, (64,)).tolist())
    # warm up on a throwaway request so the graph and allocator are ready
    warm = eng.add_request(list(ids), SamplingParams(max_tokens=8, temperature=0.0))
    while warm.state != 3:
        eng.step()
    seq = eng.add_request(ids, SamplingParams(max_tokens=64, temperature=0.0))
    while seq.state != 3:
        eng.step()
    torch.cuda.synchronize()

    itl = statistics.median(seq.itls) * 1e3 if seq.itls else float("nan")

    n = 32
    graph = eng.graphs[(1, "short")]
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        graph.replay()
    torch.cuda.synchronize()
    replay_ms = (time.perf_counter() - t0) / n * 1e3

    print("engine step (median itl) %.3f ms  (%.0f tok/s)" % (itl, 1e3 / itl))
    print("bare graph replay        %.3f ms  (%.0f tok/s)" % (replay_ms, 1e3 / replay_ms))
    print("host overhead            %.3f ms  (%.1f%% of the step)"
          % (itl - replay_ms, 100 * (itl - replay_ms) / itl))
    llm.shutdown()


if __name__ == "__main__":
    main()
