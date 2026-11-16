"""In-graph sweep of the fused int4 swiglu (gate_up) decode kernel at the 7B shape.

Nous-Capybara-7B: nh=14336, k=4096, int4 packed rows 2048 B, gate/up streams 29 MB apart.
The kernel is 60% of the 7B decode step at 223 GB/s, so this is the main target.
Timing: a captured CUDA graph of 40 launches is replayed with CUDA events, so numbers
are in-graph and stable (eager sweeps misled three times: rounds 59, 126, 140).
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kaze.kernels.quant_linear import gemv_int4_swiglu_kernel

torch.manual_seed(0)
dev = "cuda"
NH, K = 14336, 4096
x = (torch.randn(1, K, device=dev) * 0.1).half()
wq = torch.randint(-128, 127, (2 * NH, K // 2), dtype=torch.int8, device=dev)
sc = torch.rand(2 * NH, K // 128, dtype=torch.float16, device=dev) * 0.01
out = torch.empty(1, NH, dtype=torch.float16, device=dev)

def launch(bn, nw, ns):
    gemv_int4_swiglu_kernel[( (NH + bn - 1) // bn, )](
        x, wq, sc, out, NH, K, BLOCK_N=bn, BLOCK_K=128, GROUP=128, TDIM=1,
        num_warps=nw, num_stages=ns)

def time_cfg(bn, nw, ns, reps=40):
    # warm/compile outside capture
    for _ in range(3):
        launch(bn, nw, ns)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(reps):
                launch(bn, nw, ns)
    torch.cuda.synchronize()
    e1 = torch.cuda.Event(True); e2 = torch.cuda.Event(True)
    e1.record()
    for _ in range(5):
        g.replay()
    e2.record()
    torch.cuda.synchronize()
    return e1.elapsed_time(e2) / 5 / reps * 1e3   # us per launch

TILES = [(128, 8, 4), (128, 8, 5), (128, 4, 4), (128, 4, 5), (64, 8, 4),
         (64, 4, 4), (64, 4, 5), (128, 8, 3), (64, 8, 5), (32, 4, 4)]
nbytes = NH * K  # each output column reads k/2 bytes of gate + k/2 of up
res = {t: [] for t in TILES}
for r in range(3):
    order = TILES if r % 2 == 0 else TILES[::-1]
    for t in order:
        res[t].append(time_cfg(*t))
base = statistics.median(res[TILES[0]])
print("baseline (128,8,4): %.1f us  (%.0f GB/s)" % (base, nbytes / (base / 1e3) / 1e9))
for t in TILES:
    med = statistics.median(res[t])
    print("  %-12s %6.1f us  (%+.1f%%)  %.0f GB/s" % (str(t), med,
          100 * (med - base) / base, nbytes / (med / 1e3) / 1e9))
