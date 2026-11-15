"""tile-major 配置の重みを使う decode 用カーネル（2-D の重みも併せて保持する前提）。

量子化形式（int4・group 128）はそのままで、メモリ上の並びだけ [K/G, N, G/2] にしたもの。
prefill は従来の 2-D 重みを使うので、両方を持てば経路を分けられる。
"""
import triton
import triton.language as tl


@triton.jit
def gemv_swiglu(X, WQ, SC, Y, NH, K, BLOCK_N: tl.constexpr, G: tl.constexpr,
                TDIM: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, TDIM)
    offs_b = tl.arange(0, G // 2)
    g_acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    u_acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    row = 2 * NH * (G // 2)
    for kb in range(0, K // G):
        base = WQ + kb * row
        bg = tl.load(base + offs_n[:, None] * (G // 2) + offs_b[None, :])
        bu = tl.load(base + (offs_n[:, None] + NH) * (G // 2) + offs_b[None, :])
        sg = tl.load(SC + offs_n * (K // G) + kb)
        su = tl.load(SC + (offs_n + NH) * (K // G) + kb)
        xs = tl.load(X + offs_m[:, None] * K + kb * G + tl.arange(0, G)[None, :])
        xt = tl.trans(xs)
        xsum = tl.sum(xs, axis=1)
        ig = bg.to(tl.int32)
        iu = bu.to(tl.int32)
        wg = tl.interleave((ig & 0xF).to(tl.float16), ((ig >> 4) & 0xF).to(tl.float16))
        wu = tl.interleave((iu & 0xF).to(tl.float16), ((iu >> 4) & 0xF).to(tl.float16))
        g_acc += (tl.dot(wg, xt) - 8.0 * xsum[None, :]) * sg[:, None]
        u_acc += (tl.dot(wu, xt) - 8.0 * xsum[None, :]) * su[:, None]
    out = g_acc * tl.sigmoid(g_acc) * u_acc
    tl.store(Y + offs_m[None, :] * NH + offs_n[:, None], out.to(tl.float16))

@triton.jit
def gemv(X, WQ, SC, Y, N, K, BLOCK_N: tl.constexpr, G: tl.constexpr, TDIM: tl.constexpr):
    """tile-major 配置の重みで 1 行分の GEMV。"""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, TDIM)
    offs_b = tl.arange(0, G // 2)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for kb in range(0, K // G):
        base = WQ + kb * N * (G // 2)
        b = tl.load(base + offs_n[:, None] * (G // 2) + offs_b[None, :])
        s = tl.load(SC + offs_n * (K // G) + kb)
        bi = b.to(tl.int32)
        w = tl.interleave((bi & 0xF).to(tl.float16), ((bi >> 4) & 0xF).to(tl.float16))
        xs = tl.load(X + offs_m[:, None] * K + kb * G + tl.arange(0, G)[None, :])
        acc += (tl.dot(w, tl.trans(xs)) - 8.0 * tl.sum(xs, axis=1)[None, :]) * s[:, None]
    tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16))
