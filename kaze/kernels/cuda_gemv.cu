
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// int4(group 128) GEMV（実験・エンジン未接続）。
// 1 ワープ = 1 行、8 レーンで 1 グループ(64B) を 8B ずつ読み、half2 で積和する。
// 実測: 276 GB/s（Triton 版 332 GB/s）／数値は PyTorch 参照と cos 1.000000。
// 16B 読み（4 レーン/グループ）にすると 150 GB/s まで落ちるので、8B 読みが今の最良。
__global__ void gemv1_kernel(const __half* __restrict__ x,
                             const uint8_t* __restrict__ wq,
                             const __half* __restrict__ scales,
                             __half* __restrict__ y,
                             int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= N) return;
    const int ngrp = K >> 7;
    const int sub = lane & 7;
    const int gslot = lane >> 3;
    const uint8_t* wrow = wq + (size_t)row * (size_t)(K >> 1);
    const __half* srow = scales + (size_t)row * (size_t)ngrp;
    float acc = 0.f;
    for (int g = gslot; g < ngrp; g += 4) {
        const uint2 v = *reinterpret_cast<const uint2*>(wrow + (size_t)g * 64 + (size_t)sub * 8);
        const __half* xg = x + (size_t)g * 128 + (size_t)sub * 16;
        const uint32_t words[2] = {v.x, v.y};
        half2 acc2 = __float2half2_rn(0.f);
#pragma unroll
        for (int wi = 0; wi < 2; ++wi) {
            const uint32_t wd = words[wi];
            const __half* xw = xg + 8 * wi;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n0 = (int)((wd >> (8 * j)) & 0xFu) - 8;
                const int n1 = (int)((wd >> (8 * j + 4)) & 0xFu) - 8;
                const half2 wv = __halves2half2(__int2half_rn(n0), __int2half_rn(n1));
                const half2 xv = *reinterpret_cast<const half2*>(xw + 2 * j);
                acc2 = __hfma2(wv, xv, acc2);
            }
        }
        const float2 f2 = __half22float2(acc2);
        float s = f2.x + f2.y;
        s += __shfl_down_sync(0xffffffffu, s, 1, 8);
        s += __shfl_down_sync(0xffffffffu, s, 2, 8);
        s += __shfl_down_sync(0xffffffffu, s, 4, 8);
        if (sub == 0) acc += s * __half2float(srow[g]);
    }
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) y[row] = __float2half(acc);
}

torch::Tensor gemv1(torch::Tensor x, torch::Tensor wq, torch::Tensor scales, int64_t n) {
    const int N = (int)n;
    const int K = (int)(x.size(1));
    auto y = torch::empty({N}, x.options());
    gemv1_kernel<<<(N + 7) / 8, 256>>>(reinterpret_cast<const __half*>(x.data_ptr()),
                                       reinterpret_cast<const uint8_t*>(wq.data_ptr()),
                                       reinterpret_cast<const __half*>(scales.data_ptr()),
                                       reinterpret_cast<__half*>(y.data_ptr()), N, K);
    return y;
}
