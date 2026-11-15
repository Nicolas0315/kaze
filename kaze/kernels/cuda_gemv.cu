// 実験中: 手書き CUDA の int4 GEMV（エンジンには未接続）
// 現状: 数値は正しい（PyTorch 参照と cos 1.000000、relerr 2.3e-4）。
//       速度は 76 GB/s で Triton 版（328 GB/s）に負け（ALU 律速）。
// ツールチェーン（nvcc 13.4 + torch 2.15/cu132 + gcc 16）が通ることの確認が目的。
// 改善には 30B 単位のベクトル読みと half2/dp4a 化が要る。


#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// 1 行分の int4(group 128) GEMV。1 ワープが 1 行、4 レーンで 1 グループ(64B) を読む。
// パッキングは 32bit ワードに 8 個の重み（ニブル j が k = 8*w + j）。
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
    const int sub = lane & 3;
    const int gslot = lane >> 2;
    const uint8_t* wrow = wq + (size_t)row * (size_t)(K >> 1);
    const __half* srow = scales + (size_t)row * (size_t)ngrp;
    float acc = 0.f;
    for (int g = gslot; g < ngrp; g += 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(wrow + (size_t)g * 64 + (size_t)sub * 16);
        const __half* xg = x + (size_t)g * 128 + (size_t)sub * 32;
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
        float s = 0.f;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t wd = words[wi];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int nib = (int)((wd >> (4 * j)) & 0xFu) - 8;
                s += (float)nib * __half2float(xg[8 * wi + j]);
            }
        }
        s += __shfl_down_sync(0xffffffffu, s, 1, 4);
        s += __shfl_down_sync(0xffffffffu, s, 2, 4);
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
