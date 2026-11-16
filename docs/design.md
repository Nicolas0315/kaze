# 実装メモ — 何を測って、何を選んだか

対象 GPU: RTX 3070 8 GB（GA104, sm_86, HBM 448 GB/s, SM 46）
モデル: Qwen2.5-0.5B-Instruct（24 層, hidden 896, GQA 14/2, vocab 152k, tied embedding）

## 1. 律速の見積り

decode 1 トークンで流れる重み:

- 本体 = 24 層 × (qkv 1.0 + o 1.0 + gate_up 8.7 + down 4.4) MB … int8 で 24 × 16.6 = 400 MB
- lm_head = 152k × 896 × 1 B = 136 MB（埋め込み表は行参照なので流れない）
- 合計 536 MB → 448 GB/s なら **1.20 ms/token = 836 tok/s** が理論上限

実測は 2.38 ms（421 tok/s, 実効 211 GB/s）。差の内訳は「カーネル間の隙間」と
「小さい N の GEMV が CTA 不足で帯域を出せない」こと。

## 2. カーネル選択の実測

### int4 と int8、速いのはどちらか

Triton で重みを nibble 展開する場合、要素あたり shift / mask / cvt / 減算 / scale 乗算の
5 演算が要る。実測（同一形状・GPU 時間、CUDA Graph で launch を除去）:

| 層 | int4 (group 64) | int8 (group 128) |
|---|---|---|
| qkv 1152×896 | 10.5 µs (52 GB/s) | **5.0 µs (210 GB/s)** |
| o 896×1152 | 9.4 µs | **6.0 µs** |
| gate_up 9728×896 | 26.9 µs (172 GB/s) | **24.4 µs (363 GB/s)** |
| down 896×4864 | 21.6 µs | **14.8 µs** |
| lm_head 152k×896 | 299 µs | 337 µs |

バイト数が半分でも int4 の方が遅い。reshape を使わない 8-dot 版（nibble ごとに K/8 の
小さい dot を 8 回）も試したが 172 GB/s で同じ、down ではさらに悪化した。
したがって既定は int8。int4 はメモリ制約時の選択肢として残す。

### split-K と決定性

N が小さい層は CTA が足りず帯域が出ない（qkv は N=1152 / BN=32 で 36 CTA）。
K 方向に 8 分割して CTA を 8 倍にし、部分和を **atomic ではなく固定順の総和**で
まとめることで、速度を保ったまま bit 再現性を確保した（atomic_add 版は実行ごとに
argmax が変わり、テストが落ちた）。

### 融合

| 融合 | 削減 |
|---|---|
| RoPE + cos/sin gather + KV scatter → 1 カーネル | 層あたり ~17 起動 → 1 |
| 残差加算 → GEMV epilogue | 層あたり 1 起動 + 活性の往復 |
| SwiGLU → 専用カーネル（未実装、次段） | 68 µs/step 見込み |

### CUDA Graph

decode 1 ステップは ~250 カーネル。Python から launch すると 1 トークン 24.5 ms、
グラフ replay なら 5.9 ms（融合前）→ 2.4 ms（融合後）。
バケットは 16 / 32 / 64 行で、実バッチをその中に詰めて再生する。
埋め行は `seq_len=1` を入れて NaN を避ける（0 だと 0 除算になる）。

## 3. 踏んだバグ（同じ轍を踏まないために）

1. **KV キャッシュを層で共有していた** — 24 層が同じ領域に書いていたため、attention が
   最終層の KV を読んでいた。層ごとに [L, blocks, bs, kvh, d] を持つよう変更。
2. **SDPA の causal マスクは q_len < kv_len で期待通りに揃わない** — チャンク化 prefill で
   q_len=13, kv_len=61 のとき、bottom-right でも top-left でもないマスクになっていた
   （logit cos 0.75）。自前のページド prefill attention に置き換えて解決（cos 1.0）。
3. **engine を 2 スレッドから step していた** — ベンチが LLM のバックグラウンドスレッドと
   同時に `engine.step()` を呼び、同じ列が 2 回サンプリングされた。step をロックで直列化し、
   `LLM(..., start_loop=False)` を追加。
4. **量子化グループをまたぐ K タイル** — BLOCK_K が GROUP より大きいと scale が 1 個しか
   当たらず精度が崩れた（cos 0.47）。BLOCK_K ≤ GROUP を強制。
5. **int8 用の GEMM が int4 カーネルに落ちていた** — 分岐漏れで沈黙して壊れた。テストで
   カーネル別に数値一致を確認するようにした。

## 4. 測り方

```bash
python -m kaze bench --model ./models/Qwen2.5-0.5B-Instruct --prompt-len 64 --gen-len 128
```

- decode tok/s = 1 / (出力トークン間隔の平均)。TTFT は別途。
- 実効帯域 = 1 トークンで流れる重みバイト（埋め込み表を除く） / トークン間隔。
- prefill tok/s = プロンプト長 / TTFT。
- 同時実行は n 本を一斉投入し、合計出力トークン / 壁時計。

比較対象の llama.cpp は同じ GPU でビルド（commit 61dbf49, CUDA 13.5, sm_86）し、
`llama-bench -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 99`。
生ログは `bench/llamacpp/llama-bench.txt`。

## 5. 次にやるなら

1. **SwiGLU 融合**（gate_up の GEMV 内で silu(g)*u を書く）: -68 µs/step
2. **投機的デコード（n-gram 自己ドラフト）**: 重みを 1 回流して最大 k+1 トークンを検証。
   反復の多い出力（コード・表・リスト）で 1.5〜2 倍。検証は prefill 経路（M=k）で行い、
   greedy なら受理判定は argmax 一致のみ。temperature > 0 は棄却サンプリングが要る。
3. **prefill の GEMM チューニング**（BLOCK_M/N/K, stages）: 現在 39 TFLOP/s（fp16 ピークの 48%）
4. **int4 を tensor core で扱う**（Marlin 型のレイアウト + ldmatrix）: 4bit で帯域 2 倍
5. **マルチ GPU / テンソル並列**、**MoE / MLA 対応**
