# MoE / MTP / 4bit の現状（2026-11-14）
> **注意（2026-11-16 追記）**: この文書は 2026-11-14 時点の記録です。当時は int8 が主体で、
> 現在の int4 本体＋int4 ヘッド構成とは前提が異なります。
> **現在の測定値と再現手順は [final_table.md](final_table.md) を参照してください。**
> 本文は経緯（何を試し、何を否定したか）の記録として残しています。


## MoE: 実装してテストまで通した

対応したもの:

- **ルーター + top-k + 選択エキスパートのみ読む decode**（`kaze/kernels/quant_linear.py: moe_gemv`）
- **prefill はエキスパートごとに 1 回だけ重みを読む GEMM**（`_moe_prefill`）
- 重みの持ち方は 2 通り:
  - fused（`mlp.experts.gate_up_proj` [E, 2I, H] + `mlp.experts.down_proj` [E, H, I] + `mlp.router.weight`）… Qwen3-MoE 形式
  - per-expert（`mlp.experts.{e}.gate_proj/up_proj/down_proj`）… Mixtral / Qwen2-MoE 形式
  - エキスパートは 1 本の `QuantLinear`（n = E*N）に詰めており、`row_off` で切り替える
- cuDNN 依存なし。decode は**帯域律速**（選択された k 個のエキスパートだけを流す）なので、
  tensor core ではなく FMA リダクションで十分速い

検証（`tests/test_kaze.py`、12/12 通過）:

| test | 内容 | 結果 |
|---|---|---|
| `moe.block_matches_manual` | top-k ルーティング込みの MoE 出力を素の torch 実装と比較 | 一致（差 1e-6） |
| `moe.decode_equals_prefill` | decode 経路と prefill 経路の logits を比較 | cos 1.0、top-1 全一致 |

**注意**: 8GB の GPU では実用 MoE（Qwen3-30B-A3B など）は載らないため、
検証は tiny-random-Qwen3MoE（hidden 64 / 4 エキスパート / 2 層）で行っています。

## MTP: 既存の投機的デコードが該当、モデル側 MTP ヘッドは未対応

- Kaze の **n-gram 自己ドラフト**（`kaze/engine.py`）は「複数トークンを予測して 1 回の forward で検証する」
  という MTP と同じ形の実装です。greedy・1 本のときに有効で、反復の多い出力で 1.1〜1.4 倍。
- **モデルが持つ MTP ヘッド（DeepSeek-V3 系の `mtp.*`）の読み込みは未実装**です。
  実装するなら「mtp ブロックを 1 層追加で読み、draft を 1〜2 トークン出して既存の verify 経路に流す」形になりますが、
  検証用の重みが手元に無いため、未検証のコードは入れていません。

## 4bit で llama.cpp に勝てていない理由と、勝つための具体案

実測（0.5B, decode）:

| 形式 | 1 トークンで流す重み | 実測 | 実効帯域 |
|---|---|---|---|
| Kaze int8 | 502 MB | 456 tok/s | 229 GB/s |
| llama.cpp Q4_K_M | 約 290 MB | 496 tok/s | 143 GB/s |
| llama.cpp Q8_0 | 約 500 MB | 438 tok/s | 234 GB/s |

**同じ 8bit なら Kaze が上**（456 vs 438）。負けるのは 4bit のときだけで、
理由は「バイト数が 1.7 倍」＋「Triton の int4 unpack が 1 バイトあたり 2 倍遅い」の合わせ技です。

int4 GEMV の実測: 4.49 MB を 28 µs（161 GB/s）。int8 は 8.85 MB を 24.7 µs（359 GB/s）。
要素数は同じなので、**nibble 展開の ALU が効いています**（現在 1 要素あたり shift + and + cvt + sub + mul の 5 命令）。

勝つための具体策（未実装）:

1. **重みを Marlin 型に事前 repack**（`[K/16, N, 16]` の断片順に並べ替え、スケールも一緒に畳む）と
   `prmt.b32` + `cvt.rn.f16x2` で nibble→fp16 を 1/2 命令に落とす。Triton なら
   `tl.inline_asm_elementwise(..., pack=4)` で書ける。見込み +40〜60%
2. 上と合わせて **8*s を事前計算**し `fma` 1 命令にする（-20% ALU）
3. 目標: int4 で 300 GB/s 超。0.5B なら 269 MB / 300 GB/s = 0.9 ms + 非 GEMV 0.7 ms = 1.6 ms = **625 tok/s**
   （llama.cpp Q4_K_M の 496 を上回る）

## round 32: MoE 経路の実バグを修正し、恒久テストを追加（17/17）

tiny-qwen3moe（2層・4エキスパート・top-2）で MoE 経路を通したところ、
`weight_bytes()` が落ちた:

```
AttributeError: 'NoneType' object has no attribute 'nbytes'
```

MoE 層は `wgate_up`/`wdown` を持たず、エキスパートを `w_gu_e`/`w_down_e` に積んでいる。
従来の実装は dense MLP を前提にしていたため、**MoE モデルでは必ず落ちる**状態だった
（ベンチの表示も VRAM 見積りも通らない）。修正して MoE 側のバイト数も数えるようにした。

### 数値の検証（torch 参照実装との比較）

| 経路 | cos | 最大誤差 |
|---|---|---|
| MoE prefill | **0.999999** | 0.00000 |
| MoE decode | **0.999999** | — |

同じ router logits から同じ top-k を使った参照実装と一致。
（HF との比較で top-1 が 0% になるのは、この合成モデルがランダム初期化で
logits がほぼ一様なため。エキスパート選択が同点で揺れるので指標にならない）

### 追加したテスト

`moe.weight_bytes` — MoE モデルで重み集計が通ること（dense 前提の回帰を防ぐ）。
これで **17/17 passed**。

### 参考: MoE の速度（合成の極小モデルなので参考値）

| 構成 | prefill | decode |
|---|---|---|
| int8 | 161,117 tok/s | 2,900 tok/s |
| int4 | 158,653 tok/s | 2,856 tok/s |