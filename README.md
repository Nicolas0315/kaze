# kaze

8GB の GPU 1 枚で、Qwen2.5 の 0.5B / 1.5B / 3B と 7B が動く LLM 推論エンジン。

- 継続バッチング（同時リクエストをまとめて処理）、ページド KV キャッシュ、プレフィックス共有
- OpenAI 互換 HTTP サーバ（chat / completions / streaming）
- 投機的デコード（prompt-lookup、外部ドラフトモデル不要、既定 ON）
- 量子化: int4 / int8 / fp16（group 64 と 128）
- KV キャッシュ dtype: fp16 既定、fp8v で 25% 削減
- 対応モデルと形式: Llama / Qwen2 / Qwen2.5 / Qwen3 / Mistral 系 dense decoder（GQA, RoPE, RMSNorm, SwiGLU, qk-norm）、safetensors と旧 .bin の両方を読む

## 速度

計測条件: RTX 3070 8GB、int4、llama.cpp と同一条件。

### 単一ストリーム

decode は 128 トークン生成、prefill は 512 トークン。

| model | decode | llama.cpp | prefill | llama.cpp |
|---|---|---|---|---|
| 0.5B | 870.9 tok/s | 492.7 | 48,429 tok/s | 22,881 |
| 1.5B | 334.9 | 263.1 | 18,557 | 12,735 |
| 3B | 210.0 | 168.5 | 9,145 | 7,020 |
| 7B | 87.9 | 87.7 | 4,426 | 3,383 |

### 同時実行

HTTP サーバ同士の比較。プロンプト 32 トークン、生成 64 トークン、prefill 込み。

| model | 32 本 | llama.cpp | 64 本 | llama.cpp |
|---|---|---|---|---|
| 0.5B | 9,170 tok/s | 1,204 | 14,046 | 1,382 |
| 1.5B | 4,870 | 1,077 | 6,429 | 1,238 |
| 3B | 3,009 | 948 | 4,248 | 1,097 |
| 7B | 1,651 | 1,162 | 2,146 | 起動しない |

### 長文脈

- 8 本同時（4,096 トークン × 8、kaze 対 llama.cpp、単位 tok/s）:
  - 0.5B: 1,533 対 207
  - 1.5B: 959 対 110
  - 3B: 704 対 66
  - 7B: 276.6 tok/s（llama.cpp が起動しない領域）
- 文脈長:
  - 0.5B: 262,144 トークンまで needle テスト正解（`--factor 16`）
  - 1.5B: 200,000 まで
  - 7B: 16K まで（fp8v）

### 投機的デコード

単一ストリームでの実測値。

- 0.5B（1,024 トークン生成）: 5,401 tok/s
- 4K 文脈での倍率: 0.5B 4.18 倍、3B 6.33 倍、7B 10.66 倍

## 品質

- KL（fp16 基準）:
  - int8: 0.0003
  - int4: 0.0108（0.5B）、0.0041（1.5B）、0.045（3B）
  - int4 group 64: 0.0057（0.5B、速度 -3%）
- needle recall: 2K / 4K / 8K / 16K / 128K / 262K すべて正解
- 同じ入力で 2 回実行すると出力が一致する

## 使い方

HTTP サーバ起動:

```bash
python -m kaze serve --model <path> --kind int4 --port 8000
```

`/v1/chat/completions` へのリクエスト:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "..."}]}'
```

生成のみ:

```bash
python -m kaze generate --model <path> --max-tokens 96 "..."
```

Python API は `kaze.LLM`。

## 検証

```bash
./bench/verify_all.sh
```

実行項目と結果:

- テスト 191 件
- needle 4 段
- KV 逼迫（全 200 応答）
- 64 本 × 8 波の連続負荷（メモリ増加なし、例外 0）
- 出力: `ALL VERIFICATION PASSED`

## 制限

- 7B の単一 decode は llama.cpp と同等（87.9 対 87.7）
- group は 64 と 128 のみ。32 は int4 decode が落ちるため拒否する
- KV を fp8 にすると長距離の復唱が壊れる（fp8v は可）
- MoE は実装済みだが、8GB に載る実用サイズがない
- MLA / Gemma3 / Phi3 は未対応
- 14B 以上は 8GB に載らない

## 実装の補足

カーネル実装には Triton を使用する。実行には CUDA Graph を併用する。

decode は重みの読み出し帯域で決まる。実効値は 0.5B で 223 GB/s（ピークの 50%）、7B で 320 GB/s（ピークの 71%）。

層あたりの転送量が大きいほど DRAM 効率が上がるため、大きいモデルほどピークに近づく。

## 詳細

- 全セルと再現コマンド: [docs/final_table.md](docs/final_table.md)
- 測定方法・却下した案: [docs/evidence.md](docs/evidence.md)
- カーネル設計: [docs/kernels.md](docs/kernels.md)
- MoE の現状: [docs/moe.md](docs/moe.md)
