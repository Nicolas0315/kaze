# kaze

8GB の GPU 1 枚で、Qwen2.5 の 0.5B / 1.5B / 3B と 7B が動く LLM 推論エンジン。

- 継続バッチング（同時リクエストをまとめて処理）、ページド KV キャッシュ、プレフィックス共有
- OpenAI 互換 HTTP サーバ（chat / completions / streaming）
- 投機的デコード（prompt-lookup、外部ドラフトモデル不要、既定 ON）
- 量子化: int4 / int8 / fp16（group 64 と 128）
- KV キャッシュ dtype: fp16 既定、fp8v で 25% 削減
- 対応モデルと形式: Llama / Qwen2 / Qwen2.5 / Qwen3 / Mistral 系 dense decoder（GQA, RoPE, RMSNorm, SwiGLU, qk-norm）、safetensors と旧 .bin の両方を読む
- **GGUF も読める**（Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0 / Q5_0 / Q8_K、F16 / F32）。読み込み時に kaze の int4 / int8 へ再量子化する
- **AMD（ROCm）でも動く**。バックエンドを検出してグラフ取得や fp8 KV を自動で切り替える

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

## 動作環境

NVIDIA GPU（CUDA）専用。CPU だけで推論する経路は持たない。

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA、compute capability 7.5 以上（int4 / int8 のテンソルコアを使うため）。動作確認は RTX 3070（sm_86, 8GB） |
| ドライバ | 620.70.06 で確認 |
| PyTorch | 2.15.0+cu132 |
| Triton | 3.9.0 |
| CUDA | 13.2（PyTorch のビルドに同梱） |

カーネルは起動時にその GPU 向けにコンパイルされるので、Ampere 以降の別のカードでも動く。ただしタイルとパラメータは 8GB / 448 GB/s 級（RTX 3070）で調整しており、帯域や SM 数が違うカードでは再調整の余地がある。

KV キャッシュの fp8v（V のみ fp8）は sm_86 でも動作を確認済み。fp8 ネイティブ命令の有無で速度は変わる。

## GGUF

量子化済みの単一ファイル（GGUF）をそのまま読める。読み込み時に ggml ブロックを fp16 に戻し、kaze の int4 group128 / int8 に詰め直す。

```bash
python -m kaze generate --model model-Q4_K_M.gguf --tokenizer <hf_dir> --kind int8 "..."
```

- 対応ブロック: Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q4_K / Q5_K / Q6_K / Q8_K / F16 / F32 / BF16
- 逆量子化は llama.cpp の実装とビット一致（実ファイルのブロックで max abs diff 0.000e+00）
- GGUF は重みだけなので、**トークナイザは HF ディレクトリから取る**（`--tokenizer`）
- 未対応: IQ* / TQ* / MXFP4 系、分割 GGUF、GGUF 内蔵の語彙

詳細と名前対応表は [docs/gguf.md](docs/gguf.md)。

## AMD（ROCm）

バックエンドは起動時に判定する。ROCm では hipGraph の capture が不安定なので**グラフは既定オフ**、fp8 KV は非対応なら fp16 に落とす。

```bash
python -m kaze serve --model <hf_dir> --kind int4 --backend rocm --port 8000
# kaze: backend=rocm device=... graphs=off fp8=unsupported cuda_graph=off cache_dtype=fp16
```

- カーネルは Triton なので ROCm でもコンパイルできるが、**タイルは sm_86 で調整した値**であり、AMD での速度は未測定
- グラフ無効の影響は CUDA 上で実測済み（tpot 1.16 ms → 5.23 ms、約 4.5 倍）

導入手順と注意は [docs/amd.md](docs/amd.md)。

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

## 決定性

同じ入力なら同じ出力を返す。ただし**バッチの行数が変わると出力が 1〜2 トークン変わることがある**。

- 既定は速度優先。行数（1 / 16 / 32 / 64）ごとにカーネル族と設定を切り替えるため、単独実行とバッチ内実行は同じ式の順序違いではなく**別の算術**になる
- 行数を固定したいときは `KAZE_DETERMINISTIC=1`（または `engine.set_deterministic()`）。バケットを 1 本に固定し、単独実行とバッチ内実行が bit 一致する
- コストは固定したバケット幅のぶん。`max_batch=16` で単一ストリーム 870.9 → 690.9 tok/s（−20.7%）。`max_batch` は想定同時数に合わせる
- 投機的デコードとバッチ prefill は行数で形が変わるため、決定性モードでは無効化される

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

## 実装

推論の経路は自前で書いている。PyTorch はモデルを呼ぶためではなく、テンソル確保と一部の要素演算にだけ使う。

自前（Python 6,601 行、うち Triton カーネル 35 本）:

- int4 / int8 の GEMV・GEMM、融合 SwiGLU、paged attention（prefill / decode）、split-K、RoPE 融合、RMSNorm と活性量子化の融合、epilogue の畳み込み
- 継続バッチング、KV ブロック割当、プレフィックス共有
- ページド KV キャッシュ（16 トークン / ブロック、fp16 / fp8v / fp8）
- group-wise 量子化（int4 / int8 / 層ごとの mixed）
- 投機的デコード（prompt-lookup のドラフト、受理判定、受理率 EMA による自動フォールバック）
- OpenAI 互換サーバ、ワーカースレッド、`/metrics`
- 重み読み込み（safetensors と旧 .bin）

ライブラリ:

| 依存 | 用途 |
|---|---|
| PyTorch | テンソル確保、fp16 重みの `F.linear`、MoE の softmax / topk、サンプラの topk / multinomial、CPU 参照実装 |
| Triton | カーネルのコンパイラ（カーネル本体は自作） |
| transformers | トークナイザと config の読み取り |
| fastapi / uvicorn | HTTP 層 |
| safetensors | 重みファイルの読み取り |

CUDA Graph は decode の起動オーバーヘッドを消すために使う。

decode は重みの読み出し帯域で決まる。実効値は 0.5B で 223 GB/s（ピークの 50%）、7B で 320 GB/s（ピークの 71%）。

層あたりの転送量が大きいほど DRAM 効率が上がるため、大きいモデルほどピークに近づく。

## 詳細

- 全セルと再現コマンド: [docs/final_table.md](docs/final_table.md)
- 測定方法・却下した案: [docs/evidence.md](docs/evidence.md)
- カーネル設計: [docs/kernels.md](docs/kernels.md)
- MoE の現状: [docs/moe.md](docs/moe.md)
