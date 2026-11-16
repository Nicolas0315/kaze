# Kaze — 帯域律速で動く LLM 推論エンジン

Kaze は「batch=1 の decode も、長い prefill も、HBM 帯域の限界まで引き出す」ことを目的に
フルスクラッチで書いた推論エンジンです。単発の生成で終わらず、継続バッチング・ページド KV
キャッシュ・プレフィックス共有・OpenAI 互換 API まで入った実用品として動きます。

- 対象: **Llama / Qwen2 / Qwen2.5 / Qwen3 / Mistral 系**の dense decoder（GQA, RoPE, RMSNorm, SwiGLU, qk-norm, llama3 rope, sliding window は全 attention として扱う）
- 重みは safetensors と旧形式の .bin の両方を読む（Capybara 系の fine-tune は .bin が多い）
- バックエンド: Triton カーネル + CUDA Graph（PyTorch は tensor 管理とローダのみ）
- 量子化: int8 / int4 / fp16（weight-only, group-wise, 学習不要）。KV キャッシュは fp16 既定、
  `--cache-dtype fp8v` で K を fp16 のまま V だけ fp8 にして **メモリ25%削減**（品質劣化なし）
- **全体像は [docs/SUMMARY.md](docs/SUMMARY.md)** に 1 ページでまとめています
- **llama.cpp との全指標対戦表は [docs/final_table.md](docs/final_table.md)**（単一 decode / prefill /
  32・64 本同時 / 4K×8 / 16K×4 の全セルで上回り、7B の高並列と長文脈は llama.cpp が起動できない領域）
- 実測（RTX 3070 8GB、int4 本体＋int4 ヘッド・greedy・投機 OFF）:
  **decode 0.5B 870.9 / 1.5B 334.9 / 3B 210.0 tok/s**、**prefill 0.5B 48.4k / 1.5B 18.6k / 3B 9.1k tok/s**
  （`bench/compare.py`、128 トークン生成・512 トークン prefill、2026-11-16 再測定）、
  リクエスト全体（prefill 込み・32 本同時到着・プロンプト 32 tok・生成 64 tok）は
  **0.5B 9,170 / 1.5B 4,870 / 3B 3,009 / 7B 1,651 tok/s**（64 本なら 0.5B 14,046 / 7B 2,146）。
  ハーネスごとに条件が違う数字は混ぜません。比較は必ず [docs/final_table.md](docs/final_table.md) の
  同一条件の表を使ってください。
  長文は 0.5B int4 で 16K **508 tok/s**、**262K（`--factor 16`）までニードル recall 正解**。1.5B は 200K まで（262K は 8GB に載らず）
- 8GB のカードで **7B も動く**: 単一ストリーム decode **88 tok/s**（llama.cpp Q4_K_M と**互角**、prefill は +31%）・16K 文脈（`--cache-dtype fp8v` で KV 25% 削減）。
  なお 16K 文脈×16 本は KV だけで 25 GB（fp8v でも 19 GB）になるため 8GB には載りません。
  実測できた並列は 4K 文脈 × 8 本で **277 tok/s**（llama.cpp は同条件で起動できず）
- 比較（llama.cpp 同 GPU・相手は従来の実測値）: decode は int4 で全サイズ上
  **同一モデル・同一 GPU 状態で数分差の対戦**（llama-bench Q4_K_M、2026-11-16 の涼しい時間帯）で、
  decode は **+77% / +27% / +25% / ±0%**、prefill は **+112% / +46% / +30% / +31%**（0.5B / 1.5B / 3B / 7B）。
  **7B の decode は互角**（以前 +14% と書いたのは相手が throttled だった比較）
  同時実行は **サーバ同士**で同一条件（HTTP・プロンプト 32 tok・生成 64 tok・prefill 込み）:
  32 本で 0.5B **7.62 倍** / 1.5B **4.52 倍** / 3B **3.17 倍** / 7B **1.42 倍**、
  64 本で 0.5B **10.2 倍** / 1.5B **5.19 倍** / 3B **3.87 倍** / 7B 2,146 tok/s（llama 起動不可）。
  長文脈 × 並列は 0.5B 4K×8 で **7.4 倍**、7B 4K×8 は **llama.cpp が起動できない**領域（kaze 276.6 tok/s）。
  （参考: tail 修正前の同じ測定は 5,021 / 2,286 / 1,516 / 773 tok/s で、7B だけ負けていた）
  投機的デコード（prompt-lookup・既定 ON・単一ストリーム）: サーバ経由で 0.5B **5,401 tok/s**
  （1,024 トークン生成・コード入力）、4K 文脈でも **2,691 tok/s（plain の 4.18 倍）**、
  7B は 512 生成で **1,472 tok/s**。効かない入力では自動で通常 decode に戻ります（0.99〜1.01x）
- 品質: int8 は KL 0.0003（実質無損失）、int4 は KL 0.034（1.5B/3B）〜0.093（0.5B）。
  品質のレバーは **`--group 64`** が最も効率的（0.5B の KL 0.0108 → **0.0057（−47%）**、
  速度 −3%）。`--lm-head-kind int8` は KL −12% に対して単一ストリーム −39% なので割に合わず、
  両方入れると KL −67%（最高品質）。1.5B 以上では group 64 の差はほぼありません
- 検証: **ワンコマンドで再現** `bench/check_all.sh`（unittest 167 件・量子化スモーク・ストレス・
  決定性・epilogue 畳み込み・系列混在 attention・batched prefill 一致・投機一致・needle 4K + ベンチ）、
  unit/integration **24/24**（fp16/int4 × tdim 1/16 の decode 一致・融合 rope+attention の bit 一致・投機と複数ストリームの混在を含む）、
  compute-sanitizer の memcheck が int4 / int8 / MoE の 3 経路で **0 errors**、
  ランダム負荷の stress test **40 試行で失敗 0**、
  量子化の組み合わせごとの出力スモークテスト（**group 64/128 × int4/int8/mixed**）、
  サーバの HTTP E2E（chat / completions / streaming）
  → **`./bench/verify_all.sh` で全部通しで確認**（正しさ + needle 2K/4K/8K/16K + KV 逼迫 +
  8 波×64 本の連続負荷 + ベンチ。`ALL VERIFICATION PASSED` が出れば合格）
- すべての主張・測定方法・未解決点は **docs/evidence.md** に集約

## すぐ動かす

```bash
pip install torch triton transformers safetensors fastapi uvicorn
python -m kaze serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"俳句を一つ"}],"stream":true,"max_tokens":64}'
```

Python API:

```python
from kaze import Engine, SamplingParams          # 低レベル（自分で step を回す）
from kaze.llm import LLM                          # 高レベル（スレッドが step を回す）

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", kind="int8", group=128, lm_head_kind="int8")
print(llm.chat([{"role": "user", "content": "1+1 は？"}],
               SamplingParams(max_tokens=32, temperature=0.0)).text)
```

CLI:

```bash
python -m kaze generate --model ./models/Qwen2.5-0.5B-Instruct "東京の名所を3つ" --max-tokens 96
python -m kaze bench    --model ./models/Qwen2.5-0.5B-Instruct     # prefill/decode/同時実行
python tests/test_kaze.py                                          # GPU カーネル・モデル統合テスト
```

モデルの読み込みが不要な回帰テスト（CUDA がある環境ではカーネル試験も実行）:

```bash
python -m pip install httpx                                        # HTTP テスト用（開発時のみ）
python -m unittest discover -s tests -p "test_*regressions.py"
python -m unittest discover -s tests -p "test_*audit.py"
python tests/test_worker_failure.py
```

Colab（T4 / A100）は [notebooks/kaze_colab.ipynb](notebooks/kaze_colab.ipynb) を開いて上から実行。

## 実測値（RTX 3070 8GB、2026-11-16 の涼しい時間帯に同一条件で取得）

完全な表と再現コマンドは **[docs/final_table.md](docs/final_table.md)**。ここは要点だけ。

| model | decode（batch 1） | llama.cpp tg128 | prefill 512 | llama.cpp pp512 |
|---|---|---|---|---|
| Qwen2.5-0.5B | **870.9 tok/s** | 492.7 | **48,429 tok/s** | 22,881 |
| Qwen2.5-1.5B | **334.9** | 263.1 | **18,557** | 12,735 |
| Qwen2.5-3B | **210.0** | 168.5 | **9,145** | 7,020 |
| Nous-Capybara-7B | **87.9** | 87.7 | **4,426** | 3,383 |

同時実行（HTTP サーバ同士・プロンプト 32 tok・生成 64 tok・prefill 込み）:

| model | 32 本 | llama.cpp | 64 本 | llama.cpp |
|---|---|---|---|---|
| 0.5B | **9,170 tok/s** | 1,204 | **14,046** | 1,382 |
| 1.5B | **4,870** | 1,077 | **6,429** | 1,238 |
| 3B | **3,009** | 948 | **4,248** | 1,097 |
| 7B | **1,651** | 1,162 | **2,146** | 起動不可 |

長文脈 × 並列（4,096 tok × 8 本）は 0.5B **1,533** vs 207、3B **704** vs 66（7B は llama.cpp が
起動できない領域で kaze 276.6 tok/s）。投機的デコードは 4K 文脈で 0.5B 4.18x / 3B 6.33x /
7B **10.66x**。

## 量子化の選び方

| 目的 | 指定 | KL（0.5B） | 速度 |
|---|---|---|---|
| 最速（既定） | `--kind int4`（ヘッドも int4） | 0.0108 | decode 870.9 tok/s |
| 品質を少し上げる | `--group 64` | **0.0057** | −3% |
| 品質を最大化 | `--group 64 --lm-head-kind int8` | **0.0036** | 単一 −40% |
| ほぼ無損失 | `--kind int8` | 0.0003 | decode 600.8（int4 の 69%） |

- **group 64 は小モデルで効く**（0.5B は KL 半減、1.5B 以上はほぼ差なし）
- `--lm-head-kind int8` 単体では KL −12% に対して単一ストリーム −39% なので、単独では割に合わない
- **group は 64 と 128 のみ**（32 は int4 decode が違法アクセスで落ちるため入口で拒否）
- KV キャッシュは既定 fp16、`--cache-dtype fp8v` で 25% 削減（**fp8 は長距離 recall が壊れるので警告**）

## 設計上の位置づけ

- **decode は帯域律速**。int4 は 0.5B で 1 トークンあたり 254 MB を流すので、870 tok/s は
  実効 223 GB/s（ピーク 448 の 50%）。7B は 3.56 GB/token で 320 GB/s（71%）と、
  **大きいモデルほどピークに近い**（1 層あたりの転送が大きいほど DRAM 効率が上がる）
- **prefill は計算律速**。int8/int4 の整数テンソルコア + 活性量子化の融合で、llama.cpp の
  1.3〜2.1 倍
- **同時実行は継続バッチング + ページド KV + プレフィックス共有**。32〜64 本で 1.4〜10 倍
- 7B の単一 decode だけ llama.cpp と互角（両者とも帯域の壁に当たっている）

### 動作確認したモデル（このリポジトリで実測）

| model | arch | 状態 |
|---|---|---|
| Qwen2.5-0.5B / 1.5B / 3B-Instruct | qwen2 | 実測済み（logit cos 0.9997〜0.99999） |
| Qwen3-0.6B / 1.7B | qwen3（qk-norm） | 実測済み（cos 0.999998） |
| Llama-3.2-1B-Instruct | llama（llama3 rope） | 実測済み（cos 0.9998） |
| TinyLlama-1.1B-Chat | llama | 実測済み |
| Qwen2.5-1.5B-Capybara（SFT チューン） | qwen2 | 実測済み |
| Nous-Capybara-7B-V1.9 | mistral（.bin 重み） | **実測済み**: int4 で 5.40 GB 常駐・decode 87.9 tok/s・16K 文脈（round 205 で 6 モデル一斉確認） |
| Qwen3-MoE 系 / Mixtral 系 | MoE（router + top-k） | 実装・テスト済み（tiny モデルで検証）。8GB では実用サイズは載らない |

未対応: MLA（DeepSeek-V2/V3 の attention）、Gemma3、Phi3、モデル側 MTP ヘッド。
MoE の現状と 4bit で勝つための具体策は [docs/moe.md](docs/moe.md)。

## 設計


### 1. decode は「重みを 1 回だけ流す」カーネル

decode の 1 トークンは、どの層でも M=1 の行列ベクトル積です。計算はほぼ無く、律速は
「重みを何バイト読むか」だけ。そこで

- 重みは `int8/int4 + group-wise scale` で保持し、GEMV の中で展開する（メモリに書き戻さない）
- 出力 N が小さい層（qkv, o_proj, down_proj）は **split-K**（K 方向に分割）で CTA を増やし、
  SM を埋めて帯域を出す。部分和は **atomic ではなく固定順の総和**（結果が再現する）
- 残差加算と SwiGLU は epilogue に融合（活性の往復を消す）
- RoPE と KV キャッシュへの書き込みは 1 カーネル（`rope_scatter`）

### 2. decode は CUDA Graph で丸ごと再生

batch=1 では 1 トークンあたり ~250 個のカーネルが出ます。Python から launch すると
それだけで 1 トークン 1ms 以上かかるので、バッチサイズごと（16/32/64 行）に
**モデル 1 ステップ全体を CUDA Graph に焼き**、入力を静的な tensor に書いて replay します。
実測で eager 24.5 ms → graph 5.9 ms（最適化前）→ 現在 2.4 ms。

### 3. prefill は paged flash attention + dequant GEMM

- prefill はページド KV を直接読む causal flash attention（Triton, GQA 対応）。
  KV を連続領域に gather しないので、層あたり 2 回のコピーが消えます。
- 量子化重みのままタイル GEMM（dequant on the fly, fp16 tensor core）
- RoPE + KV 書き込みは decode と同じ 1 カーネル

### 4. 実用品としての中身（batch=1 で終わらない）

- **継続バッチング**: 毎ステップ、走っている列の decode をまとめ、空きに prefill を詰める
- **ページド KV キャッシュ**: 16 トークンのブロック、ブロックテーブル、参照カウント
- **プレフィックス共有**: ブロック単位のハッシュで共通プロンプトの KV を再利用（LRU 追い出し）
- **チャンク化 prefill**: `max_batched_tokens` 予算で長いプロンプトを分割し、decode を飢えさせない
- **プリエンプション**: KV が尽きたら新しい列を捨てて再計算（プレフィックスは再利用）
- **OpenAI 互換 API**: `/v1/chat/completions`, `/v1/completions`（stream 対応）, `/v1/models`, `/health`, `/metrics`
- **決定的**: split-K の部分和を固定順にし、同じ入力なら同じ出力（greedy）

## 量子化と品質

logit 一致度（fp16 の transformers と比較、4 プロンプト、Qwen2.5-0.5B）:

| 設定 | 重み | logit cos | top-1 一致 |
|---|---|---|---|
| int8 g=128 + lm int8 | 774 MB | 0.99975 | 4/4 |
| int4 g=64 + lm int8 | 603 MB | 0.95947 | 4/4 |
| int4 g=128 + lm int4 | 527 MB | 0.93542 | 4/4 |
| fp16 | 1.5 GB | 1.00000 | 4/4 |

**既定は int8**。理由は品質だけではなく速度で、RTX 3070 の Triton カーネルでは
4bit の nibble 展開（shift/mask/convert）が余分なバイト削減を上回り、
int4 の方が遅くなるためです（gate_up: int8 24 µs vs int4 27 µs、同じ層）。
メモリが厳しいときは `--kind int4 --group 64` を選べます。

## 構成

```
kaze/
  config.py     HF config.json の読み込み
  loader.py     safetensors → 量子化済み重み（層ごとにストリーム）
  quant.py      int4/int8 のパック表現（K 方向 8 nibble = 1 int32）
  model.py      prefill / decode の forward、CUDA Graph 用の静的バッファ
  cache.py      ブロックアロケータ、ページド KV、プレフィックスハッシュ
  engine.py     継続バッチング・チャンク prefill・プリエンプション・グラフ再生
  sampler.py    temperature / top-k / top-p / repetition penalty
  llm.py        スレッドで engine を回す高レベル API
  server.py     OpenAI 互換 HTTP
  kernels/
    quant_linear.py  GEMV（split-K 込み）/ タイル GEMM
    attn.py          ページド decode attention（flash-decoding）と prefill attention
    fused.py         RoPE + KV scatter
    norm.py          RMSNorm
    act.py           SwiGLU, 埋め込み gather
bench/run.py    prefill/decode/同時実行のベンチ
bench/kernel_lab.py  カーネル単体を CUDA Graph 内で測る（形状別スイープ、roofline 比つき）
bench/rows_ab.py     batch=1 専用経路の A/B（同一プロセス内で交互に測る）
bench/steady_decode.py  同時 N 本の decode だけを取り出す（prefill を含めない）
bench/prefill_lab.py    512 トークン prefill の内訳（カーネル別）
tests/          量子化・カーネル・モデル一致・MoE・スケジューラ・API/スレッド処理の回帰テスト
docs/design.md  実装メモ（何を測って何を選んだか、踏んだバグ）
```

## 制限

- 未対応: MLA（DeepSeek-V2/V3 の attention）、Gemma3 / Phi3、モデル側 MTP ヘッド
- マルチ GPU・テンソル並列は未対応（1 GPU 前提）
- サンプリングは temperature/top-k/top-p/repetition/presence penalty（grammar 拘束なし）。反復ペナルティは `1e-6 <= repetition_penalty <= 1e6`、presence penalty は `-2 <= presence_penalty <= 2`
- 未対応のarchitecture・活性化・dynamic NTK RoPE・非2冪head_dimなどは、黙って別の計算をせず読み込み時にエラーを返す
- HTTP API は生成長を正の整数、temperature を有限の非負値、top_p を `0 < top_p <= 1` として検証し、不正な入力には HTTP 422 を返す
- `/v1/completions` は単一プロンプトのみ（文字列または要素 1 個の配列）。空配列・複数プロンプトの配列・トークン列が空になる入力は HTTP 422 を返す
- `seed` 指定時はリクエストごとに乱数状態を保持する。同じモデル・実行条件・seed で、他リクエストのサンプリングに左右されない
- `repetition_penalty != 1` または `presence_penalty != 0` では、全トークン履歴へペナルティを適用し、投機的デコードを使わず通常経路で処理する
- テキスト出力はstop文字列を除去し、ストリーミング中の未完成UTF-8文字・stop接頭辞は確定するまで保留する
- KV容量を超えるプロンプトは受付時にエラーとする。生成中のKV容量不足はそのリクエストだけを終了し、他のリクエストへ波及させない
- 推論スレッドが異常終了した場合、待機中の呼び出しにも失敗を通知する。生成 API と `/health` は HTTP 503、開始済みストリームは error イベントと `[DONE]` を返す
- 投機的デコードは n-gram 自己ドラフトのみ（greedy・batch 1。モデル側 MTP ヘッドは未対応）

## ライセンス

MIT。