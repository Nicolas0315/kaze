# Kaze

Triton で書いた LLM 推論エンジンです。1 枚の GPU で、重みの読み出し帯域を使い切るところを狙っています。
PyTorch はテンソル管理と重みの読み込みにだけ使い、計算はすべて自前の Triton カーネルと CUDA Graph で動きます。

対象は Llama / Qwen2 / Qwen2.5 / Qwen3 / Mistral 系の dense decoder です（GQA、RoPE、RMSNorm、SwiGLU、qk-norm に対応しています）。
重みは safetensors と旧形式の .bin の両方を読みます。

## できること

- weight-only 量子化（int8 / int4、group-wise）。量子化のための学習は不要です
- ページド KV キャッシュと continuous batching。系列ごとにブロックを割り当て、終わった系列から即座に回収します
- chunked prefill。長いプロンプトを分割して、生成中の系列と混ぜて流せます
- プレフィックス共有。同じシステムプロンプトを使う複数リクエストで KV を共有します
- 投機的デコード。プロンプト内の繰り返しと、生成済みの記録からの外挿で候補を作り、まとめて検証します
- OpenAI 互換の HTTP サーバを備えています（/v1/chat/completions、/v1/completions、SSE ストリーミング）

## 実測値

RTX 3070（8 GB）での実測です。量子化は int4（lm_head は int8）、greedy、プロンプトは 48 トークン。

| モデル | decode | prefill | 同時 32 本 | 同時 64 本 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 636 tok/s | 40,000 tok/s | 14,199 tok/s | 21,039 tok/s |
| Qwen2.5-1.5B | 273 tok/s | 13,700 tok/s | — | — |
| Qwen2.5-3B | 167 tok/s | 7,550 tok/s | — | — |
| Qwen2.5-7B | 64 tok/s | 3,180 tok/s | 16 本で 1,023 tok/s | — |

同じ条件で 5 回測った中央値で、ばらつきは 0.2% 程度です。decode は 1 トークンあたりの重み読み出しが
そのまま速度になるので、バッチを増やしても 1 本あたりの速度はほとんど落ちません（7B は 1 本 64 tok/s のまま
16 本で 1,023 tok/s）。逆に言うと、batch=1 の速度は量子化のビット数でほぼ決まります。

長い文脈では、0.5B で 8K が 446 tok/s、32K が 257 tok/s、64K が 165 tok/s（KV は fp16）。
YaRN の倍率を上げると 128K までニードル（長文中に埋め込んだ情報の recall）が通ります。

7B を 8 GB のカードで動かす場合は `--cache-dtype fp8v` を使ってください。KV の K は fp16 のまま、
V だけ fp8 にすることで、メモリを 25% 減らせます。この設定なら 7B で 16K 文脈が載ります。

## 使い方

```bash
pip install torch triton transformers safetensors fastapi uvicorn
```

コマンドラインから生成する場合:

```bash
python -m kaze generate --model Qwen/Qwen2.5-0.5B-Instruct --kind int4 "The capital of France is"
```

サーバとして動かす場合:

```bash
python -m kaze serve --model Qwen/Qwen2.5-0.5B-Instruct --kind int4 --port 8000
```

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"俳句を一つ"}],"stream":true,"max_tokens":64}'
```

Python から直接使う場合:

```python
from kaze.llm import LLM

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", kind="int4", lm_head_kind="int8")
for text, done in llm.stream_text("Write a haiku about GPUs.", max_tokens=48):
    print(text, end="", flush=True)
llm.shutdown()
```

## 仕組み

**decode のカーネル。** batch=1 の decode は、重みを 1 回読む時間がそのまま 1 トークンの時間になります。
そのため、GEMV は計算ではなくメモリ帯域が律速となるよう最適化しています。int4 の重みはニブルを展開しながら、列ごとのスケールを掛けて
fp16 に戻します。出力次元（n）が小さい層（down_proj）は CTA が足りず SM を埋められないので、K 方向に分割して
部分和を別カーネルで足しています。分割数は実測で決めていて、4 分割より 8 分割のほうが 0.5B で 4% 速くなります。

**prefill のカーネル。** prefill は計算律速なので、int4 の重みを int8 に展開し、活性も int8 に量子化して
int8 テンソルコアで積みます。fp16 で 34 TFLOPS だったものが 47 TFLOPS になり、prefill 全体で 1.5 倍ほど速くなります。
タイルは 32×64 が最良で、64×64 にするとレジスタが溢れて逆に遅くなります。

**KV キャッシュ。** 16 トークン単位のブロックで管理し、系列ごとにブロック表を持ちます。
プレフィックスが一致する系列は同じブロックを共有し、参照がゼロになったら回収します。
KV の型は fp16、fp8（K と V の両方）、fp8v（V だけ fp8）から選べます。

**投機的デコード。** 直前の文脈と一致する系列をプロンプトと生成履歴から探し、続きのトークンを候補にします。
JSON やコードのように繰り返しの多い入力では 3 倍以上速くなり、自由文ではほぼ効きません。
候補が当たり続けているかを常に測っていて、当たらないときは通常の decode に戻ります。

**量子化。** 重みを 128 要素ごとにグループ化し、グループ内の最大値からスケールを決めます。
活性は量子化しません（W8A8 は decode では帯域律速で効かず、prefill でのみ int8 を使います）。
品質は参照実装との KL で測っていて、int8 で 0.0003、int4 で 0.03〜0.09 です。
lm_head を int8 にすると int4 の品質劣化が半分ほどに収まります。

## テスト

```bash
python3 bench/verify.py
```

カーネルの一致検査、prefill と decode の一致、投機的デコードの受理判定、量子化の組み合わせごとの
出力チェック、HTTP 経由の E2E、ランダム負荷の stress テスト、主要ベンチマークをまとめて実行し、
しきい値を下回れば非ゼロの終了コードで終了します。モデルのパスは `KAZE_MODELS` 環境変数で指定します。

個別に走らせる場合:

```bash
python3 tests/test_kaze.py        # カーネルと推論の正しさ
python3 tests/test_stress.py 24   # ランダムな同時実行
python3 tests/test_smoke.py       # 量子化の組み合わせごとの出力
python3 tests/test_server.py      # HTTP の E2E
```

## 制約

- 単体 GPU のみ。マルチ GPU や分散は未対応です
- dense decoder のみ。MoE のコードはありますが、手元にモデルがなく実測できていません
- sliding window を持つモデルも、全 attention として計算します（学習時の窓は超えられません）
- 8 GB のカードでは 7B が上限です。14B 以上は載りません
- 量子化は学習不要な方式のみで、GPTQ や AWQ のような復元計算は行いません
