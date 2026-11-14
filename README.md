# Kaze

Triton で書いた LLM 推論エンジンです。1 枚の GPU で、重みの読み出し帯域を使い切るところを狙っています。
PyTorch はテンソル管理と重みの読み込みにだけ使い、計算はすべて自前の Triton カーネルと CUDA Graph で動きます。

対象は Llama / Qwen2 / Qwen2.5 / Qwen3 / Mistral 系の dense decoder です（GQA、RoPE、RMSNorm、SwiGLU、
qk-norm に対応しています）。重みは safetensors と旧形式の .bin の両方を読みます。

## できること

- weight-only 量子化（int8 / int4、group-wise）。量子化のための学習は不要です
- ページド KV キャッシュと continuous batching。終わった系列から即座に回収します
- chunked prefill。長いプロンプトを分割して、生成中の系列と混ぜて流せます
- プレフィックス共有。同じシステムプロンプトを使う複数リクエストで KV を共有します
- 投機的デコード。プロンプト内の繰り返しと、生成済みの記録からの外挿で候補を作り、まとめて検証します
- OpenAI 互換の HTTP サーバを備えています（chat / completions / SSE ストリーミング）

## 速度

RTX 3070（8 GB）での実測です。量子化は int4（lm_head は int8）、greedy、プロンプトは 48 トークン。

| モデル | decode | prefill |
|---|---|---|
| Qwen2.5-0.5B | 636 tok/s | 40,000 tok/s |
| Qwen2.5-1.5B | 273 tok/s | 13,700 tok/s |
| Qwen2.5-3B | 167 tok/s | 7,550 tok/s |
| Qwen2.5-7B | 64 tok/s | 3,180 tok/s |

5 回測った中央値で、ばらつきは 0.2% 程度です。decode は 1 トークンあたりの重み読み出しがそのまま
速度になるので、量子化のビット数を下げるほど速くなります。同時に複数のリクエストを流しても
重みの読み出しは 1 回で済むため、1 本あたりの速度はほとんど落ちません。

長い文脈では、0.5B で 8K が 446 tok/s、32K が 257 tok/s、64K が 165 tok/s です（KV は fp16）。
YaRN の倍率を上げると 128K までニードル（長文中に埋め込んだ情報の recall）が通ります。

## インストール

```bash
git clone https://github.com/sahenjp/kaze && cd kaze
pip install -e .
```

モデルは初回に Hugging Face から自動で取ってきます。量子化は読み込み時にその場で行うので、
変換済みのファイルを用意する必要はありません。

```bash
python -m kaze serve --model Qwen/Qwen2.5-0.5B-Instruct --kind int4 --port 8000
```

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"俳句を一つ"}],"stream":true,"max_tokens":64}'
```

コマンドラインから 1 回だけ生成する場合:

```bash
python -m kaze generate --model Qwen/Qwen2.5-0.5B-Instruct --kind int4 "The capital of France is"
```

Python から使う場合:

```python
from kaze.llm import LLM

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", kind="int4", lm_head_kind="int8")
for text, done in llm.stream_text("Write a haiku about GPUs.", max_tokens=48):
    print(text, end="", flush=True)
llm.shutdown()
```

8 GB のカードで 7B を動かす場合は `--cache-dtype fp8v` を付けてください。KV の K は fp16 のまま、
V だけ fp8 にすることでメモリを 25% 減らせます。

## 仕組み

**decode は帯域律速です。** batch=1 の decode では、重みを 1 回読む時間がそのまま 1 トークンの時間に
なります。そのため GEMV は、計算ではなくメモリ帯域が律速になるように作っています。int4 の重みは
ニブルを展開しながら列ごとのスケールを掛けて fp16 に戻します。出力次元が小さい層（down_proj）は
CTA が足りず SM を埋められないので、K 方向に分割して部分和を別カーネルで足しています。

**prefill は計算律速です。** こちらは int4 の重みを int8 に展開し、活性も int8 にして
int8 テンソルコアで積みます。fp16 で 34 TFLOPS だったものが 47 TFLOPS になり、prefill 全体で
1.5 倍ほど速くなります。

**KV キャッシュ**は 16 トークン単位のブロックで管理し、系列ごとにブロック表を持ちます。
プレフィックスが一致する系列は同じブロックを共有し、参照がゼロになったら回収します。
KV の型は fp16、fp8、fp8v（V だけ fp8）から選べます。

**投機的デコード**は、直前の文脈と一致する系列をプロンプトと生成履歴から探し、続きのトークンを
候補にします。JSON やコードのように繰り返しの多い入力では 3 倍以上速くなり、自由文ではほとんど
効きません。候補が当たり続けているかを常に測っていて、当たらないときは通常の decode に戻ります。

## 4bit が 2 倍速くならない理由

重みを int4 にすると、読むバイト数は int8 の半分になります。それでも速度は 1.35 倍ほどにしか
ならないのは、ニブル展開のコストで実効帯域が落ちるためです。実測では、int4 の GEMV は層によって 52〜172 GB/s、
int8 は 210〜363 GB/s でした。要素あたり shift・mask・cvt・減算・スケール乗算の 5 演算が増えるので、
帯域が余っていても追いつかない場面が出ます。とくに出力次元が小さい層（qkv、o_proj）は CTA が
足りず、この差がそのまま出ます。

prefill では int4 の重みを int8 に展開する作業が加わりますが、int8 テンソルコアで積めるぶん全体では
速くなります。詳しい実測は [docs/evidence.md](docs/evidence.md) にあります。

## テスト

```bash
python3 bench/verify.py
```

カーネルの一致検査、prefill と decode の一致、投機的デコードの受理判定、量子化の組み合わせごとの
出力チェック、HTTP 経由の E2E、ランダム負荷の stress テスト、主要ベンチマークをまとめて実行し、
しきい値を下回れば非ゼロの終了コードで終了します。テスト用のモデルは `KAZE_MODELS` 環境変数で
指定します（既定は `~/models`）。

## 制約

- 単体 GPU のみで、マルチ GPU や分散実行には未対応です
- dense decoder のみです。MoE のコードはありますが、手元にモデルがなく実測できていません
- sliding window を持つモデルも全 attention として計算します（学習時の窓は超えられません）
- 8 GB のカードでは 7B が上限です。14B 以上は載りません
- 量子化は学習不要な方式のみで、GPTQ や AWQ のような復元計算は行いません
