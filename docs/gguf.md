# GGUF 入力対応（stage A、2026-11-16）

kaze に GGUF ファイルを読む入口を足した。GGUF の量子化ブロックを fp16 に戻し、
kaze 自身の int4 / int8 レイアウトへ再量子化して、既存の safetensors 経路と同じ
`KazeModel` を組み立てる。**GGUF の量子化形式のまま推論するのではなく、
読み込み時に一度だけ展開する**（stage A）ので、実行時経路は今までと変わらない。

## 使い方

```python
from kaze.loader import load_model

model = load_model(
    "bench/llamacpp/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    kind="int8", group=128,
    tokenizer_dir="/models/Qwen2.5-0.5B-Instruct",
)
```

`LLM` から使う場合は既存の `tokenizer_path` を使う（トークナイザは GGUF からはまだ読まない）。

```python
LLM("..../Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    tokenizer_path="/models/Qwen2.5-0.5B-Instruct")
```

- `path` は `.gguf` ファイルでもよいし、`.gguf` を 1 つだけ含むディレクトリでもよい
  （`config.json` とトークナイザを同居させたディレクトリを想定）。
- `tokenizer_dir` を省いた場合、`.gguf` と同じディレクトリに `config.json` があれば使う。
  無ければ GGUF のメタデータだけから `ModelConfig` を作る。
- 設定は **GGUF のメタデータを正**とし、`config.json` があるときは
  `vocab_size / hidden_size / intermediate_size / num_hidden_layers /
  num_attention_heads / num_key_value_heads / head_dim` が一致するか検証する
  （不一致は `ValueError`）。rope 設定は GGUF から完全には復元できないため、
  `config.json` があればそちらの `rope_scaling` を使う。

## 対応している ggml 形式

| id | 形式 | ブロック | １ブロックのバイト数 | 実ファイルでの使用例 |
|---|---|---|---|---|
| 0 | F32 | 1 | 4 | norm、attn の bias |
| 1 | F16 | 1 | 2 | （今回は現物なし） |
| 2 | Q4_0 | 32 | 18 | — |
| 3 | Q4_1 | 32 | 20 | — |
| 6 | Q5_0 | 32 | 22 | Qwen2.5-0.5B の attn_q/k/output、ffn_gate/up |
| 7 | Q5_1 | 32 | 24 | — |
| 8 | Q8_0 | 32 | 34 | Qwen2.5-0.5B の token_embd、attn_v |
| 12 | Q4_K | 256 | 144 | Llama-3.2-1B / Qwen2.5-1.5B 以降の大半 |
| 13 | Q5_K | 256 | 176 | — |
| 14 | Q6_K | 256 | 210 | ffn_down、attn_v（Q4_K_M の「M」部分） |
| 15 | Q8_K | 256 | 292 | — |
| 30 | BF16 | 1 | 2 | — |

「使用例」は手元の Q4_K_M 7 本を走査した実測（Qwen2.5-0.5B は
F32 121 / Q5_0 132 / Q8_0 13 / Q4_K 12 / Q6_K 12、
Llama-3.2-1B は F32 34 / Q4_K 96 / Q6_K 17）。

## 対応していない形式（黙って値を返さず例外にする）

| 形式 | 扱い |
|---|---|
| Q2_K (10) / Q3_K (11) | 未実装。読み込み時に `ValueError` |
| IQ1_S / IQ2_* / IQ3_* / IQ4_* | 未実装（格子探索が要る）。`ValueError` |
| TQ1_0 / TQ2_0 | 未実装。`ValueError` |
| MXFP4 / Q4_0_4_4 系 | 未実装。`ValueError` |
| MoE（`expert_count > 0`） | 未対応。モデル構築前に明示エラー（llama.cpp は `ffn_*_exps` を 3 次元で持つため分解が要る） |
| 分割 GGUF（`*-00001-of-0000N.gguf`） | 未対応。1 ファイル前提 |

理由: 現物の Q4_K_M は Q4_K / Q5_0 / Q6_K / Q8_0 だけで構成されていたので、
まずこの 4 形式を C 実装と一致させた。未対応形式は「間違った数値を返す」のではなく
読み込み時に落ちる。

## テンソル名の対応

| GGUF | kaze（HF 名） |
|---|---|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `output.weight` | `model.lm_head.weight`（無ければ tied） |
| `blk.N.attn_norm.weight` | `model.layers.N.input_layernorm.weight` |
| `blk.N.ffn_norm.weight` | `model.layers.N.post_attention_layernorm.weight` |
| `blk.N.attn_q / attn_k / attn_v / attn_output` | `self_attn.q_proj / k_proj / v_proj / o_proj` |
| `blk.N.attn_q.bias` など | `self_attn.*_proj.bias` |
| `blk.N.attn_q_norm / attn_k_norm` | `self_attn.q_norm / k_norm` |
| `blk.N.ffn_gate / ffn_up / ffn_down` | `mlp.gate_proj / up_proj / down_proj` |
| `blk.N.attn_qkv.weight`（融合） | q/k/v に分割（q_size, kv_size, kv_size 行） |
| `blk.N.ffn_gate_up.weight`（融合） | gate/up に分割 |
| `rope_freqs.weight` | 使わない（下記の制限を参照） |

kaze は q/k/v と gate/up を別テンソルとして受け取って自分で連結するので、
融合テンソルはローダ側で行方向に切り分ける。

### llama アーキの q/k は行が並べ替えられている

llama.cpp の HF→GGUF 変換は、**llama アーキのときだけ** `q_proj` / `k_proj` の
行を ggml のインターリーブ配置に並べ替えて書き出す（`conversion/llama.py` の
`LlamaModel.permute`、頭の中で 2 行ずつ交互に並ぶ形）。そのため GGUF の q/k を
そのまま kaze に渡すと、RoPE が回す次元の対応がずれて出力が壊れる。
ローダ側で同じ置換を掛け直して HF の head-major 配置に戻している。

```python
# w.reshape(n_head, head_dim, 2).swapaxes(1, 2) ではなく、
# GGUF 側は (head_dim/2, 2) で並んでいるのでこちらが逆変換
w.reshape(n_head, head_dim // 2, 2, ...).swapaxes(1, 2).reshape(w.shape)
```

| テンソル（Llama-3.2-1B） | 置換前 cos | 置換後 cos |
|---|---|---|
| `blk.0.attn_q.weight` | 0.0214 | 0.9974 |
| `blk.0.attn_k.weight` | 0.0405 | 0.9971 |

置換するのは `llama` アーキだけで、`mistral` / `llama4` / `apertus` は変換側が
`undo_permute = False`、qwen 系は元々並べ替えない（現物の Qwen2.5 で cos 0.9989 を確認）。
古い変換スクリプトで作った GGUF はこの限りでないので、その場合は q/k の cos を
一度確かめるのが早い。

## 検証（実行したものだけ）

### 1. ggml の C 実装とのビット一致

llama.cpp のビルド済み `libggml-base.so` の `dequantize_row_*` を ctypes で呼び、
実ファイルのブロック（テンソル中間の 2 行）と比較した。結果は **全形式で完全一致**。

| テンソル | 形式 | max abs diff |
|---|---|---|
| `blk.3.ffn_down.weight`（Qwen2.5-0.5B） | Q4_K | 0.000e+00 |
| `blk.0.attn_k.weight`（Qwen2.5-0.5B） | Q5_0 | 0.000e+00 |
| `blk.0.ffn_down.weight`（Qwen2.5-0.5B） | Q6_K | 0.000e+00 |
| `token_embd.weight`（Qwen2.5-0.5B） | Q8_0 | 0.000e+00 |
| `blk.0.attn_k.weight`（Llama-3.2-1B） | Q4_K | 0.000e+00 |
| `token_embd.weight`（Llama-3.2-1B） | Q6_K | 0.000e+00 |

### 2. 逆量子化した重みと HF の fp16 重みの一致（Qwen2.5-0.5B Q4_K_M）

| テンソル | 形式 | cosine | 相対 L2 |
|---|---|---|---|
| `blk.0.attn_q.weight` | Q5_0 | 0.998863 | 0.0472 |
| `blk.0.ffn_gate.weight` | Q5_0 | 0.999025 | 0.0436 |
| `blk.0.attn_v.weight` | Q8_0 | 0.999985 | 0.0058 |
| `blk.0.ffn_down.weight` | Q6_K | 0.999921 | 0.0188 |
| `blk.3.ffn_down.weight` | Q4_K | 0.997423 | 0.0733 |
| `token_embd.weight` | Q8_0 | ~1.000000 | 0.0056 |
| `output_norm.weight` | F32 | 1.000000 | 0.0000 |

（相対 L2 の大きさは llama.cpp の量子化誤差そのもので、読み違いではない。
Q8_0 < Q6_K < Q5_0 < Q4_K の順に誤差が増える。）

### 3. ロジット比較（Qwen2.5-0.5B、どちらも int8 / group 128）

39 トークンの prefill で全位置のロジットを比較。

| 指標 | 値 |
|---|---|
| 最終行の cosine | 0.994813 |
| 位置ごとの cosine 平均 | 0.982669 |
| 位置ごとの cosine 最小 | 0.727710 |
| argmax 一致率 | 0.949（39 位置中） |
| greedy 16 トークン | 7 トークン目で分岐 |

Llama-3.2-1B（Q4_K_M、どちらも int8 / group 128、config は HF 側）でも同じ比較をした。

| 指標 | 値 |
|---|---|
| 最終行の cosine | 0.992241 |
| 位置ごとの cosine 平均 | 0.989263 |
| 位置ごとの cosine 最小 | 0.955675 |
| argmax 一致率 | 0.900（40 位置中） |
| greedy 16 トークン | 11 トークン目まで一致、その後分岐 |

同じ safetensors を fp16 で読んだ場合と int8 で読んだ場合でも同じ比較をした（差し引き用の基準値）。

| 比較 | 最終行 cosine | 平均 | 最小 | argmax 一致 |
|---|---|---|---|---|
| safetensors fp16 vs safetensors int8 | 0.999616 | 0.999256 | 0.990809 | 0.974 |
| safetensors int8 vs GGUF(Q4_K_M) int8 | 0.994813 | 0.982669 | 0.727710 | 0.949 |

kaze 自身の int8 化はここまで素直（0.9996）なので、上の差は
GGUF（Q4_K_M）の重み量子化ぶんと読める。

最小 0.7277 の 1 点は **先頭トークン（文脈が 1 トークンだけ）** で、
top-1 は両方 ` following` のまま、2 位以下（` ` vs ` CB`/` SMART`、ロジット 12 前後）が
入れ替わった箇所だった。分布が尖っている位置では cosine が敏感になるだけで、
argmax は一致している（39 位置中 2 位置だけ不一致）。

### 4. llama.cpp との突き合わせ（同じ GGUF・同じ chat テンプレ・temperature 0）

`llama-completion`（`-ngl 99 --temp 0 --top-k 1`）と kaze で同じプロンプトを回すと、
最初の 6 トークン（`The history of computing is indeed fascinating`）が一致し、
その次で分岐した。kaze 側は GGUF を int8 に再量子化しているので、
llama.cpp の Q4_K_M 実行とは計算精度が違う。

### 5. ロード時間（参考、0.5B）

| 経路 | 時間 |
|---|---|
| Qwen2.5-0.5B safetensors → int8 | 8.2 s |
| Qwen2.5-0.5B GGUF（Q4_K_M）→ int8 | 14.5 s |
| Llama-3.2-1B safetensors → int8 | 9.8 s |
| Llama-3.2-1B GGUF（Q4_K_M）→ int8 | 10.7 s |

GGUF はブロック展開のぶん遅い。容量は 954 MB → 380 MB になる。

## テスト

```
python3 tests/test_gguf.py        # 31 tests, 約 1.3 秒（GPU 不要）
```

- `dequantize_blocks` を**手計算した値**で検証（Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 /
  Q4_K / Q5_K / Q6_K / Q8_K）。6 ビット scale/min の詰め方、5 番目のビットプレーン、
  Q6_K の 16 値ごとの尺度切り替えなど、壊れやすい箇所を個別に固定している。
- 合成 GGUF をテスト内で書き出し、v2 / v3 のヘッダ、メタデータ（文字列・配列・
  bool・float）、alignment、テンソル表、切り詰め、未知の型、過大な配列長を検証。
- 名前マッピング、`attn_qkv` の分割、`config.json` との不一致検出、
  CPU 上での `load_model` 組み立て（1 層の合成モデル）を検証。
- llama の q/k 置換の往復（変換側の式 → `unpermute_qk` → 元に戻る）、
  llama アーキの合成 GGUF で q/k が head-major に戻ること、
  qwen アーキでは並べ替えないことを検証。
- 実ファイルのテストは `bench/llamacpp/models` に現物があるときだけ実行（`skipUnless`）。

なお、テストを書く過程で **1 行だけのテンソルで `view(dtype)` が
`storage_offset` 制約で落ちる**バグが見つかり、バイト列を明示的にコピーする実装に直した。

## 制限

- トークナイザは GGUF 内蔵のもの（`tokenizer.ggml.tokens`）をまだ読まない。
  HF ディレクトリを用意する必要がある。
- llama3 系の rope scaling は GGUF に周波数テーブル（`rope_freqs.weight`）として
  入るため、`config.json` が無いと復元できない。その場合は警告を出して
  `rope_theta` だけの素の RoPE で動く（生成は壊れる）。
- GGUF の vocab 行数と HF `config.json` の `vocab_size` が違う場合はエラーにする
  （padding 行の扱いを決めていない）。
- chat template や sampling 既定値など、メタデータの一部は使っていない。
- MoE と分割 GGUF は未対応（上表）。
