# Kaze 最終サマリ（2026-11-16 時点）

RTX 3070 8GB / 単一 GPU で動く、フルスクラッチの LLM 推論エンジン。
Triton カーネル + CUDA Graph、継続バッチング、ページド KV、プレフィックス共有、
OpenAI 互換サーバ、投機的デコード（prompt-lookup）まで入っています。

## 1. 数値（llama.cpp との同一条件対戦）

単一ストリーム（両者 128 トークン生成・ignore_eos / prefill 512）:

| model | decode | llama.cpp tg128 | prefill | llama.cpp pp512 |
|---|---|---|---|---|
| Qwen2.5-0.5B | **870.9** | 492.7（+77%） | **48,429** | 22,881（+112%） |
| Qwen2.5-1.5B | **334.9** | 263.1（+27%） | **18,557** | 12,735（+46%） |
| Qwen2.5-3B | **210.0** | 168.5（+25%） | **9,145** | 7,020（+30%） |
| Nous-Capybara-7B | **87.9** | 87.7（±0%） | **4,426** | 3,383（+31%） |

同時実行（HTTP サーバ同士・プロンプト 32 tok・生成 64 tok・prefill 込み）:

| model | 32 本 | llama.cpp | 64 本 | llama.cpp |
|---|---|---|---|---|
| 0.5B | **9,170** | 1,204（7.6x） | **14,046** | 1,382（10.2x） |
| 1.5B | **4,870** | 1,077（4.5x） | **6,429** | 1,238（5.2x） |
| 3B | **3,009** | 948（3.2x） | **4,248** | 1,097（3.9x） |
| 7B | **1,651** | 1,162（1.4x） | **2,146** | 起動不可 |

長文脈 × 並列（4,096 tok × 8 本）: 0.5B **1,533** vs 207（7.4x）、1.5B **959** vs 110（8.7x）、
3B **704** vs 66（10.7x）、7B **277**（llama.cpp は起動できない）。

長文脈の単一ストリーム（0.5B int4）: 1K 807 / 4K 676 / 8K 631 / 16K 511 tok/s。
**128K までニードル recall 正解**（YaRN ×8）。

投機的デコード（単一ストリーム・既定 ON）: 0.5B 1,024 生成で **5,401 tok/s**、
4K 文脈で 0.5B 4.18x / 3B 6.33x / 7B **10.66x**（llama.cpp 単一ストリーム 77 tok/s の 19 倍）。

**19 セル中 18 セル勝ち、1 セル（7B 単一 decode）が互角**。

## 2. 品質

| 項目 | 値 |
|---|---|
| int8 | KL 0.0003（実質無損失） |
| int4（0.5B / 1.5B / 3B） | KL 0.0108 / 0.0041 / 0.045（fp16 基準、複数文脈） |
| int4 group 64（0.5B） | KL **0.0057**（−47%）、速度 −3% |
| 長距離 recall | needle 2K/4K/8K/16K すべて HIT、128K も HIT（YaRN ×8） |
| 決定性 | 同一入力で bit 一致（テストで担保） |

## 3. 検証（ワンコマンド）

```
$ ./bench/verify_all.sh
== correctness suite     ALL CHECKS PASSED      # tests/ 167 件 + test_kaze 24 件 +
                                                # 一致検査・決定性・needle 4K・ベンチ
== needle ladder (0.5B)  HIT / HIT / HIT / HIT  # 2K / 4K / 8K / 16K
== KV pressure           PASS (all 200)         # 容量逼迫時もエラーにせず待たせる
== server soak           waves: 8, worker errors: 0   # 64 本同時 × 8 波、メモリ増加 0
ALL VERIFICATION PASSED
```

## 4. 設計の要点

- **decode は帯域律速**: 0.5B は 254 MB/token を流し実効 223 GB/s（ピークの 50%）、
  7B は 3.56 GB/token で 320 GB/s（71%）。層あたりの転送が大きいほど DRAM 効率が上がる
- **prefill は計算律速**: int4/int8 の整数テンソルコア + 活性量子化の融合で 1.3〜2.1 倍
- **同時実行**: 継続バッチング + ページド KV + プレフィックス共有で 1.4〜10 倍
- **投機**: prompt-lookup（外部ドラフタ不要）。受理率は `/metrics` の `spec_accept` で見える

## 5. 既知の制限

| 項目 | 状況 |
|---|---|
| 7B 単一 decode | llama.cpp と互角（両者とも帯域の壁） |
| 量子化グループ | **64 と 128 のみ**（32 は int4 decode が違法アクセス） |
| MoE | 実装・テスト済みだが 8GB で載る実用サイズが無い |
| int8 KV | 長距離 recall が壊れるため無効（fp8v は可） |
| MLA / Gemma3 / Phi3 | 未対応 |

## 6. 再現

```bash
./bench/verify_all.sh          # 検証一式
python3 bench/compare.py '{"engine":"kaze","model":"...","kind":"int4","lm_head_kind":"int4","speculative":false}'
python3 -m kaze serve --model ... --kind int4 --port 8000
```

詳細は [final_table.md](final_table.md)（全セル + コマンド）と [evidence.md](evidence.md)（測定方法・却下案）。
