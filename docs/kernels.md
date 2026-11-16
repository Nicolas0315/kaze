# decode カーネルの再チューニング（2026-11-15）

batch=1 の decode で回る GEMV / attention / 起動回数を、**CUDA Graph 内の実時間**で測り直して詰めました。
結果は 0.5B int4 で **1.13 倍**、3B int4 で **1.12 倍**、1.5B int4 で **1.22 倍**（同一プロセス内の交互 A/B、中央値）。

## 先に結果

エンジン通し（interleaved A/B、64 トークン入力・greedy・中央値）:

| model | kind | 変更前 tok/s | 変更後 tok/s | 比 |
|---|---|---|---|---|
| Qwen2.5-0.5B | int8 | 445.2 | **491.8** | 1.10x |
| Qwen2.5-0.5B | int4 | 546.9 | **615.2** | 1.12x |
| Qwen2.5-1.5B | int8 | 165.8 | **173.2** | 1.04x |
| Qwen2.5-1.5B | int4 | 263.3 | **320.3** | 1.22x |
| Qwen2.5-3B | int4 | 139.4 | **156.0** | 1.12x |

この A/B は「設定表の入れ替え」と「batch=1 専用カーネル」の合計です。設定表だけの効果は
0.5B で int8 +3.7% / int4 +3.4%（`bench/cfg_ab.py`）。

カーネル単体（CUDA Graph 内、0.5B）:

| shape | 変更前 | 変更後 | メモ |
|---|---|---|---|
| qkv int8 1152x896 | 7.4 us | **3.7 us** | matvec + split 4 |
| o_proj int8 896x896 | 7.2 us | **3.3 us** | matvec + split 4 |
| gate_up int8 9728x896 | 25.0 us | **22.4 us** | matvec + evict_first |
| down_proj int4 896x4864 | 13.3 us | **10.4 us** | split 8 |
| lm_head int4 151936x896 | 375.8 us | **186.0 us** | block_n 64 / warps 4 / stages 2 |
| lm_head int8 151936x896 | 339.9 us | **338.8 us** | 既に 91% |
| qkv int8 1.5B 2048x1536 | 12.2 us | **8.6 us** | split 4 |
| o_proj int8 1.5B 1536x1536 | 12.8 us | **7.3 us** | split 8 |
| qkv int8 3B 2560x2048 | 18.3 us | **15.6 us** | stages 4 |
| o_proj int8 3B 2048x2048 | 17.0 us | **11.9 us** | block_n 32 / warps 2 / stages 3 |
| down_proj int8 3B 2048x11008 | 66.9 us | **60.7 us** | split 2 / stages 3 |
| lm_head int8 3B | 927.4 us | **834.0 us** | block_n 128 / stages 4 |

## 目次

- **先に結果**: decode 867→877 tok/s ほか（2026-11-15 時点のまとめ）
- **1-2**: 測り方の転換（in-process A/B）とドリフト前提の測定規律
- **3, 7**: batch=1 専用 matvec 経路とその設定表
- **4**: attention の split 数（デバイス適合）
- **8-11**: prefill の 2 カーネル化 / split-K epilogue 融合 / 残差加算
- **5**: 否定したもの一覧（再試行しない）
- **10**: 未着手の候補

### 後日の追加ラウンド（scorecard 参照）

- round 126: batch >= 16 はレイテンシ律速（replay 1.9 ms vs sync 3.7 ms）
- round 127: rope+attention 融合（batch 1 限定・bit 一致・+2.5%）
- round 128-129: fp16 partials 不整合の修正と全 tdim 畳み込みの再採用
- round 133-134: 長文脈 (128,4,3) 既定化・短文脈は b>=32 で stages 3
- round 130, 137: 測定規律（同一コード対照でノイズフロア確認。wall-clock の数%差は判定不能）

## 1. 測り方を変えた（これが一番効いた）

これまでの `_cfg` は「Python から 1 回 launch するのにかかる時間」で調整されていました。
ところが decode は CUDA Graph で再生されるので、**launch コストは存在しません**。
launch を含めると小さいカーネルが過大評価され、その対価として split-K の epilogue（2 回目の launch）が
過小評価されます。実際、in-graph で測り直すと結論が変わりました:

- qkv / o_proj（N が狭い）は split-K 4〜8 が **26〜43% 速い**（旧表は split 1）
- 逆に wide な gate_up は split-K 不要（旧表どおり）

道具は `bench/kernel_lab.py`（in-graph 計測 + 形状別スイープ）、`bench/variance_check.py`（ドリフト測定）。

## 2. ドリフトを前提にした A/B

同じ設定を 6 回測ると **30 秒で 10〜15% 遅くなる**（`bench/variance_check.py`）。
掃引の後半で測った設定は不利になるので、設定の採否は同一プロセス内で交互に測る
`bench/cfg_ab.py` / `rows_ab.py` / `attn_ab.py` で決めました。

## 3. batch=1 専用の matvec 経路（TDIM=1）

decode の実バッチは 1 でも、`tl.dot` が 16 行を要求するため **X を 16 行読み、16 行ぶん計算**していました。
X の再読は CTA 数だけ増えるので、qkv では重みと同等以上のバイト数になります。

そこで engine の行バケットに `1` を足し（`kaze/engine.py`）、カーネル側に `TDIM == 1` の分岐を入れました:

- `tl.dot` を使わず `tl.sum` の matvec（int8 / int4 とも）
- int4 は nibble を interleave せず、X を偶数/奇数レーンに分けて別々に当てる
- split-K の部分和バッファが `[SPLIT_K, 16, N]` → `[SPLIT_K, 1, N]` fp32 に縮む

数値は `tests/test_kaze.py` の `kernel.gemv_rows1_matches_wide` で 16 行経路と cos > 0.9999 を検査します。

## 4. attention の split 数をデバイスに合わせる

long 文脈の decode attention は grid が `(batch, KV heads, splits)` で、splits が 8 固定でした。
2 KV head のモデルでは 16 CTA しか出ず、46 SM の 33% しか使いません。`48 / kvh` に自動化しました
（`kaze/model.py`、`KAZE_ATTN_SPLITS` で上書き可）。combine カーネルは `[S, TDIM, D]` の 3D タイルを
1 回で読む実装だと S=24 で 98 KB を registers に載せることになるため、splits を回す 3 パスに書き換えました。

| 条件 | splits 8 | splits 24 |
|---|---|---|
| 0.5B int8, 4K 文脈 | 438.0 tok/s | 440.9 tok/s（中立） |
| 0.5B int8, 16K 文脈 | 258.3 tok/s | **274.7 tok/s**（+6.4%） |

## 5. 否定したもの（再試行しない）

- **狭い形状に `eviction_policy="evict_first"`**: qkv 3.7 → 4.7 us。広い形状（gate_up）だけ効く
- **wide 形状の `block_k=64`**: gate_up / lm_head は 128 のままが最良
- **`SPLITK_STAGES` で stages を 2 に固定**: 形状別の stages（3〜4）を潰していたので外した

## 6. 学び

1. **decode のカーネルは graph 内で測る**。launch を含めた数字は「単体で速いのに実機で負ける」を量産する
2. **掃引は時間ドリフトとの戦い**。採否は交互 A/B、それも同一プロセスで
3. **X の再読は重みと同じくらい高い**。バッチ 1 専用経路を持つ価値はここにある

## 7. batch=1 経路の設定表（round 95）

TDIM=1 のカーネルを入れたのに設定表は 16 行経路のままだと、利点の半分しか取れません。
1 行しか読まない経路では「CTA を増やすほど良い」に変わるので、`bench/kernel_lab.py micro --tdim 1`
で掃引し直しました（`_cfg_rows1`）。

| shape（0.5B int4） | 16 行表 | 1 行表 | 差 |
|---|---|---|---|
| gate_up 9728x896 | 15.9 us | **12.5 us** | +21% |
| down_proj 896x4864 | 8.3 us | **7.4 us** | +11% |
| qkv 1152x896 | 4.2 us | **3.9 us** | +7% |
| o_proj 896x896 | 3.8 us | **3.6 us** | +5% |

エンジン通し（同一プロセス内 A/B、`bench/cfg_ab.py --modes rows16cfg,tuned`）:
0.5B int4 676.6 → **751.3 tok/s**（1.11x）、0.5B int8 536.8 → 548.1（1.02x）、
1.5B int4 337.2 → 347.1（1.03x）、3B int4 191.6 → 198.7（1.04x）。

## 8. prefill の 2 カーネル（round 95）

`bench/prefill_lab.py` で 512 トークン prefill の内訳を取ると、gemm_swiglu_w8a8（40%）と
gemm_int4_w8a8（39%）で 8 割、活性量子化（quant_act）が 8% でした。

| 変更 | 前 | 後 | prefill 0.5B int4 |
|---|---|---|---|
| quant_act: 128 列・1 warp → 512 列・4 warp（グループ amax はタイル内 reshape） | 998 us | 723 us | 39,827 → 40,421 tok/s |
| swiglu tile (32,64,4,2) → (64,128,8,2)（BLOCK_M=64 は 8 warp でないと spill する） | 183 us/層 | 150 us/層 | 40,592 → 43,687 tok/s |
| int4 GEMM tile の block_n 64 → 128（stages 3 のまま） | — | — | 43,681 → **46,421 tok/s** |

合計で **39.8k → 46.4k tok/s（+17%）**。BLOCK_M=128 は 35.5k まで落ちるので使わない。

## 9. split-K epilogue の融合（round 96）

`gemv_epilogue_kernel` は decode 1 step で 72 回・87 us（6.6%）を占めていました。やっていることは
「部分和を sp 昇順で足す → bias → 残差 → fp16 丸め」だけなので、**消費側と同じ順序で畳めば**
カーネルごと消せます（rmsnorm と rope_scatter に `partials=` を渡す）。

- 有効化は tdim==1 のバケットのみ（`KAZE_FUSE_EPILOGUE=0` か `set_fuse_epilogue(False)` で従来経路）
- 効果: 0.5B int4 733.1 → **754.4 tok/s**、0.5B int8 534.5 → **550.9**、1.5B int4 340.9 → 344.1
- 一致: `bench/check_epilogue_fold.py` が `torch.equal` で確認（0.5B int4/int8, 1.5B int4）
- 罠: `num_warps` を変えると `tl.sum` の fp32 リダクション木が変わって 1〜2 ULP ずれる（4 warps 固定）

prefill 側も同じ発想で 1 本減らしました。`quant_act_kernel` は int4 GEMM の入力を作るために
96 回/step 走っていましたが、rmsnorm が `out_q`/`out_s` を書けるので、そこで作って `aq`/`as_` として
渡します（丸めは quant_act と同じ「0 から遠い方へ」）。0.5B prefill 43.9k → **47.6k tok/s**、
1.5B +5.2%、3B +5.8%（`KAZE_FUSE_PREFILL_ACTQUANT=0` で従来経路）。

## 11. prefill の残差加算と attention タイル（round 97）

- int4 prefill GEMM に `HAS_RES` を追加し、`hidden = hidden + o` を epilogue に畳んだ（1 層 2 回）。
  1.5B で +1.2%。
- prefill attention のタイルは (64,64,4,3) が最良（0.5B +2.1%、1.5B +0.7%）。
  8 通り試して、`KAZE_ATTN_PREFILL_TILE` で切り替えられるようにした。

## 12. round 141-148: in-graph 掃引の全面展開（2026-11-15 追記）

eager マイクロベンチが 3 回（round 59・126・140）誤判定したことを受けて、
**captured-graph リプレイ + CUDA イベント**の掃引手法を全カーネルに展開しました。

### decode（全モデル形状を in-graph で確定）

| 項目 | 旧 | **新** | 効果（end-to-end） |
|---|---|---|---|
| 7B swiglu (gate_up 融合) | (128,8,4) 231 µs | **(32,4,4) 191 µs（−17%）** | decode +10.8% |
| 7B qkv | (128,1,8,2) plain 55.4 µs | **(16,8,2,2) 38.2 µs（−31%）** | （↓と合算 +8.9%） |
| 7B down | (32,8,4,2) 91.6 µs | **(32,4,2,2) 77.9 µs（−15%）** | |
| 3B down | (32,8,4,2) 35.9 µs | **(32,8,2,2) 31.3 µs（−13%）** | decode +3.3% |
| 0.5B / 1.5B 全形状 | — | 既定どおり最良 | 変更なし |

7B の結果: decode 64.4 → **88.0 tok/s**（llama.cpp 77.0 を +14% 上回る）。
「小さい nh での否定」を大きい nh に安易に適用しない（CTA 数が SM 周回数を変えると
最適点が逆転する）。

### prefill（W4A8 GEMM）

- stages 3 → **2**: 0.5B prefill 47.3k → **49.2k tok/s（+2.1%）**、3B +1.0%。
  swiglu 融合カーネルは既定 (64,128,8,2) が最良（stages 3 は +11% 悪い）

### 長文脈（7B ラダー）と正しさ

- 7B: 4K 78.1 / 8K 70.8 tok/s（fp16 KV）、16K 62.7 tok/s（fp8v — fp16 KV では 4.3 GB で載らない）
- needle 4K HIT — 全変更後も長文脈 recall を維持

### 新しい掃引ツール

- `bench/ingraph_all_shapes.py`: 全モデル形状の swiglu + splitk を一括 in-graph 掃引
- `bench/ingraph_prefill_ab.py`: prefill GEMM/swiglu の in-graph 掃引
- `bench/swiglu_ingraph_ab.py` / `bench/ingraph_gemv_ab.py`: 単一形状の詳細掃引

## 10. 未着手（次の候補）

- split-K の epilogue を消費側（rmsnorm / rope）に畳む（決定的なまま起動 1 回と往復 1 回を削れる）
- rmsnorm + qkv / gate_up の融合（1 層あたり 2 launch と中間テンソルの往復を削る）
- short 文脈用 `attn_decode_direct_kernel` の CTA 数（今は 1 KV head = 1 CTA）
- int4 のスケール乗算を dot 後へ（`docs/scorecard.md` の指摘。ALU よりレジスタが本命）
