# 現行コードの横断監査・修正

2026-11-15。前回のレビュー対象に限定せず、現行の推論エンジン・キャッシュ・モデル計算・重み読み込み・HTTP・CLI・検証スクリプトを点検しました。第三者コードは変更していません。

## 修正した主な不具合

| 領域 | 確認した問題と修正 |
|---|---|
| スレッドの終了 | shutdown後も生成イベント・出力キューを待ち続けていたため、全待機者へ失敗を通知。未起動workerへの受付も即時エラーにし、配送と終了処理を同じロックで保護 |
| テキスト出力 | 日本語の途中バイトを置換文字として先に送る問題、stop文字列やその接頭辞の漏出、トークンなし終端時の保留文字の消失を修正。HTTPとPython APIで同じ出力処理を使用 |
| サンプリング | presence penaltyの未適用、128トークン以前の履歴無視、設定cloneの配列共有、FP32入力logitsの破壊、top-pの境界・極小値で候補が消える問題を修正 |
| 入力設定 | 無効な生成長・バッチ予算・token ID・非有限数値・seed範囲を早期検証。反復ペナルティはFP32 overflowを避けるため `1e-6〜1e6` の対応範囲を明示 |
| HTTP | completionsでseed等の設定を黙って無視していた問題を修正。エンジンが登録前に退けるプロンプトは422で返す |
| KVメモリ | ブロック確保の途中失敗をrollback。予備ブロックの不要な要求を除き、過大プロンプトは受付時にエラー。生成中に単独要求が容量を使い切った場合はその要求だけを終了 |
| キャッシュ所有権 | prefix再計算時に共有最終ブロックへ書き込む問題、重複hash再登録で別ブロックの対応が消える問題、draft KV解放漏れを修正 |
| 投機的デコード | 分岐用KVコピーが高度索引の一時tensorにしか反映されない問題、draft容量不足時の部分確保、文脈限界・停止理由・実出力数の集計を修正。penalty指定時は通常経路へ戻す |
| モデル設定 | 未対応architecture・活性化・RoPEを別の計算で黙認せず早期エラー。新旧RoPE設定の優先順位、MoE設定の別名、Mixtralのtop-k再正規化既定、寸法整合を修正 |
| 量子化 | FP16へ保存したscaleと量子化計算の不一致、微小scaleのゼロ化、無効なgroup/chunk/bias、shape broadcastによる誤復元を修正。重み検証の一時領域は最大1,048,576要素に分割し、巨大headでのピークメモリを抑制 |
| 重み読み込み | MoEの未定義結合関数、mixed時のgate/up量子化種別不一致、重みshape・q/k正規化ペア・biasの検証不足、safetensorsの境界・空tensor・shard重複／参照先、形式選択の不整合を修正 |
| GPU数値計算 | RMSNorm scaleの行stride越境、GQA24/32のshape・未書込み、MoEのsoftmax欠落とrouter mask、INT4 prefill末尾store越境、FP16 fallbackのK末尾読込みを修正。並行追加されたTDIM=1の通常／SplitK／SwiGLU経路のhigh nibbleゼロ点欠落も修正 |
| CLI | 1トークン生成時のNone計時表示、例外時shutdown漏れ、不正引数、benchの背景workerと直接stepの競合を修正 |
| ベンチマーク | 指定した長さ・反復・同時数の未反映、固定生成数による誤集計、子process／HTTP失敗や欠損結果の見逃し、終了処理・CWD依存パスを修正 |
| 検証ゲート | 終了コードだけでなく成功要約・試験件数・JSON数値を確認し、今回の回帰試験も一括検証へ登録 |

## 検証方法

各修正は、実コードを通る失敗再現を先に確認してから変更しました。GPU依存の制御試験では該当境界だけをCPU fixtureへ置き換え、モデルとカーネルの計算はTriton interpreterと実GPUの両方で確認しました。Tokenizer後処理で既送信の空白を書き換える問題も、実際のローカルTokenizerで再現・修正しています。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python3 -B -m unittest discover -s tests -p "test_*audit.py"
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -p "test_*regressions.py"
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_worker_failure.py
PYTHONDONTWRITEBYTECODE=1 KAZE_AUDIT_DEVICE=cuda python3 -B tests/test_model_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_kaze.py
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_stress.py 12
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_smoke.py
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_server.py
```

## 最終検証結果

| 検証 | 最終結果 |
|---|---|
| 今回追加したCPU監査テスト | 126件成功 |
| 既存CPU回帰 | 30件成功、GPU専用2件skip |
| worker障害回帰 | 9件成功 |
| 実GPUの局所モデル／カーネル回帰 | 11件＋2件成功 |
| 実モデル統合 | 18/18成功 |
| 品質スモーク | 6条件成功 |
| 実HTTPサーバー | chat・completions・streaming成功 |
| 最終のランダム負荷再試験 | 並行GPU処理のメモリ競合によりCUDA OOMで途中停止 |
| Python構文・git diff空白検査 | 成功 |

CPU検証の成功数は計165件です。局所GPU回帰は13件で、CPU実行とは別に実デバイスでも計算を確認しています。負荷試験は同セッション中に12/12の成功も確認しましたが、その後の並行変更と最終再試験のOOMがあるため、最終版の負荷試験まで完了した扱いにはしていません。GPUの競合がない状態で、上記の12試行をもう一度実行するのが残る確認です。

## 変更の保護と検証の範囲

- 作業開始時の未コミット文書は保持しています。並行して追加・調整されたカーネルのretuneとベンチ用コードも、全面上書きせず保持しています。
- 自動生成された追跡済みbytecodeの差分だけを元へ戻し、以後のbytecode生成を抑えて試験しました。新しいキャッシュファイルはignoreします。
- prefixキャッシュのPython hash値が衝突する理論上の制限は残ります。完全なprefixキーへの変更やdigest方式の移行は今回実施していません。
- dynamic NTK RoPEなど未実装の計算方式は、対応済みとして近似実行せずエラーにします。新しいモデル方式の実装は追加していません。
- 全モデル・全GPUでの網羅試験や、性能比較ベンチマーク一式の再測定は実施していません。既存の性能値は更新していません。
- commit・push・サービス変更は実施していません。
