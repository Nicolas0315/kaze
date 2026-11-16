# 外部 GPU での検証手順

kaze を別の NVIDIA GPU で測るときの手順。数字を比較できるように、環境の記録と同一セッション内 2 回の測定を必須にする。

## 1. 前提

- NVIDIA GPU（compute capability 7.5 以上）
- Python 3.11 以上、`pip install torch triton transformers safetensors fastapi uvicorn`
- モデルは HF 形式（safetensors）で用意する。GGUF は読み込み非対応
- llama.cpp を比較に使う場合は同じリポジトリの `bench/llamacpp` をビルドし、同じモデルの GGUF を用意する

## 2. 環境の記録（必須）

```bash
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.version.cuda)"
```

消費電力上限とクロックを必ず添える。GPU が熱で throttling すると 15% 程度平気でぶれるため、この情報が無い数字は採用できない。

## 3. 検証一式

```bash
./bench/verify_all.sh
```

次をまとめて実行し、最後に `ALL VERIFICATION PASSED` が出れば合格。

| 項目 | 内容 |
|---|---|
| 正しさ | `tests/` 191 件、カーネル一致、決定性、epilogue 畳み込み、投機一致 |
| needle | 2K / 4K / 8K / 16K の復唱 |
| KV 逼迫 | 容量不足時にエラーにせず待たせる |
| 連続負荷 | 64 本同時 × 8 波、メモリ増加なし |

## 4. 個別の測定

```bash
# 単一ストリーム（decode 128 トークン / prefill 512 トークン）
python3 bench/compare.py '{"engine":"kaze","model":"<hf_dir>","kind":"int4","lm_head_kind":"int4","speculative":false}'

# 同時実行（kaze サーバ）
python3 -m kaze serve --model <hf_dir> --kind int4 --port 8099 --host 127.0.0.1 &
python3 bench/compare_kaze_server.py 8099 64 32 64

# 同時実行（llama.cpp サーバ、比較用）
python3 bench/compare_llama_server.py <gguf> 64 32 64

# 長文脈（24GB なら 7B で 128K が狙える）
python3 bench/needle.py --model <hf_dir> --ctx 131072 --factor 8
```

## 5. 記録してほしい表

| model | decode | prefill | 32 本 | 64 本 | 備考 |
|---|---|---|---|---|---|
| 0.5B | | | | | |
| 1.5B | | | | | |
| 3B | | | | | |
| 7B | | | | | |

## 6. 注意

- **1 回だけの測定は採用しない。** 同一セッション内で 2 回取り、良い方（または中央値）を使う
- カーネルのタイルは 8GB / 448 GB/s（RTX 3070）で調整してある。帯域や SM 数が大きく違うカードでは、そのまま最適とは限らない。**「動く」と「速い」を分けて報告**してほしい
- RTX 50 系（sm_120）は PyTorch 2.9 以降が必要。動かない場合はバージョンを添えて報告
