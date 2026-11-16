# AMD GPU（ROCm）で kaze を動かす

この文書は「ROCm 実機で動かすための手順」と「このリポジトリで実際に確かめられた範囲」を
分けて書いています。手元に AMD GPU が無いため、実機の速度・Triton の gfx 向けコンパイル・
hipGraph は **未検証** です。一方、バックエンド判定とエンジンの分岐は `KAZE_BACKEND` で
CUDA 機から再現でき、テストで固定してあります。

## 1. 先に要点

- カーネルは素の Triton なので、gfx 向けにもそのままコンパイルされます。移植のための書き換えはありません。
- ROCm では **CUDA Graph（hipGraph）を既定で使いません**（`graphs_ok=False`）。エンジンは eager 経路に落ちます。
- `--cache-dtype fp8` / `fp8v` は **gfx94x 級のみ**。無い環境では fp16 に落ちて警告を出します（実行時に落ちません）。
- 速度は sm_86（RTX 3070）で掃引した tile のままで、ROCm の数値は未測定です。§6 を読んでください。

## 2. インストール

### 2.1 ROCm 本体

1. AMD の配布する `amdgpu-install` で ROCm を入れます（Ubuntu 22.04 / 24.04 など、配布物の対応表に従う）。
2. GPU を使うユーザーを `render` と `video` グループに追加します（再ログインが必要）。

```bash
sudo usermod -aG render,video "$USER"
rocm-smi                      # GPU が見えること
ls -l /dev/kfd /dev/dri       # renderD* が読めること
```

### 2.2 PyTorch（ROCm wheel）

`requirements.txt` の `torch` は CUDA 版が入るので、ROCm wheel で置き換えます。
`ROCM_INDEX` は環境に合わせて選んでください（例: `rocm6.2`、`rocm6.3`）。

```bash
python3 -m venv .venv-rocm && . .venv-rocm/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install --force-reinstall --no-deps torch \
    --index-url https://download.pytorch.org/whl/ROCM_INDEX
```

**注意**

- `triton` を PyPI から単体で入れ直さないでください（CUDA 向けの wheel です）。ROCm 版の torch wheel が同梱の Triton を使います。
- `torch.version.hip` が `None` なら CUDA 版が残っています。kaze はこの値で ROCm を判定します。
- ROCm 版 torch は CUDA 版より版が遅れることがあります。`torch>=2.4` を満たす ROCm wheel を選んでください。
- RDNA で未対応の gfx しか無い場合は `HSA_OVERRIDE_GFX_VERSION=10.3.0` のような回避策がありますが、
  カーネルが誤動作しうる自己責任の領域です（kaze 側は何も補正しません）。

確認:

```bash
python3 - <<'PY'
import torch, triton
print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

### 2.3 その他の依存

`transformers` / `safetensors` / `fastapi` / `uvicorn` は変更ありません。

## 3. 何が自動で切り替わるか

判定は `kaze/backend.py` の `detect_device()` に集約されています。

| 項目 | CUDA | ROCm | CPU（リファレンス） |
|---|---|---|---|
| `vendor` | `torch.version.hip` が None なら `cuda` | `torch.version.hip` があれば `rocm` | `cpu` |
| `graphs_ok` | capability >= 7 で True | **False**（`KAZE_GRAPHS=1` で明示的に有効化可） | False（`KAZE_GRAPHS` は効かない） |
| `fp8_ok` | True | capability >= 9（gfx94x 級）のみ True | False |

エンジン側の反応（`kaze/engine.py`）:

1. `graphs_ok=False` のとき `use_cuda_graph` を落とし、**1 行だけ**ログを出して eager 経路に切り替えます。
   decode 用 graph（`capture_graphs`）と投機的検証用 graph（`capture_spec_graph`）の両方が対象です。
2. `--cache-dtype fp8` / `fp8v` かつ `fp8_ok=False` のとき、`cache_dtype_policy()` が **fp16 に落として警告**します。
   （8bit KV はメモリ削減の機能なので、未対応バックエンドでは「無効化」が正しい振る舞いです）
3. replayer（`_decode_* / _spec_* / _draft_*`）は従来どおり `self.use_cuda_graph` と graph の有無の両方を見るので、
   eager 経路は既存の `use_cuda_graph=False` と同じコードを通ります。

CLI（`kaze/__main__.py`）:

```bash
python3 -m kaze serve   --model MODEL --backend rocm --no-cuda-graph ...
python3 -m kaze generate --model MODEL --backend rocm "こんにちは"
```

- `--backend {cuda,rocm,cpu}` は **ポリシーだけ**を強制します（`KAZE_BACKEND` と同じ）。デバイスは PyTorch 側のままです。
- `--no-cuda-graph` は従来どおり「graph を使わない」指示です。
- 起動時に 1 行だけ出ます:

```text
kaze: backend=rocm device=... sm_86 graphs=off fp8=unsupported cuda_graph=off cache_dtype=fp16
```

## 4. 実機 ROCm での検証手順

小さい順に流してください。0.5B モデル（`Qwen2.5-0.5B-Instruct`）だけで十分です。

```bash
# 1. ポリシーとエンジンの分岐（GPU 不要・数秒）
python3 tests/test_backend.py

# 2. カーネル / エンジンの GPU テスト（0.5B が必要）
python3 tests/test_kaze.py

# 3. CPU だけで走る回帰一式（全テスト）
python3 -m unittest discover -s tests -p 'test_*.py'

# 4. 単発生成で配管を確認（--backend は省略しても ROCm 機なら rocm 判定）
python3 -m kaze generate \
  --model /models/Qwen2.5-0.5B-Instruct \
  --kind int4 --max-tokens 64 "AMD で動く?"

# 5. まとめ検証（テスト + needle + KV 圧力 + soak）
./bench/verify_all.sh

# 6. しきい値付きのベンチ（任意）
python3 bench/verify.py
```

**しきい値の扱い**: `bench/verify.py` の decode/prefill しきい値（0.5B で 550 tok/s など）は
sm_86 の実測に合わせてあります。ROCm で下回っても異常とは限りません。まず `bench/run.py` で
手元の数値を取り、その値にしきい値を置き換えてから回してください。落ちた項目は
「速度が sm_86 基準に届かない」なのか「出力が壊れている」なのかを切り分けてください。

**未登録のテスト**: `bench/verify.py` の `REGRESSIONS` 一覧には `test_backend.py` がまだ入っていません
（今回の変更範囲外）。手元では `python3 -m unittest discover` と `python3 tests/test_backend.py` で走ります。

## 5. 検証済み / 未検証

| 項目 | 状態 | 根拠 |
|---|---|---|
| バックエンド判定（`KAZE_BACKEND`、`KAZE_GRAPHS`） | 検証済み | `tests/test_backend.py` DetectionTests（CUDA 機で rocm/cuda/cpu を強制） |
| graph 無効化と eager 経路 | 検証済み | 同 EngineGatingTests（`graphs=={}`、eager の出力が既存 no-graph 経路と一致） |
| fp8/fp8v → fp16 フォールバック | 検証済み | 同（警告文と KV dtype を確認） |
| CLI の `--backend` / 起動行 | 検証済み | 同 CliBackendTests（環境変数への反映と 1 行出力） |
| CUDA 経路の非退行 | 検証済み | `tests/test_kaze.py` 24/24、`unittest discover` 184 tests OK、CLI 生成が backend 強制で同一 |
| AMD 実機での実行 | **未検証** | 手元に AMD GPU が無い |
| Triton の gfx 向けコンパイル | **未検証** | 同上 |
| hipGraph capture | **未検証**（既定 off） | 同上。`KAZE_GRAPHS=1` は自己責任 |
| ROCm での速度・しきい値 | **未測定** | 同上。§6 の注意どおり読み替える |

### 実測ログ（CUDA 機で ROCm ポリシーを強制した例・ROCm 実機ではない）

RTX 3070、Qwen2.5-0.5B-Instruct int4、24 トークン貪欲生成。同じプロンプトで 3 通り:

| 実行 | 起動行 | 出力 | tpot |
|---|---|---|---|
| `--backend cuda` | `graphs=on fp8=ok cuda_graph=on` | 同一トークン列 | 1.16 ms（205.5 tok/s） |
| `--backend rocm` | `graphs=off fp8=unsupported cuda_graph=off` | 同一トークン列 | 5.23 ms（40.5 tok/s） |
| `--no-cuda-graph`（参照） | `graphs=on ... cuda_graph=off` | 同一トークン列 | 5.16 ms（39.5 tok/s） |

つまり ROCm ポリシーで落ちる経路は、既存の no-graph 経路と同じ出力・同じ速度帯になります
（graph の有無で 5 倍近く変わるのは sm_86 の launch コストのため。ROCm 実機の値ではありません）。

graph を落としたときのログは 1 行だけ出ます:

```text
  cuda graph: disabled by backend policy (rocm); running the eager path
kaze: backend=rocm device=NVIDIA GeForce RTX 3070 sm_86 graphs=off fp8=unsupported cuda_graph=off cache_dtype=fp16
```

## 6. 性能の注意（数値は出しません）

- tile 表は sm_86（SM 46、448 GB/s、warp 32）で掃引したものです。wavefront 64 の AMD では
  `tl.dot` の形・BN・split-K の分割数・occupancy の最適点がずれます。
- 調整ノブは `KAZE_SPLITK_MAX`、`KAZE_ATTN_SPLITS`、`KAZE_ATTN_PREFILL_TILE`、`KAZE_KERNEL_CFG` など
  （`docs/kernels.md` 参照）。ROCm では掃引し直しが必要です。
- CUDA Graph が無いぶん decode の launch オーバーヘッドが効きます。sm_86 では graph 有無で差が出ていたので、
  ROCm の単一ストリームは相対的に不利になります（バッチを増やすほど希薄化）。
- fp8 が使えないと KV は fp16 のままなので、長文脈の同時実行本数は CUDA の fp8v より減ります。

## 7. 既知の制約

- `--backend` / `KAZE_BACKEND` は **ベンダーポリシーの上書き**であり、デバイスの切り替えではありません
  （ROCm では `torch.device("cuda")` が HIP の別名です）。
- ポリシーを強制した場合は `tuning_note` に「detected … を KAZE_BACKEND で上書きした」と残ります。
- CPU リファレンス経路では `KAZE_GRAPHS` は無視されます（capture 自体が不可能なため）。
- 不正な `KAZE_BACKEND`（例: `tpu`）は `ValueError` で落ちます。CLI では argparse が弾きます。

## 8. 環境変数まとめ

| 変数 | 値 | 意味 |
|---|---|---|
| `KAZE_BACKEND` | `cuda` / `rocm` / `cpu` | ベンダーポリシーを強制（デバイスは変えない） |
| `KAZE_GRAPHS` | `1`/`0`（`true`/`false`/`on`/`off` も可） | graph 方針の上書き。CPU 経路では無効 |
| `KAZE_SPLITK_MAX` ほか | 数値 | カーネルの調整ノブ（`docs/kernels.md`） |
