#!/usr/bin/env bash
# RunPod 環境 setup + M0-M3 完整執行腳本
# 在 RunPod instance 的 /workspace 或你指定的目錄下跑
# 假設已在 RunPod 上 clone 本 repo 或掛載 volume

set -e   # 任何指令失敗立刻停

# ── 1. 路徑設定（依 RunPod 實際掛載點修改）─────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-/workspace/01_navipath}"
DATA_DIR="${DATA_DIR:-/workspace/can_dataset}"
CONCH_CKPT="${CONCH_CKPT:-/workspace/checkpoints/conch/pytorch_model.bin}"
FOLD="${FOLD:-1}"
EPOCHS="${EPOCHS:-12}"   # 對齊 QPMIL 論文設定

cd "$PROJECT_DIR"
echo "[setup] PROJECT_DIR=$PROJECT_DIR DATA_DIR=$DATA_DIR"

# ── 2. Python 環境 ────────────────────────────────────────────────────────────
# 若無 uv，改用 pip
if command -v uv &>/dev/null; then
    uv venv .venv --python 3.11 2>/dev/null || true
    source .venv/bin/activate
    uv pip install "torch>=2.1" torchvision torchaudio \
        "transformers>=4.40,<5" huggingface-hub==0.36.2 \
        numpy "pyyaml" "scikit-learn" "tqdm" \
        timm einops h5py matplotlib seaborn \
        wandb pytest ruff openpyxl pandas
else
    pip install "torch>=2.1" torchvision torchaudio \
        "transformers>=4.40,<5" huggingface-hub==0.36.2 \
        numpy pyyaml scikit-learn tqdm \
        timm einops h5py matplotlib seaborn \
        wandb pytest ruff openpyxl pandas
fi

# ── 3. Clone QPMIL-VL（若無）────────────────────────────────────────────────
if [ ! -d QPMIL-VL ]; then
    git clone https://github.com/can-can-ya/QPMIL-VL.git
fi

# ── 4. 資料 symlink ───────────────────────────────────────────────────────────
if [ ! -e data ]; then
    ln -s "$DATA_DIR" data
fi
echo "[setup] data -> $(readlink data)"

# ── 5. 更新 QPMIL config 路徑 ─────────────────────────────────────────────────
python - <<EOF
import yaml

cfg_path = "QPMIL-VL/configs/main.yaml"
with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["dataset_root_dir"]  = "$PROJECT_DIR/data"
cfg["class_ensemble_path"] = "$PROJECT_DIR/QPMIL-VL/class_ensemble/class_ensemble.json"
cfg["conch_ckpt_path"]   = "$CONCH_CKPT"

# RunPod 有 GPU
import torch
if torch.cuda.is_available():
    cfg["cuda_id"] = 0
cfg["num_workers"] = 4

with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print("[setup] QPMIL config updated ->", cfg_path)
EOF

# 同步更新我們自己的 config
sed -i "s|data_root: .*|data_root: $PROJECT_DIR/data|" configs/*.yaml

# ── 6. 驗證環境 ──────────────────────────────────────────────────────────────
echo "[verify] torch / cuda / data"
python -c "
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
import os, torch as T
f=sorted(os.listdir('data/tcga_esca/feats-l1-s256_CONCH/pt_files'))[0]
x=T.load('data/tcga_esca/feats-l1-s256_CONCH/pt_files/'+f, map_location='cpu')
print('esca feat shape', x.shape, x.dtype)
"
echo "[verify] skeleton tests"
PYTHONPATH=. python tests/test_shapes.py

# ── 7. M1 — QPMIL paper order（全資料，存 checkpoint）──────────────────────
echo "[M1] paper order fold=$FOLD epochs=$EPOCHS"
python train_qpmil_runner.py \
    --order paper \
    --fold "$FOLD" \
    --epochs "$EPOCHS" \
    --save-ckpt \
    2>&1 | tee outputs/run_paper_fold${FOLD}.log

echo "[M1] reverse order fold=$FOLD epochs=$EPOCHS"
python train_qpmil_runner.py \
    --order reverse \
    --fold "$FOLD" \
    --epochs "$EPOCHS" \
    --save-ckpt \
    2>&1 | tee outputs/run_reverse_fold${FOLD}.log

# ── 8. M2 — debug internals（驗 shape）───────────────────────────────────────
echo "[M2] debug internals"
python debug_internals.py \
    --ckpt "outputs/qpmil_paper_fold${FOLD}.pt" \
    --order paper \
    --slide-dir data/tcga_lung/feats-l1-s256_CONCH/pt_files

# ── 9. M3 — patch-budget table（全資料 eval，4 個任務都出）──────────────────
echo "[M3] patch-budget eval — all 4 tasks, paper order"
for TASK_IDX in 0 1 2 3; do
    python run_patch_budget.py \
        --ckpt "outputs/qpmil_paper_fold${FOLD}.pt" \
        --order paper \
        --task-index "$TASK_IDX" \
        2>&1 | tee "outputs/budget_paper_task${TASK_IDX}_fold${FOLD}.log"
done

echo ""
echo "============================================================"
echo " ALL DONE. Key outputs:"
echo "   outputs/qpmil_paper_fold${FOLD}.json   <- R matrix + summary"
echo "   outputs/qpmil_reverse_fold${FOLD}.json  <- reverse R matrix"
echo "   outputs/qpmil_paper_fold${FOLD}.pt      <- backbone checkpoint"
echo "   outputs/budget_paper_task*_fold${FOLD}.log <- ACC@K tables"
echo "============================================================"
