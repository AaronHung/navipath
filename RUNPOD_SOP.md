# RunPod 小白 SOP — NaviPath-MoE 實驗

---

## 準備工作（在 Mac 上做一次）

### 1. 建 GitHub private repo 推上去

```bash
cd ~/research/01_navipath
git remote add origin https://github.com/<你的帳號>/01_navipath.git
git push -u origin main
```

### 2. 確認 .gitignore 已排除大檔（已設定好）

`.gitignore` 已排除：`QPMIL-VL/`、`outputs/`、`checkpoints/`、`*.pt`、`*.bin`。
→ 只有程式碼上去，**資料和模型不走 git**，用另外的方式傳（見下方）。

---

## 每次開 RunPod 的 SOP

### Step 1：開機器

- 租 GPU：建議 RTX 4090（24GB）或 A100 40GB
- Image：選 `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`（或任何 PyTorch 2.x）
- Storage：Network Volume 50GB（用來放資料，跨機器共用）
- 開機後點 **Connect → Connect to Jupyter**（或 SSH）

### Step 2：上傳資料（第一次）

資料很大（~100GB），不走 git，用 **Network Volume** 或 `rclone`。

**方案 A：Network Volume（最推薦）**
```
RunPod 控制台 → Storage → 建一個 50GB Network Volume
掛到 /workspace/can_dataset
之後每次開機都選同一個 Volume，資料永久保留
```

**方案 B：從 Mac 用 scp 上傳（只需上傳一次）**
```bash
# RunPod 開 SSH，取得 host:port，例如 ssh.runpod.io:12345
scp -P 12345 -r ~/research/can_dataset root@ssh.runpod.io:/workspace/can_dataset
scp -P 12345 ~/research/01_navipath/checkpoints/conch/pytorch_model.bin \
    root@ssh.runpod.io:/workspace/checkpoints/conch/pytorch_model.bin
```

**方案 C：用 HuggingFace Hub 下載 CONCH（需申請 token）**
```bash
huggingface-cli login   # 輸入 token
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('MahmoodLab/CONCH', 'pytorch_model.bin', local_dir='/workspace/checkpoints/conch')"
```

### Step 3：Clone repo 並設定環境

```bash
cd /workspace
git clone https://github.com/<你的帳號>/01_navipath.git
cd 01_navipath

# Clone QPMIL-VL（程式碼，不在 git 裡）
git clone https://github.com/can-can-ya/QPMIL-VL.git

# 安裝套件（RunPod PyTorch image 已有 torch，只需補其他）
pip install "transformers>=4.40,<5" huggingface-hub==0.36.2 \
    timm einops h5py openpyxl wandb scikit-learn tqdm seaborn

# 設定路徑（一次）
python - <<EOF
import yaml
cfg = yaml.load(open("QPMIL-VL/configs/main.yaml"), Loader=yaml.FullLoader)
cfg["dataset_root_dir"]    = "/workspace/can_dataset"
cfg["class_ensemble_path"] = "/workspace/01_navipath/QPMIL-VL/class_ensemble/class_ensemble.json"
cfg["conch_ckpt_path"]     = "/workspace/checkpoints/conch/pytorch_model.bin"
yaml.dump(cfg, open("QPMIL-VL/configs/main.yaml","w"), allow_unicode=True)
print("config updated")
EOF

sed -i "s|data_root: .*|data_root: /workspace/can_dataset|" configs/*.yaml

# 驗證
PYTHONPATH=. python tests/test_shapes.py
```

### Step 4：跑實驗

```bash
cd /workspace/01_navipath

# M1：QPMIL baseline（paper + reverse，fold 1，存 checkpoint）
python train_qpmil_runner.py --order paper   --fold 1 --save-ckpt 2>&1 | tee outputs/run_paper_f1.log
python train_qpmil_runner.py --order reverse --fold 1 --save-ckpt 2>&1 | tee outputs/run_reverse_f1.log

# M3：patch-budget 表（不需訓練，直接跑 inference）
for TASK in 0 1 2 3; do
  python run_patch_budget.py --ckpt outputs/qpmil_paper_fold1.pt \
      --order paper --task-index $TASK 2>&1 | tee outputs/budget_task${TASK}.log
done

# M4：MicroRouter v0（go/no-go 驗證）
python train_router_v0.py \
    --backbone-ckpt outputs/qpmil_paper_fold1.pt \
    --order paper --fold 1 --epochs 5 2>&1 | tee outputs/router_v0_f1.log

# 看 go/no-go 結果
grep "go/no-go" outputs/router_v0_f1.log
```

---

## Step 5：跑完把結果拉回 Mac

```bash
# 在 Mac 上執行
RUNPOD_HOST="ssh.runpod.io"
RUNPOD_PORT="12345"   # 換成你的 port
RUNPOD_DIR="/workspace/01_navipath/outputs"
LOCAL_DIR="$HOME/research/01_navipath/outputs"

mkdir -p "$LOCAL_DIR"
scp -P $RUNPOD_PORT -r root@$RUNPOD_HOST:$RUNPOD_DIR/*.json "$LOCAL_DIR/"
scp -P $RUNPOD_PORT -r root@$RUNPOD_HOST:$RUNPOD_DIR/*.log  "$LOCAL_DIR/"
scp -P $RUNPOD_PORT -r root@$RUNPOD_HOST:$RUNPOD_DIR/*.pt   "$LOCAL_DIR/"
echo "sync done"
```

---

## 常見問題

| 問題 | 解法 |
|---|---|
| `ModuleNotFoundError: navipath_moe` | 確認在 `/workspace/01_navipath` 目錄下，不是子目錄 |
| `No such file: can_dataset` | 確認 symlink 或 volume 掛對了 |
| transformers tokenizer 錯誤 | `pip install "transformers>=4.40,<5" huggingface-hub==0.36.2` |
| CUDA out of memory | 減小 `--epochs` 或每任務 `--max-train 200` 先跑 1 fold |
| 想要 10-fold | `for FOLD in 1 2 3 4 5 6 7 8 9 10; do python train_qpmil_runner.py --fold $FOLD --order paper --save-ckpt; done` |
