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
cfg["dataset_root_dir"]    = "/workspace/datasets/can_dataset"
cfg["class_ensemble_path"] = "/workspace/src/navipath/QPMIL-VL/class_ensemble/class_ensemble.json"
cfg["conch_ckpt_path"]     = "/workspace/src/navipath/checkpoints/conch/pytorch_model.bin"
yaml.dump(cfg, open("QPMIL-VL/configs/main.yaml","w"), allow_unicode=True)
print("config updated")
EOF

sed -i "s|data_root: .*|data_root: /workspace/can_dataset|" configs/*.yaml

# 驗證
PYTHONPATH=. python tests/test_shapes.py
```

### Step 4：跑實驗

> **核心概念**：NaviPath 的實驗分兩種，必須清楚區分：
>
> | 類型 | 跑什麼 | 要 GPU 多久 | 輸出什麼 |
> |---|---|---|---|
> | **訓練（一次性）** | 訓練 MicroRouterV0，儲存 NSM Skill Bank | 3–5 小時（3 fold） | `skill_bank_*.pt` |
> | **Inference-only（重複跑）** | 載入已存 skill bank，只改推論參數（λ/budget 等）掃描 | 幾分鐘/組 | json 結果 |
>
> **原則：router 只要訓練一次、存好 skill bank；之後所有參數掃描都用 `--skill-bank-in` 跳過訓練。**

```bash
cd $REPO   # /workspace/src/navipath

# ── M1：QPMIL backbone 訓練（凍結 backbone，這是整個實驗的地基）─────────────
# 一次性，約 1-2 小時，輸出 outputs/qpmil_{order}_fold{F}.pt
python train_qpmil_runner.py --order paper   --fold 1 --save-ckpt 2>&1 | tee outputs/run_paper_f1.log
python train_qpmil_runner.py --order reverse --fold 1 --save-ckpt 2>&1 | tee outputs/run_reverse_f1.log

# ── N2：router + NSM Skill Bank 訓練（一次性，約 1-1.5 小時 / fold）──────────
# 訓練 4 個任務的 MicroRouterV0，每任務 router snapshot 存入 NavigationSkillBank
# --skill-bank-out：存下來，之後不用再訓練
python eval_sequential_observation.py \
    --backbone-ckpt outputs/qpmil_reverse_fold1.pt \
    --order reverse --fold 1 --eval-tasks 0,1,2,3 \
    --epochs 5 --budgets 0,128,64,32,16 --step-size 16 \
    --skill-bank-out outputs/skill_bank_reverse_f1.pt \
    --out outputs/n2_reverse_f1 \
    2>&1 | tee outputs/n2_reverse_f1.log

# ── 參數掃描（inference-only，幾分鐘/組，不需要 GPU 長時間）────────────────
# 已有 skill bank → 用 --skill-bank-in 跳過訓練，只改 λ/budget 等參數
# 永遠不要在掃描時重新訓練！
for LAMBDA in 0.0 1.0 2.0 4.0; do
  python eval_sequential_observation.py \
    --order reverse --fold 1 --eval-tasks 0,1,2,3 \
    --budgets 0,64,32,16 --step-size 16 \
    --redundancy $LAMBDA \
    --normalize-base true \
    --redundancy-mode maxsim \
    --skill-bank-in  outputs/skill_bank_reverse_f1.pt \
    --out outputs/routeA_sweep/lambda_${LAMBDA} \
    2>&1 | tee outputs/routeA_sweep/lambda_${LAMBDA}.log
done
```

```bash
# ── M3：patch-budget 表（不需訓練，直接跑 inference）─────────────────────────
for TASK in 0 1 2 3; do
  python run_patch_budget.py --ckpt outputs/qpmil_paper_fold1.pt \
      --order paper --task-index $TASK 2>&1 | tee outputs/budget_task${TASK}.log
done

# ── M4：MicroRouter v0 go/no-go 驗證────────────────────────────────────────
python train_router_v0.py \
    --backbone-ckpt outputs/qpmil_paper_fold1.pt \
    --order paper --fold 1 --epochs 5 2>&1 | tee outputs/router_v0_f1.log
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

## Step 4.5：確認 skill bank 存在（inference-only 的前提）

```bash
# 確認 N2 訓練結果是否已存在（每 fold 一份 .pt）
ls -lh outputs/skill_bank_reverse_f*.pt

# 如果只有 fold 1，想重用：
#   --skill-bank-in outputs/skill_bank_reverse_f1.pt

# 如果沒有（新 pod 或被刪了）→ 重跑 N2 訓練那段（Step 4 N2 區塊）存一次
```

> **skill bank 不走 git**（太大），跨 pod 要用 `scp` 或 Network Volume 保留。

---

## 常見問題

| 問題 | 解法 |
|---|---|
| `ModuleNotFoundError: navipath_moe` | 確認在 `$REPO` 目錄下，執行 `cd $REPO` |
| `No such file: can_dataset` | 確認 symlink 或 volume 掛對了 |
| transformers tokenizer 錯誤 | `pip install "transformers>=4.40,<5" huggingface-hub==0.36.2` |
| CUDA out of memory | 減小 `--epochs` 或加 `--max-train 200` 先小跑 |
| λ sweep 重複在訓練 | 沒帶 `--skill-bank-in`！加上後會完全跳過訓練 |
| skill bank .pt 不見了 | 新 pod 需重跑 N2 訓練那段重新存，或用 Network Volume 保留 |
| 想要 3-fold | `for F in 1 2 3; do python eval_sequential_observation.py ... --fold $F --skill-bank-out outputs/skill_bank_reverse_f${F}.pt; done` |
