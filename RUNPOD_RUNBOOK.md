# RunPod 操作 Runbook（隨時看，跑系統用）

> 這是**操作手冊**（怎麼登入、開機 setup、tmux 跑、斷線重連、同步結果、踩坑）。
> **實驗要跑什麼指令**看當前 SOP：`specs/01_sop_navipath-cl_phase0.md`（N1/N2…）。
> 萃取自 `legacy/ONBOARDING_runbook.md` + `legacy/RUN_TASK2_tmux.md` 並更新到目前架構。

```bash
# 全程用這個變數（依你的 pod 調整）
REPO=/workspace/src/navipath
```

---

## 0. 每日標準流程（TL;DR）

1. SSH 進 pod → `cd $REPO`
2. **開機 setup**（重置過才需要）：見 §2 一行指令
3. `git pull --ff-only`（取最新 code）
4. **tmux 跑實驗**（§3）→ `Ctrl+b d` detach，可關電腦
5. 跑完它自己 `git push` → **Mac 端 `git pull`** 看結果（§4 同步機制）

---

## 1. 登入 RunPod

- 控制台租 GPU（建議 RTX 4090 24GB / A100 40GB），Image 選 PyTorch 2.x。
- **資料用 Network Volume**（跨機器保留），掛到固定路徑；每次開機選同一個 Volume。
- Connect → SSH（或 WezTerm SSH）。拿到 `host:port`。

---

## 2. 開機 setup（重置環境後，跑任何實驗前先貼）

```bash
cd $REPO && pip install -q "transformers>=4.40,<5" huggingface-hub==0.36.2 \
  timm einops h5py openpyxl wandb scikit-learn tqdm seaborn pandas pyyaml && \
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

- 看到 `cuda available: True` 才開跑。
- 程式與結果都在 git，重裝只補這些套件。
- **不要** `pip install -r QPMIL-VL/requirements.txt`（含 conda 套件會失敗）。
- CONCH 權重（~1GB，`*.bin` 不進 git）需自行放到 `conch_ckpt_path`：
  ```bash
  huggingface-cli login   # 需先在 HF 申請 MahmoodLab/CONCH 存取
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download('MahmoodLab/CONCH','pytorch_model.bin', local_dir='$REPO/checkpoints/conch')"
  ```

---

## 3. tmux 跑法（出差/斷線不中斷）

**為什麼**：detach 後可關電腦/斷網，程式繼續跑、跑完自己 push。

```bash
# 1) 確保 tmux + 開 session
command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux)
tmux new -s run

# 2) 在 tmux 裡跑（範例樣板；實際指令照 SOP）
cd $REPO && git pull --ff-only && set -o pipefail && \
  python <實驗腳本> <參數> 2>&1 | tee outputs/<canonical_name>.log && \
  echo "=== RUNS DONE, pushing ===" && \
  git add outputs/ && \
  git commit -m "results: <what> ($(date +%F))" && \
  git pull --rebase && git push && \
  echo "=== PUSHED ==="

# 3) Detach（背景跑）：按 Ctrl+b 放開，再按 d
```

**斷線回來：**

```bash
tmux attach -t run      # 看到 "=== PUSHED ===" 即完成；看完 Ctrl+b d 離開（別直接關程式）
tmux kill-session -t run   # 跑完想清掉 session
```

---

## 4. 同步機制（我們的契約）

**RunPod 出，Mac 入。結果只走 git（`outputs/`），不 scp。**

```bash
# RunPod（跑完）：
cd $REPO && git add outputs/ && git commit -m "results: <what>" && git pull --rebase && git push

# Mac（我這端 check）：
cd ~/research/01_navipath && git pull
```

- 你跑完 push 後，跟我說「跑完了」，我從 Mac pull 下來做結果 check。
- **檔名照 SOP 命名規則、不覆蓋舊 canonical**（重跑加 `__rerun_YYYYMMDD_HHMM`）。
- 大檔（`*.pt`/`*.bin`/資料）依 `.gitignore` 不進 git；只有 `outputs/` 的 json/log（必要時小 pt）走 git。

---

## 5. Git 認證（換新 pod 時）

push 要帳密就用 **GitHub PAT** 當密碼：

```bash
git config user.email "you@example.com" && git config user.name "Aaron"
git remote set-url origin https://<YOUR_GITHUB_TOKEN>@github.com/<owner>/<repo>.git
```

---

## 6. 踩坑清單（已踩過）

- **push 被拒 `fetch first`** → `git pull --rebase` 再 push。
- **`ModuleNotFoundError: navipath_moe`** → 確認 `cd $REPO`（在 repo 根目錄，不是子目錄）；必要時 `PYTHONPATH=. python ...`。
- **CUDA OOM** → 減 `--epochs` 或加 `--max-train` 先小跑 1 fold。
- **QPMIL-VL 已 vendored（收進 repo）** → clone 即有，**不要**另外 clone upstream（upstream: can-can-ya/QPMIL-VL @ 3a7a769, AAAI 2025）。
- **⚠️ 第一次 pull 到含 QPMIL-VL 的版本會卡 `untracked working tree files would be overwritten`**（pod 本地有 untracked QPMIL-VL）：
  ```bash
  cd $REPO && git fetch origin main
  git show origin/main:QPMIL-VL/configs/main.yaml > /tmp/vendored_main.yaml
  diff /tmp/vendored_main.yaml QPMIL-VL/configs/main.yaml
  # 差異「只該在路徑」(dataset_root_dir / conch_ckpt_path / class_ensemble_path)，超參數須一致才安全
  rm -rf QPMIL-VL && git pull --rebase origin main   # 確認安全後才移除本地副本
  ```
- **⚠️ `QPMIL-VL/configs/main.yaml` 是絕對路徑**：每台機器要自己改這三行指到本機資源，**超參數（epochs/lr/wd）不要動**（公平性基準）：
  - `dataset_root_dir` → 本機資料 `data/`
  - `conch_ckpt_path` → CONCH 權重
  - `class_ensemble_path` → `class_ensemble.json`

---

## 7. 相關檔

- 實驗指令／里程碑：`specs/01_sop_navipath-cl_phase0.md`
- 深度首次 setup（資料上傳/volume）：`RUNPOD_SETUP.md`、`RUNPOD_SOP.md`
- 架構／設計：`site/architecture.html`、`docs/wiki/`
- 凍結 backbone ckpt（router/agent 掛載用）：`outputs/qpmil_{order}_fold{F}.pt`

---

---

---

> ## NaviPath Research RunPod Commands

---

全部就緒並 push 完成。下面是 N2 的完整 step-by-step（也存成 `specs/02_n2_runpod_commands.md`，RunPod pull 得到）。

## 先說明：N2 我多做了一步

N2 要在 RunPod 跑真數字，需要一支把 `sequential_observation` 接上**真 QPMIL backbone + 真資料 + 算 metrics** 的驅動腳本。我剛寫好了 `eval_sequential_observation.py`，它一次跑 **4 個 mode 的格點**：`nsm_seq / nsm_oneshot / nonsm_seq / nonsm_oneshot`（有/無 NSM × 序列/一次性）——正好涵蓋你要的三組實驗。Mac 上已驗證：ruff 乾淨、能正常啟動、CONCH 載入正常（只是 CPU 慢，所以真跑交給 RunPod）。已 push（`c104ec9`）。

## N2 你要 follow 的步驟（RunPod）

**Step 0｜進 pod、setup、pull**

```bash
REPO=/workspace/src/navipath; cd $REPO
pip install -q "transformers>=4.40,<5" huggingface-hub==0.36.2 timm einops h5py openpyxl wandb scikit-learn tqdm seaborn pandas pyyaml && python -c "import torch; print('cuda', torch.cuda.is_available())"
git pull --ff-only
```

**Step 1｜體檢**

```bash
ls -1 outputs/qpmil_reverse_fold{1,2,3}.pt
python eval_sequential_observation.py --help | head -3
nvidia-smi | head -12
```

**Step 2｜GPU 快速 smoke（先確認會動）**

```bash
python eval_sequential_observation.py --backbone-ckpt outputs/qpmil_reverse_fold1.pt \
  --order reverse --fold 1 --eval-task 0 \
  --epochs 1 --max-train 8 --max-eval 4 --budgets 0,16,8 --step-size 4 --out outputs/_smoke
rm -rf outputs/_smoke
```

> 印出 4 個 mode 的 `{budget: acc}` 就 OK。報錯就停、貼我。

**Step 3｜正式跑（reverse 3 折，tmux，跑完自動 push）**

```bash
tmux new -s n2
```

tmux 裡貼整段：

```bash
cd $REPO && git pull --ff-only && set -o pipefail && \
for F in 1 2 3; do \
  python eval_sequential_observation.py \
    --backbone-ckpt outputs/qpmil_reverse_fold${F}.pt \
    --order reverse --fold ${F} --eval-task 0 \
    --epochs 5 --budgets 0,128,64,32,16 --step-size 16 --redundancy 0.5 \
    --skill-bank-out outputs/skill_bank_reverse_f${F}.pt \
    2>&1 | tee outputs/seqobs_reverse_f${F}_task0.log || break; \
done && \
echo "=== N2 RUNS DONE, pushing ===" && \
git add outputs/ && \
git commit -m "results(N2): seqobs reverse f1-3 task0 $(date +%F)" && \
git pull --rebase && git push && echo "=== N2 PUSHED ==="
```

然後 `Ctrl+b` → `d` detach，可關電腦。

**Step 4｜回來 + 通知我**

```bash
tmux attach -t n2     # 看到 "=== N2 PUSHED ===" 即完成
```

跟我說「N2 跑完了」→ 我 Mac `git pull` 做 **N3**（彙整 + 出圖 + check）。

## 怎麼看算成功（N3 會正式整理）

- **CL 保留（核心）**：old-task 上 `nsm_*` 應明顯 > `nonsm_*`（技能記憶能恢復「該看哪」）。
- **Budget 效率**：`acc@64/32` 接近 `acc@All`。
- **Agentic**：`*_seq` vs `*_oneshot` 同 budget 的差異。

## 提醒

- **預估 1–1.5 小時**（3 折）。想先試水：把 `for F in 1 2 3` 改成只 `F=1`，確認數字合理我再喊跑 2、3。
- 不覆蓋 canonical（重跑加 `__rerun_時間`）；失敗即止，不硬跑。

要不要我**先讓你只跑 fold 1**（約 20–30 分）確認數字方向對，再跑 2、3？我建議這樣最穩。
