# COLLAB PLAYBOOK — 跨 session「湯底」（每次換 chat 都要帶）

> 這份是**和具體研究主題無關**的協作 skill memory：我（Aaron）和 AI pair 怎麼一起工作。
> 換 chat / 換 model / pivot 都不會變的共通規則。**新 session 第一件事：讀本檔 + `SESSION_CONTEXT.md`。**
> 專案進度/主張看 `SESSION_CONTEXT.md`；怎麼跑實驗看 `ONBOARDING_runbook.md`；本檔只講「合作方式」。

---

## 0. 黃金鐵則（最常違反，最重要）

1. **兩台機器、一份 code**：Mac M1（MPS）開發 + smoke test；RunPod（CUDA）跑重活。**同一份 git code**，靠 device 自動選擇相容，不准為某台機器分叉。
2. **AI 在 Mac 上產出「可直接 copy/paste 到 RunPod 跑」的指令**：每段給我的指令要能整段貼進 RunPod 的 tmux 就跑，不需我再改。
3. **成本紀律(最高優先)**：預設最小 context。讀檔前先說「要讀哪個檔、為什麼」，只讀最小子集；用 pinpoint search（函式/類別名、error string、symbol），**不要掃整個 repo**。有歧義就做 1–2 個假設往下做，不要連問。
4. **先讀、不要改**：拿到任務先讀 code，講清楚要動哪些檔，**小步 patch**，附 shape/smoke test，再跑 debug。
5. **失敗即止**：任一步失敗就停、貼 error、提**最小修補**等我批准，不要連環改、不要捏造結果。
6. **語言**：繁中輸入→zh-TW 回答；英文技術名詞保留原文，必要時括號加中文註解。

---

## 1. 開發環境（device-agnostic 寫法）

- **一律自動選 device**：`cuda > mps > cpu`。專案內用
  `from navipath_moe import get_device, setup_mps; setup_mps(); dev = get_device()`。
- **MPS 坑**：用 `float32`（勿 float64）；`setup_mps()` 已開 CPU fallback；`map_location='cpu'` 只是 load data，模型仍在 `dev`（log 開頭會印 device）。
- **誰跑什麼**：
  - **Mac M1**：開發、單 fold、overfit、shape test、第一張表、smoke。資料很小（單張切片 ≤8000 patch×512×4B ≈16MB），M1 16G 夠。
  - **RunPod CUDA**：平行吞吐才上 → 10-fold CV、多 seed、超參 sweep。

---

## 2. Mac → RunPod 標準流程（copy/paste 文化）

**(A) Mac 端：改 code → smoke → push**
```bash
# smoke 一律寫到丟棄目錄，禁止碰 canonical 結果
ruff check .
python <script>.py --help
python -c "import <module>"
python <script>.py ... --epochs 1 --max-train 16 --max-eval 8 --out outputs/_smoke
rm -rf outputs/_smoke
git add <只加改到的檔> && git commit -m "..." && git pull --rebase && git push
```

**(B) RunPod 端：重開機 setup（一行，裝缺套件 + 確認 GPU）**
```bash
cd /workspace/src/navipath && pip install -q "transformers>=4.40,<5" huggingface-hub==0.36.2 timm einops h5py openpyxl wandb scikit-learn tqdm seaborn pandas pyyaml && python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```
- 看到 `cuda available: True` 才開跑。**不要** `pip install -r QPMIL-VL/requirements.txt`（含 conda 套件會炸）。

**(C) RunPod 端：長任務一律 tmux（斷線/出差不掛）**
```bash
command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux)
tmux new -s run
# 在 tmux 裡貼「一整段 && 串起來、跑完自己 git push」的指令（見下範式）
# detach：Ctrl+b 放開再按 d ；回來：tmux attach -t run ；清掉：tmux kill-session -t run
```

**(D) 「跑完自己 push」的 tmux 範式（出差版）**
```bash
cd /workspace/src/navipath && git pull --ff-only && set -o pipefail && \
python <cmd1> 2>&1 | tee outputs/<log1>.log && \
python <cmd2> 2>&1 | tee outputs/<log2>.log && \
echo "=== RUNS DONE, pushing ===" && \
git add outputs/ && git commit -m "results: <what>" && git push && \
echo "=== PUSHED ==="
```
跑完我在 Mac `git pull` 就有結果。

---

## 3. 測試鐵則（每步必跑，省 debug token）

1. **Shape test**：`PYTHONPATH=. python tests/test_shapes.py`（先測形狀，別先跑 full train）。
2. **Overfit-4**：4 張切片 200 步，loss 要降、train ACC→~100%。降不了 = loss 沒接 / label 錯 / features 錯 / optimizer 沒含新參數 / 無梯度。
3. **Param check**：印 `requires_grad` 參數，確認**只有**該訓練的模組在學（如 `micro.* / experts.*`），backbone 應凍結。
4. **No-collapse check**（有 routing/expert 時）：每 task 後印 `w.mean(0)`，不應 `[0.99,0,0,0.01]`。
5. **Sanity 單調**：random selector 的 ACC 應隨 K 單調上升；不單調 = Top-K indexing / 子集沒正確送進 backbone。

---

## 4. Git / 結果管理規矩

- **push 前先** `git pull --rebase`；被拒 `fetch first` → 同樣 `git pull --rebase origin main` 再 push。
- **要帳密** → 用 GitHub **PAT** 當密碼：`git remote set-url origin https://<PAT>@github.com/AaronHung/navipath.git`。
- **大檔不進 git**：`*.pt/*.bin/*.pth`、資料、CONCH 權重走 Network Volume / scp / HF Hub，不走 git（見 `.gitignore`）。
- **結果 `outputs/*.json` 進 git**（小、是證據）；commit 時 `git status` 確認**只有 outputs/**，沒夾帶 code。
- **防覆蓋**：canonical 檔已存在就 `[skip]`；重跑用 `__rerun_YYYYMMDD_HHMM` 後綴，不動 canonical。先備份或改 `--out`。
- **命名規則**：`{stage}_{order}_fold{F}.json/.pt/.log`；回看舊任務 `oldtask_budget_{order}_f{F}_task{t}.json`。
- 每個 TASK 結束更新 `outputs/PROGRESS.md`。

---

## 5. 任務交付給 AI 的標準模板（貼這個）

```
你在我的 <Mac repo /Users/aaron/research/01_navipath | RunPod /workspace/src/navipath> 協助。
逐項做，任一失敗就停、貼 error、提最小修補等我批准。不捏造、不重寫既有主流程。
[背景] <一句話>
[要動的檔] <列出，理由一句>
[步驟] 1)... 2)... （小步）
[smoke] <寫到 outputs/_smoke，跑完 rm；或 --help / import 檢查>
[commit] git add <檔> && git commit -m "..." && git pull --rebase && git push
```

---

## 6. Model 選擇（省 token）

- **動 code / 寫核心邏輯 / 設計**：Opus 4.8 High（在 Cursor 寫程式、smoke test）。
- **純跑指令 / pull / commit / 盤點**：Sonnet 4.6 medium 即可，省 token。
- 構思、畫架構圖、討論在 Mac 上；可跑的東西貼給我，我去 RunPod tmux 跑。

---

## 7. 不要做（踩過的雷）

- 不要為了「省 compute」而在 encode-all 後談 Top-K 省算力（CONCH 已抽完）。
- 不要 `pip install -r QPMIL-VL/requirements.txt`（conda 套件）。
- 不要在 RunPod 第一次 pull 直接覆蓋 untracked `QPMIL-VL/`（先 diff configs/main.yaml，確認只差路徑再 `rm -rf QPMIL-VL && git pull --rebase`）。
- 不要動 `QPMIL-VL/configs/main.yaml` 的超參（epochs/lr/wd）——那是公平性基準；**每台機器只改三條絕對路徑**（dataset_root_dir / conch_ckpt_path / class_ensemble_path）。
- 不要把臨時 clone（`.navipath_inspect/`）或大檔 commit 進 git。

---

## 8. 一眼地圖：換 session 先讀哪幾份

| 檔案 | 作用 |
|---|---|
| `COLLAB_PLAYBOOK.md`（本檔）| **湯底**：怎麼合作（device/tmux/git/測試/成本）。 |
| `SESSION_CONTEXT.md` | **接手文件**：目前主軸、架構、6 天計畫、風險。 |
| `ONBOARDING_runbook.md` | 怎麼跑每條實驗、跑過什麼、對應哪個 claim。 |
| `RUNPOD_SETUP.md` / `RUNPOD_SOP.md` / `RUN_TASK2_tmux.md` | RunPod 開機、SOP、tmux 出差跑法。 |
| `KICKOFF_PLAYBOOK.md` | Milestone 級 code/test/debug prompt 範本。 |
| `CODEBASE_MAP.md` | QPMIL-VL 結構與插入點。 |
```
