# NaviPath — Onboarding & Re-run Runbook (v1.0)

> 給新加入的 team（目前是我和 AI pair）。讀完這份你能：(1) 看懂跑過什麼、為什麼跑；
> (2) 自己 re-run 任一單一任務；(3) 知道每個結果支撐論文哪個 claim；(4) 知道下一步補什麼。
> 對應論文：`paper/paper_body.tex`；數據：`outputs/*.json`；圖：`outputs/figs/`。

---

## 0. 一句話故事（先記住這個）
在 **frozen 病理基礎模型 + prompt-MIL** 的持續學習裡，分類器不可能遺忘（backbone 凍結，Forgetting≡0 是恆等式）。
但我們在前面加了一個**可訓練的 patch 選擇器（router）**，發現它對**舊任務會「選擇遺忘」（selection forgetting）**：
近期任務選得比 random/prototype/semantic 都好（GO），舊任務卻**比亂選還差**（NO-GO），而且是**自信地選錯**。
我們用 **same-task recency 翻轉** 做乾淨因果證明，用 **lung（樣本最多）** 排除樣本數干擾，
再用 **per-task router 上界（0.33→0.93）** 證明可恢復，**EWC 救不動（0.40）** 指出需要 selection-aware 方法。

---

## 1. 環境（RunPod）
```bash
# 重開機後一行設定（細節見 RUNPOD_SETUP.md）
cd /workspace/src/navipath && bash runpod_setup.sh
# 或手動補套件：
pip install "transformers>=4.40,<5" huggingface-hub==0.36.2 \
    timm einops h5py openpyxl wandb scikit-learn tqdm seaborn pandas pyyaml
```
- 資料：QPMIL 整理好的 4 個 TCGA cohort（lung/brca/rcc/esca）已切好特徵，CONCH 512-d。
- **GPU 確認**：log 裡 `map_location='cpu'` 只是 data loading，模型在 `device=cuda`（log 開頭會印）。
- **長任務務必用 tmux**（SSH 斷線不影響）：`tmux new -s run` → 跑 → `Ctrl+b d` detach → 回來 `tmux attach -t run`。

---

## 2. 任務鏈（Pipeline）— 次序、原因、產物、對應 claim

> 變數：`ORDER ∈ {paper, reverse}`、`FOLD ∈ {1,2,3}`。
> `paper` = lung→brca→rcc→esca（esca 最新）；`reverse` = esca→rcc→brca→lung（lung 最新）。
> 命名規則：`{stage}_{order}_fold{F}.json/.pt`，舊任務預算另存 `oldtask_budget_{order}_f{F}_task0[__{consol}].json`。
> **重跑同名會 overwrite**（router 腳本若 .pt 已存在會 `[skip]`，只重跑 eval）。先備份或改 `--out`。

### 步驟 A — QPMIL baseline + 凍結 backbone（前置，所有後續都依賴它）
- **為什麼**：(1) 產生 Table 1 的 baseline 數字；(2) 產生 router 要掛上去的 frozen backbone ckpt。
- **跑**：
```bash
python train_qpmil_runner.py --order $ORDER --fold $FOLD --epochs 12 --save-ckpt
```
- **產物**：`outputs/qpmil_{order}_fold{F}.json`（ACC/Forgetting/BWT）、`outputs/qpmil_{order}_fold{F}.pt`（← router 的 `--backbone-ckpt`）。
- **支撐**：Table 1（QPMIL 列）、Fig 6 R-matrix（baseline 有輕微真遺忘）。

### 步驟 B — NaviPath decoupled（Table 1 我們這側 + 證明 decoupling 必要）
- **為什麼**：證明 decoupled 設計 Forgetting=0（identity），且非 decoupled 會災難性干擾。
- **跑**：
```bash
python train_navipath.py --config configs/navipath_full.yaml --order $ORDER --fold $FOLD --save-ckpt
```
- **產物**：`outputs/navipath_full_{order}_fold{F}.json/.pt`。
- **支撐**：Table 1（NaviPath 列，Forgetting=0）、§4.2、Fig 6。
  （非 decoupled 的災難數字 0.735/0.950 來自早期 buggy 版本，已記錄於 README/§4.2，不需重跑。）

### 步驟 C — Router 訓練 + patch-budget 評估（**核心**）
- **為什麼**：這是論文主角。一次 run 內：M4 風格訓練 router（每任務 5 epochs），然後在指定任務做 budget 掃描。
- **近期任務（recent，預期 GO）**：`--eval-tasks="-1"`（-1=最後學的任務）
```bash
python train_router_v0.py --backbone-ckpt outputs/qpmil_${ORDER}_fold${FOLD}.pt \
  --order $ORDER --fold $FOLD --eval-tasks="-1" --epochs 5 \
  2>&1 | tee outputs/router_v0_${ORDER}_fold${FOLD}.log
```
- **舊任務（old，預期 NO-GO=selection forgetting）**：`--eval-tasks="0"`（0=最先學的任務）
```bash
python train_router_v0.py --backbone-ckpt outputs/qpmil_${ORDER}_fold${FOLD}.pt \
  --order $ORDER --fold $FOLD --eval-tasks="0" --epochs 5 \
  2>&1 | tee outputs/oldtask_budget_${ORDER}_f${FOLD}_task0.log
```
- **產物**：`router_v0_{order}_fold{F}.json`（recent）、`oldtask_budget_{order}_f{F}_task0.json`（old）。
  每個 json 內含 `results.{router,random,prototype,semantic}` × budgets，與 `go` 旗標。
- **支撐**：Table 2、Fig 2（recent budget）、Fig 3（old budget）、Fig 4（recency flip 由 recent+old 對照而來）。
- **小抄**：`--eval-tasks="-1,0"` 可一次評估最新+最舊。

### 步驟 D — Plan B：router 一致性化（mitigation，Table 3）
- **為什麼**：證明「選擇訊號還在、可恢復」（per-task 上界）且「權重級 EWC 不夠」。
- **跑（reverse、評估最舊 esca=task0；目前 3-fold 完成的就是這條）**：
```bash
# 上界：per-task router
python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold${FOLD}.pt \
  --order reverse --fold $FOLD --eval-tasks="-1,0" --epochs 5 --router-consol pertask \
  2>&1 | tee outputs/router_pertask_reverse_f${FOLD}.log
# 真修法：EWC-on-router（λ 預設 1000）
python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold${FOLD}.pt \
  --order reverse --fold $FOLD --eval-tasks="-1,0" --epochs 5 --router-consol ewc --consol-lam 1000 \
  2>&1 | tee outputs/router_ewc_reverse_f${FOLD}.log
```
- **產物**：`oldtask_budget_reverse_f{F}_task0__pertask.json`、`...__ewc.json`（及 router_v0_reverse_fold{F}__{consol}.json）。
- **支撐**：Table 3、§4.6、Abstract 的 0.33→0.93 / 0.40。

### 步驟 E — 彙整成表 + 出圖（在 Mac 或有 matplotlib 的環境）
```bash
python tools/collect_results.py            # 讀 outputs/*.json → Markdown/CSV 表（Table 1/2/3 數字來源）
python tools/plot_results.py --all         # 產 outputs/figs/P0_*, P0b_*, P1_*, P2contrast_*
python tools/draw_arch.py                  # 產 Fig1_arch.pdf / FigS1_arch.pdf（架構圖）
```
- **支撐**：所有 Fig 與表的數字。圖 PDF 已 copy 到 `paper/figs/` 供 Overleaf。

---

## 3. 目前結果總結（3-fold；論文定稿用）
| 主張 | 數字 | 證據 |
|---|---|---|
| backbone 不遺忘是 identity | NaviPath F=0；QPMIL F=0.017/0.041 | Table 1, Fig 6 |
| router 對 recent 有用 | @64 +2.5–6.7pp，**6/6 GO** | Table 2, Fig 2 |
| 對 old 崩潰（selection forgetting）| esca@64=0.333, lung@64=0.397，**6/6 NO-GO** | Table 2, Fig 3 |
| recency 是因（非樣本/難度）| lung 0.922→0.397；esca 0.956→0.333 | §4.4, Fig 4 |
| 機制=自信選錯 | top-K 聚於 later-task-salient 區 | §4.5, Fig 5 |
| 可恢復但 EWC 不夠 | per-task 0.33→**0.933 (3/3 GO)**；EWC 0.40 (0/3) | Table 3, §4.6 |

---

## 4. 跑過 vs 沒跑（重要！避免誤會）
- ✅ 已完成 3-fold：QPMIL、NaviPath、router recent/old（兩 order）、**Plan B reverse（pertask+ewc）**。
- ✅ 已完成：paper-order **old baseline**（`oldtask_budget_paper_f1-3_task0.json`，none）。
- ⬜ **未跑**：paper-order 的 **Plan B**（lung 當最舊的 pertask/ewc）。→ 見下「未來任務 F2」。
- ⬜ 未跑：EWC λ-sweep；多 slide 機制統計；額外 cohort。

---

## 5. 未來任務鏈（rebuttal 前的補強，按優先序）
- **F1 — EWC λ-sweep**（堵「λ 沒掃好」質疑）：對 reverse f1，掃 `--consol-lam {100,300,1000,3000,10000}`，看 old@64 是否始終 NO-GO。產一張附錄曲線。
- **F2 — paper-order Plan B**（把 Table 3 從 esca-only 擴成 esca+lung 兩任務）：
```bash
for F in 1 2 3; do for C in pertask ewc; do
  python train_router_v0.py --backbone-ckpt outputs/qpmil_paper_fold${F}.pt \
    --order paper --fold $F --eval-tasks="-1,0" --epochs 5 --router-consol $C \
    2>&1 | tee outputs/router_${C}_paper_f${F}.log
done; done
```
- **F3 — 泛化**：加 colon 或更長序列；換 attention-based selector 看現象是否仍在（也是未來方法工作）。
- 完成後：`git add outputs/ && git commit -m "results: <what>" && git push`（先 `git pull --rebase`）。

---

## 6. 常見坑（已踩過）
- push 被拒 `fetch first` → `git pull --rebase origin main` 再 push。
- RunPod 重置 `ModuleNotFoundError` → 跑 §1 安裝。**不要** `pip install -r QPMIL-VL/requirements.txt`（含 conda 套件會失敗）。
- git push 要帳密 → 用 GitHub **PAT** 當密碼。
- 重跑前先確認 `.pt` 是否已存在（會 `[skip]` 只重評估）；要全重跑就先刪該 `.pt` 或改 `--out`。
