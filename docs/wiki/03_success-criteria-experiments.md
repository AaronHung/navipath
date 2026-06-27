# 03 · 成功判準與實驗設計（POC 怎樣算成功）

> 回答：什麼數字算成功？跟誰比？為何不用贏 QPMIL？要不要做 SOTA？

## 0. 最關鍵的觀念：換比較的「軸」

我們**不在「full-observation 準確率」這條軸上和 QPMIL 拚**——那會輸，也沒意義（QPMIL@All 是上界）。
我們開一條**新軸**：**budgeted + continual + interpretable navigation**。
- QPMIL@All = **upper bound / oracle**，不是對手。
- 我們的價值：用**少數 patch** 逼近上界、**跨任務不忘**、且**可解釋（看哪裡）**。

## 1. 什麼數字算 POC 成功（三個要看的指標）

### 指標 A — Budget 效率（navigation 有挑對地方）
- `acc@K`（例如 K=64）**逼近** `acc@All`，gap 小。
- 解讀：只看少數 patch 就接近全看 → 我們的 navigation 有效。

### 指標 B — CL retention（**核心 CL 成功訊號**）
- 學完新任務後，**舊任務的 `acc@K`** 是否維持：
  - **有 NSM/CNL** → 舊任務 acc@K ≈ 它單獨訓練時的水準（**保住**）。
  - **無 NSM（naive 連續更新）** → 舊任務 acc@K 崩（接近我們舊證據的 0.133 那種**主動選錯**）。
- 解讀：**recency / 舊任務崩潰被修好** = CL 成功的最直接證據。

### 指標 C — Sequential > One-shot（Agent 性的證據）
- 序列多步觀察，在**相同 K** 下 acc 更高，或**更少 patch** 達到同樣 acc。
- 解讀：證明「會累積 state 的多步 agent」優於「一次性挑 K」（甩開選擇器指控）。

> 一句話總結成功：**「用 64 個 patch 逼近全看；學新癌種後舊癌種仍知道該看哪；序列比單步省。」**

## 2. 跟誰比（baselines，新問題沒有現成 SOTA）

| 角色 | 對象 | 預期 |
|---|---|---|
| **上界** | QPMIL@All（全看） | 我們逼近它即可，不需超過 |
| **同 budget 下界** | random / prototype / semantic 選 @K | 我們應**贏** |
| **遺忘 baseline** | naive 連續更新（無記憶） | 舊任務崩，凸顯我們的 retention |
| **標準 CL baseline** | regularization（如 EWC）套在 navigation 上 | 當「新問題的 baseline」，**不是**「機制比較」 |
| **Hero（我們）** | CNL = 序列觀察 + NSM + oracle gate | 兼顧 budget 效率 + retention |

> 注意措辭：EWC 等只是「**新問題的對照 baseline**」，**不要**寫成「比較哪種機制好」。

## 3. 自己比 vs 跟別人比（ablation 策略）

- **跟別人比**：對 QPMIL@All（上界）、heuristic@K（下界）。立位置，不爭 SOTA 準確率。
- **自己比（ablation = 設計佐證，主力）**：
  - 序列 vs 單步觀察（證明 Agent 性）。
  - 有 / 無 NSM（證明 CL retention）。
  - Observation State 組成（哪些證據重要）。
- 舊的 `MECHANISM_SELECTION` / `mechanism_table.py` 數字 → **重新利用**成上面的 ablation / motivation，不浪費、不主打。

## 4. 我們的「SOTA」是什麼意思（給老師）
不是「準確率打敗 QPMIL」。而是：
> **第一個定義並解決「budgeted WSI navigation policy 的持續學習」；在低 budget 下逼近全看上界，且跨任務保住 navigation 能力。**

## 5. Motivation 一句話（為何有人要用）
即使 QPMIL 分類很強，真實部署要**少看、可解釋、且面對不斷新增的癌種/器官**。
我們提供一層**可持續學習的導覽能力**，讓 frozen 診斷 backbone 能在 **budget 受限**下、**跨任務**、**可解釋地**運作——這是 QPMIL 沒處理的維度。

## 待你 / 老師確認的開放點
- 把 EWC 當「baseline」可接受嗎？（標準做法，但你對舊詞敏感，故標記）
- budget K 的代表值（64？）與任務流順序，沿用既有設定即可。
