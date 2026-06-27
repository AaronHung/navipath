# NaviPath 進度摘要（給教授／team，v0.4）

## 一句話
在 frozen pathology foundation model（CONCH）+ prompt-based 持續學習 WSI 分類上，外掛一個可訓練的 patch router；我們發現並乾淨量化了一個新現象——**selection forgetting（選擇遺忘）**：router 對近期任務選得好，但對久遠任務的「挑 patch 能力」會被遺忘，甚至低於隨機。

## 問題背景
- frozen FM 讓 backbone 不會漂移，傳統的「分類器遺忘」幾乎被消除。
- 但一張切片有數千個 patch，預算受限時必須**選 patch**；這個選擇步驟是可學習的，**它會不會被遺忘，沒人研究過**。
- 我們的設定剛好把「預測（凍結）」與「選擇（router）」分家，能乾淨觀測選擇遺忘。

## 核心發現（3-fold、兩種任務順序）
1. **近期任務：router 有效**。@64 patch 下贏 random/prototype/semantic 約 +2.5–6.7pp（6/6 GO）。
2. **久遠任務：選擇崩潰**。@64 反而**低於隨機**（6/6 NO-GO）。
3. **乾淨因果（招牌證據）**：同一任務、同資料、同 router，只改它在序列中「最新 vs 最舊」，準確率由 ~0.9 翻到 0.33–0.40；連**樣本最多的 lung** 也翻 → 排除「樣本少／難」的混淆。
4. **機制**：舊 router 不是變雜訊，而是「**自信地選錯**」——把重要性押在被後續任務調教過的 patch 群，所以才會低於隨機。

## Plan B：能不能救？
- **per-task router（上界）**：完全恢復——最舊任務 @64 由 0.33 回到 **0.87（＝用全部 patch 的水準）**，且贏 heuristics（GO）。→ **證明訊號一直都在、純粹是遺忘**。
- **EWC-on-router（replay-free 修法）**：**救不動**（@64=0.40，仍 NO-GO）。→ weight-level 正則不足以保護「選擇排序」。
- 結論：需要 **selection-aware 的持續學習**（我們的 future work）。

## 貢獻定位（誠實，先發制人）
- 我們**不主張 ACC 贏**：公平比較下 QPMIL baseline（~0.92）其實 ≥ decoupled NaviPath（~0.88）。
- decoupled 的 `Forgetting=0` 是**設計恆等式（no-op），不是功勞**，我們主動寫明。
- 真正貢獻＝**指出+量化 selection forgetting + 乾淨因果 + 機制 + 可恢復性（上界成立、EWC 不足）**。
- 另有一個有價值的**負面結果**：早期讓 expert 改寫特徵餵回 frozen backbone，造成嚴重干擾（Forgetting 0.735/0.950）→ 佐證 decoupling 的必要性。

## 現況
- 論文**初稿已完成**（Abstract、§1–§6、Method 含 5 公式、Table 1/2、6 張正圖 + 1 張附錄圖、captions）。
- 實驗：baselines、NaviPath、router recent/old、recency flip、機制、R-matrix 全部 3-fold 完成。
- Plan B：reverse fold-1 完成（結論如上）；**fold 2–3 跑中**補成 3-fold。

## 待辦（不含寫作中可能的新發現）
1. Plan B reverse fold 2–3（跑中）→ Table 2 改 3-fold 平均。
2. 可選：paper-order 的 Plan B（補另一個任務 lung 的舊任務修復，證明跨 order 一致）。
3. references BibTeX（投稿前補）。

## 時間
- 目標 COMPAYL（7/1）。數據面 fold 2–3 一回來就齊，其餘為寫作與潤稿。
