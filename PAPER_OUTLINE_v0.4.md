# Paper Outline (v0.4) — 誠實版 A（COMPAYL workshop）

> 主線：在 frozen-FM 持續學習 WSI 分類上外掛 trainable patch router——對近期任務有效，
> 但「挑選能力」本身會被遺忘（selection forgetting）。乾淨對照 + 排除混淆（+ Plan B：上界/修法）。
> 證據與數字見 [ROUTER_FORGETTING_v0.4.md](ROUTER_FORGETTING_v0.4.md)；圖在 `outputs/figs/`。
> Plan B 結果回來後填 §Results 的 Table 2 與 §Mitigation。

---

## 標題候選
1. **Selection Forgetting: Continual Patch Routing Forgets Old Tasks in Frozen-Foundation-Model WSI Classification**
2. On the Limits of Trainable Patch Selection for Continual Whole-Slide Image Classification
3. Your Router Forgets Too: Recency-Dependent Patch Selection in Continual Pathology MIL

（主推 1：直接點出 "selection forgetting" 這個新詞與設定）

---

## Abstract（草稿，待 Plan B 數字微調）
- 背景：frozen pathology FM（CONCH）+ prompt-based MIL（QPMIL）做 class-IL WSI 分類；patch budget 下需要選 patch。
- 我們把一個輕量 trainable router 接上去做 patch 選擇，並與 random/prototype/semantic 比較。
- 發現 1：router 對**最近學的任務**選 patch 顯著優於 heuristics（@64 +2~7pp，6/6 複現）。
- 發現 2：對**久遠任務**，router 選擇能力崩潰，@64 反而**低於 random**（6/6 NO-GO）。
- 關鍵證據：**same-task recency flip**——同一任務（lung/esca），只改它在序列中是新是舊，router 由 0.9 翻到 0.3–0.4；lung（樣本充足）亦崩 → 排除樣本數/難度混淆。
- 機制：router 對舊任務的 score 分布退化（feature-space 可視化）。
- （Plan B）per-task router 上界證明訊號仍在；EWC-on-router 為 replay-free 緩解 [結果待填]。
- 結論：在 frozen-FM 上加 trainable 選擇/路由，需正視「選擇本身也會被遺忘」。

---

## 1. Introduction
- WSI + MIL；frozen FM（CONCH）省算力、避免遺忘 → 但 patch 多、需要選 patch（compute/budget）。
- 持續學習文獻聚焦**分類器**遺忘；**選擇/注意力機制的遺忘**少被討論。
- 我們的設定剛好把兩者分離：backbone 凍結（不會忘，見 R-matrix），只有 router 在學 → 可乾淨觀測「選擇遺忘」。
- 貢獻：(i) 指出並量化 selection forgetting；(ii) same-task recency flip 的乾淨因果證據 + 排除混淆；(iii) 機制可視化；(iv) Plan B 上界與 replay-free 緩解 [待填]。

## 2. Related Work
- Continual learning（regularization/distillation/replay；EWC/LwF）。
- Frozen-FM / prompt-based continual（QPMIL-VL 等）。
- MIL patch selection / attention / budget inference。
- 缺口：沒人談「持續學習下 patch 選擇器本身會遺忘」。

## 3. Setup / Method
- Backbone：CONCH（frozen）+ QPMIL prompts；class-IL，backbone 對預測路徑 decoupled（→ Forgetting=0 是恆等，誠實說明，Fig 4）。
- Router：MicroRouterV0（132K，patch→scalar score→Top-K），continual 訓練、backbone 凍結。
- 選擇基線：random / prototype / semantic。
- 評估：4 TCGA 任務（lung/brca/rcc/esca），2 orders（paper/reverse），3 folds；budget = All/256/128/64/32；指標 ACC + GO/NO-GO 判準。
- （Plan B）RouterEWC（diagonal Fisher）+ per-task router（上界）。

## 4. Experiments / Results
- **4.1 近期任務：router 有效（Fig 1）** — paper-esca、reverse-lung 6/6 GO。
- **4.2 舊任務：selection forgetting（Fig 2a/P0_oldtask）** — 6/6 NO-GO，router@64 < random。
- **4.3 Same-task recency flip（Fig 2/P0b，核心）** — lung 0.89→0.40、esca 0.93→0.33；lung 樣本多亦崩 → 排除混淆。
- **4.4 機制（Fig 3/P2-lite）** — router 對舊任務 score 退化/反向。
- **4.5 Mitigation（Table 2，Plan B）[待填]** — per-task 上界恢復 GO（證明是遺忘）；EWC 部分/全部恢復 or 不足。

### 圖表清單
| 元件 | 檔案 | 說明 |
|---|---|---|
| Fig 1 | `outputs/figs/P0_router_v0.png` | 近期任務 budget 曲線，router>heuristics |
| Fig 2 | `outputs/figs/P0b_recency_flip.png` | **核心**：same-task recency flip |
| Fig 2-supp | `outputs/figs/P0_oldtask_budget.png` | 舊任務 budget 曲線（router 崩） |
| Fig 3 | `outputs/figs/P2lite_*.png` | 機制：router score 特徵空間分布 |
| Fig 4 | `outputs/figs/P1_r_matrix.png` | backbone decouple → F=0 恆等（誠實說明）|
| Table 1 | `collect_results.py` accuracy | ACC/Forgetting（backbone 層級）|
| Table 2 | `collect_results.py` budget + Plan B | router vs heuristics × {recent,old} × {none,pertask,ewc} |

## 5. Discussion / Limitations
- selection forgetting vs classifier forgetting 的關係。
- 為何 decouple 後 backbone F=0（恆等，不是貢獻）——主動誠實說明。
- 限制：4 任務、單資料源、esca 小樣本（但 lung 大樣本佐證）；router 為 scalar v0。
- 若 EWC 不足：說明 weight-level 正則不足以救 selection forgetting，為 future work。

## 6. Conclusion
- 一句話：trainable 選擇/路由在 frozen-FM 持續學習中，對近期有效但會遺忘舊任務的選擇；需要 selection-aware 的持續學習。

---

## 待我補：每張圖 caption + defend 講稿（下一步寫，放本檔 §附錄或獨立檔）
## 待你給：Plan B（pertask/ewc）結果 → 填 §4.5 + Table 2 + Abstract Plan B 句
