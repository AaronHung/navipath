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

## 待你給：Plan B（pertask/ewc）結果 → 填 §4.5 + Table 2 + Abstract Plan B 句

---

# 附錄 A：Figure captions + defend 講稿

### Fig 1 — `P0_router_v0.png`（近期任務 budget 曲線）
**Caption**：Patch-budget accuracy on the **most-recently-learned** task. The learned router
(red) matches/exceeds random, prototype, semantic across budgets; at K=64 it gives **+2.5–6.7 pp**
over the best heuristic (3/3 folds, both orders). Using only 64 of thousands of patches, the
router nearly matches all-patch accuracy.
**Defend**：
- Q：router 只是跟 random 打平？→ 在 tight budget（64/32）router 明顯在上，random 掉下去；@All 大家一樣（無選擇）。
- Q：cherry-pick？→ paper-esca、reverse-lung，各 3 folds，**6/6 GO**。

### Fig 2 — `P0b_recency_flip.png`（**核心**：same-task recency flip）
**Caption**：**Same-task recency flip.** For the *identical* task and test set, the router selects
well when the task was learned **last** (solid, ~0.9) but **collapses below the random baseline**
when the *same* task was learned **first** and then overwritten by later tasks (dashed, 0.33–0.40).
Holds for both lung and esca.
**Defend**：
- Q：esca 是不是太難/太小才崩？→ **lung（test ~95 張，樣本多）同樣翻轉**，排除樣本數/難度。
- Q：是不是 backbone 壞了？→ @All 四法相同（只有「選擇」這一步不同），且 R-matrix（Fig 4）顯示 backbone 不變。
- Q：只差 recency 怎麼確定是因果？→ 同一任務、同資料、同 router，**只改它在序列中的新舊**，結果翻轉 = 乾淨單變因。

### Fig 3 — `P2contrast_esca_fold1.png`（機制：recent vs forgotten router，同一切片）
**Caption**：The *same* esca slide in CONCH feature space (shared t-SNE), scored by the router
when esca was **recent** (paper, left) vs when esca was **old/overwritten** (reverse, right).
Both routers produce structured scores, but they prioritise **different patch sub-populations**:
the forgotten router's top-64 shifts to a region made salient by *later* tasks, which is
sub-optimal for esca. This mis-prioritisation (not noise) explains why router@64 falls **below**
random on old tasks.
**Defend**：
- Q：舊 router 的分數不是退化成雜訊嗎？→ 不是。圖顯示它仍有結構，但**把重要性押在錯的 patch 群**（被後續任務調教過的）→ 這正是「router@64 < random」（自信選錯）而非「= random」（無訊號）的原因。
- Q：只看一張切片代表性？→ 作為 illustrative 機制圖（如 QPMIL Fig 5 亦用示例切片）；量化結論由 Table 2 / Fig 2（6/6 fold）承擔。

### Fig 4 — `P1_r_matrix.png`（backbone decouple → F=0 恆等）
**Caption**：Per-task accuracy R[i,j] (acc on task j after learning task i). NaviPath columns are
**flat by construction** (decoupled frozen backbone → Forgetting=0 is an identity, *not* a
contribution); the QPMIL baseline shows mild genuine drift. We surface this to transparently
locate our contribution in the **selection** analysis, not in a trivial F=0.
**Defend**：主動回答「為何剛好 0」，把它定位成恆等式，避免被當賣點質疑。

---

# 附錄 B：Tables

## Table 1 — Continual accuracy (backbone level, mean±std, 3 folds)
| Method | Order | ACC | Forgetting | BWT |
|---|---|---|---|---|
| QPMIL baseline | paper | 0.924±0.016 | 0.017±0.022 | −0.017±0.022 |
| QPMIL baseline | reverse | 0.917±0.026 | 0.041±0.023 | −0.041±0.023 |
| NaviPath (decoupled) | paper | 0.879±0.030 | 0.000 | 0.000 |
| NaviPath (decoupled) | reverse | 0.886±0.030 | 0.000 | 0.000 |
（註：NaviPath F=0 為 decouple 恆等，見 Fig 4；ACC 為 router-free 全 patch。）

## Table 2 — Router patch selection: recent vs old (router@K, mean over 3 folds)
> 「best heur」= max(random, prototype, semantic)。GO = router > best heur（finite budget）。
> Plan B 兩欄待跑（pertask=上界、ewc=replay-free 修法）。

| Task | Condition (recency) | @256 | @128 | @64 | @32 | best heur@64 | GO | **pertask@64** | **ewc@64** |
|---|---|---|---|---|---|---|---|---|---|
| esca | RECENT (paper, last) | 0.956 | 0.956 | **0.956** | 0.933 | 0.889 | ✓ | — | — |
| esca | OLD (reverse, first) | 0.511 | 0.400 | **0.333** | 0.333 | 0.822 | ✗ | _[待填]_ | _[待填]_ |
| lung | RECENT (reverse, last) | 0.904 | 0.915 | **0.922** | 0.918 | 0.897 | ✓ | — | — |
| lung | OLD (paper, first) | 0.512 | 0.453 | **0.397** | 0.353 | 0.813 | ✗ | _[待填]_ | _[待填]_ |

**讀法**：同任務 RECENT→OLD，router@64 由 0.92–0.96 崩到 0.33–0.40（且 < best heur）。
Plan B 目標：OLD 列的 pertask@64 應回到 ≈ RECENT 水準（上界）；ewc@64 至少 ≥ best heur(~0.8)。

## Table 2-supp — full budget curves（給 appendix，數字見 `collect_results.py` 輸出）
- recent：paper-esca、reverse-lung（GO 3/3）。
- old：reverse-esca、paper-lung（NO-GO 3/3，router < random 全 budget）。
