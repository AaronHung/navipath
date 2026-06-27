# Paper Outline (v0.4) — 誠實版 A（COMPAYL workshop）

> 主線：在 frozen-FM 持續學習 WSI 分類上外掛 trainable patch router——對近期任務有效，
> 但「挑選能力」本身會被遺忘（selection forgetting）。乾淨對照 + 排除混淆（+ Plan B：上界/修法）。
> 證據與數字見 [ROUTER_FORGETTING_v0.4.md](ROUTER_FORGETTING_v0.4.md)；圖在 `outputs/figs/`。
> Plan B 結果回來後填 §Results 的 Table 2 與 §Mitigation。

---

## 0. 進度與依賴關係（先讀這個，避免腦袋亂）

### 一句話 storyline（定位整篇）
> 公平比較下 **QPMIL baseline ACC ~0.92 ≥ NaviPath decoupled ~0.88**——**我們的貢獻不是 ACC**，
> 而是首次指出並乾淨量化「**selection forgetting**：trainable patch router 對近期任務有效，
> 但對久遠任務的『挑選能力』會被遺忘」。decoupled 的 `Forgetting=0` 是恆等式（no-op），
> 我們主動誠實說明，把貢獻明確放在「選擇分析」而非 trivial 的 F=0。

### 已完成（鎖定，不會再變）
- ✅ **公平 baseline（reviewer 問題2）**：QPMIL baseline 與 NaviPath **完全對齊**——
  `QPMIL-VL/configs/main.yaml` `epochs:[12,12,12,12]`、`adam_lr 1e-3`、`adam_weight_decay 5e-4`；
  NaviPath config 同值；6 份 `outputs/qpmil_*.json` 全部 `epochs_override:0`（確跑滿 12ep）。
  → 當初 critique 的「reverse 0.594 / 5 epochs」是**舊跑，已被取代**。**不需重跑、不需別人 PDF。**
- ✅ **Table 1**（ACC/Forgetting，3-fold）數字已填（見附錄 B）。
- ✅ **Fig 1**（架構，`Fig1_arch`）、**Fig 2**（近期 GO）、**Fig 3**（舊任務崩）、
  **Fig 4/P0b**（recency flip，6/6，核心）、**Fig 5**（機制對照）、**Fig 6/R-matrix**（F=0 恆等）、
  **Fig S1**（評估流程，`FigS1_arch`，附錄）全部生成於 `outputs/figs/`。
- ✅ **Table 2 的 none 欄**（recent GO / old NO-GO）已填。

### Plan B 狀態
- ✅ **reverse f1 已完成並判讀**：pertask 上界恢復（0.33→0.867，GO）、ewc 不足（0.40，NO-GO）。
  Table 2 esca-OLD、§4.5、Abstract 已填 f1 數字。
- 🟡 **reverse f2-3 跑中**（tmux `planb`）→ 回來把 esca-OLD 的 pertask/ewc 改成 3-fold 平均。
- ⬜ **可選**：paper-order Plan B（補 Table 2 的 lung-OLD 列，證明跨 order 一致）。
- ⚠️ Plan B **不影響**：Table 1、Fig 1–6、§0–§4.4、storyline。所以這些**現在就能定稿**。

### Plan B 結果（reverse f1，已確認）
- **pertask（上界）= 0.867**：esca-OLD @64 由 0.33 回到 **0.867**，全 budget 持平、贏 best heur → **GO**。
  ✅ 證明「訊號還在、是遺忘」。
- **ewc（replay-free 修法）= 0.400**：仍 < best heur(0.82) → **NO-GO**。
  → 採誠實版敘事：「weight-level 正則不足以救 selection forgetting」＝ honest negative + future work（§5 已預留）。
- **待補**：reverse f2-3（3-fold 平均）；可選 paper-order Plan B（補 lung-OLD 列，證明跨 order 一致）。

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
- （Plan B）per-task router 上界**完全恢復**舊任務選擇（崩潰 0.33→0.87，GO），證明訊號仍在、是遺忘；
  但 replay-free 的 EWC-on-router **不足以修復**（0.40，NO-GO）→ selection forgetting 需 selection-aware 方法。
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
- Backbone：CONCH（frozen）+ QPMIL prompts；class-IL，backbone 對預測路徑 decoupled（→ Forgetting=0 是恆等，誠實說明，Fig 6）。
- Router：MicroRouterV0（132K，patch→scalar score→Top-K），continual 訓練、backbone 凍結。
- 選擇基線：random / prototype / semantic。
- 評估：4 TCGA 任務（lung/brca/rcc/esca），2 orders（paper/reverse），3 folds；budget = All/256/128/64/32；指標 ACC + GO/NO-GO 判準。
- （Plan B）RouterEWC（diagonal Fisher）+ per-task router（上界）。

## 4. Experiments / Results
- **4.1 近期任務：router 有效（Fig 2）** — paper-esca、reverse-lung 6/6 GO。
- **4.2 舊任務：selection forgetting（Fig 3/P0_oldtask）** — 6/6 NO-GO，router@64 < random。
- **4.3 Same-task recency flip（Fig 4/P0b，核心）** — lung 0.89→0.40、esca 0.93→0.33；lung 樣本多亦崩 → 排除混淆。
- **4.4 機制（Fig 5/P2-contrast）** — router 對舊任務「自信選錯」（押到被後續任務調教的 patch 群）。
- **4.5 Mitigation（Table 2，Plan B）** — per-task router **完全恢復**（esca-OLD @64: 0.33→**0.87**，全 budget 持平、GO）→ 證明是遺忘、訊號仍在；**EWC 不足**（@64=0.40，仍 NO-GO）→ weight-level 正則救不動 selection forgetting，需 selection-aware 方法（future work）。[esca-OLD f1；f2-3 補 3-fold]

### 圖表清單（圖號＝出現順序；Fig 1 為架構）
| 元件 | 檔案 | 說明 |
|---|---|---|
| Fig 1 | `outputs/figs/Fig1_arch.png` | **架構圖（正文 §3）**：decoupled 凍結預測 vs 可訓練 router（`tools/draw_arch.py`）|
| Fig 2 | `outputs/figs/P0_router_v0.png` | 近期任務 budget 曲線，router>heuristics（§4.1）|
| Fig 3 | `outputs/figs/P0_oldtask_budget.png` | 舊任務 budget 曲線，router 崩（§4.2）|
| Fig 4 | `outputs/figs/P0b_recency_flip.png` | **核心**：same-task recency flip（§4.3）|
| Fig 5 | `outputs/figs/P2contrast_esca_fold1.png` | 機制：recent vs forgotten router 同一切片對照（§4.4）|
| Fig 6 | `outputs/figs/P1_r_matrix.png` | backbone decouple → F=0 恆等（誠實說明）|
| **Fig S1（附錄）** | `outputs/figs/FigS1_arch.png` | **評估流程圖**：router vs baselines 共用同一凍結 backbone → ACC@K → GO/NO-GO（§3.5/§3.7）|
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

### Fig 1 — `Fig1_arch.png`（架構圖；caption 待版面定案後寫）
**Caption**：[TODO] NaviPath decoupled design — a frozen CONCH+QPMIL prediction
path (Forgetting=0 by construction) and a lightweight, continually-trained patch
router whose selection ability is what we study. Plan B (RouterEWC / per-task)
consolidates the router.
**備註**：向量檔 `Fig1_arch.pdf`。評估流程圖見 **Fig S1**（`FigS1_arch.pdf`，附錄）。
改圖編 `tools/draw_arch.py` 重跑。

### Fig 2 — `P0_router_v0.png`（近期任務 budget 曲線）
**Caption**：Patch-budget accuracy on the **most-recently-learned** task. The learned router
(red) matches/exceeds random, prototype, semantic across budgets; at K=64 it gives **+2.5–6.7 pp**
over the best heuristic (3/3 folds, both orders). Using only 64 of thousands of patches, the
router nearly matches all-patch accuracy.
**Defend**：
- Q：router 只是跟 random 打平？→ 在 tight budget（64/32）router 明顯在上，random 掉下去；@All 大家一樣（無選擇）。
- Q：cherry-pick？→ paper-esca、reverse-lung，各 3 folds，**6/6 GO**。

### Fig 3 — `P0_oldtask_budget.png`（舊任務 budget 曲線，router 崩）
**Caption**：Patch-budget accuracy on an **old** task (learned first, then overwritten
by later tasks). Unlike the recent case, the learned router falls **below random** at
tight budgets (router@64 $<$ random@64; 3/3 folds, both orders), i.e. selection
forgetting. Heuristics (prototype/semantic), being training-free, are unaffected.
**Defend**：
- Q：是不是 budget 太小才崩？→ 連 @256/@128 router 也輸 heuristics，全 budget NO-GO。
- Q：跟近期比？→ 同一 router、同評估流程，差別只在該任務在序列中的新舊（見 Fig 4）。

### Fig 4 — `P0b_recency_flip.png`（**核心**：same-task recency flip）
**Caption**：**Same-task recency flip.** For the *identical* task and test set, the router selects
well when the task was learned **last** (solid, ~0.9) but **collapses below the random baseline**
when the *same* task was learned **first** and then overwritten by later tasks (dashed, 0.33–0.40).
Holds for both lung and esca.
**Defend**：
- Q：esca 是不是太難/太小才崩？→ **lung（test ~95 張，樣本多）同樣翻轉**，排除樣本數/難度。
- Q：是不是 backbone 壞了？→ @All 四法相同（只有「選擇」這一步不同），且 R-matrix（Fig 6）顯示 backbone 不變。
- Q：只差 recency 怎麼確定是因果？→ 同一任務、同資料、同 router，**只改它在序列中的新舊**，結果翻轉 = 乾淨單變因。

### Fig 5 — `P2contrast_esca_fold1.png`（機制：recent vs forgotten router，同一切片）
**Caption**：The *same* esca slide in CONCH feature space (shared t-SNE), scored by the router
when esca was **recent** (paper, left) vs when esca was **old/overwritten** (reverse, right).
Both routers produce structured scores, but they prioritise **different patch sub-populations**:
the forgotten router's top-64 shifts to a region made salient by *later* tasks, which is
sub-optimal for esca. This mis-prioritisation (not noise) explains why router@64 falls **below**
random on old tasks.
**Defend**：
- Q：舊 router 的分數不是退化成雜訊嗎？→ 不是。圖顯示它仍有結構，但**把重要性押在錯的 patch 群**（被後續任務調教過的）→ 這正是「router@64 < random」（自信選錯）而非「= random」（無訊號）的原因。
- Q：只看一張切片代表性？→ 作為 illustrative 機制圖（如 QPMIL 亦用示例切片）；量化結論由 Table 2 / Fig 4（6/6 fold）承擔。

### Fig 6 — `P1_r_matrix.png`（backbone decouple → F=0 恆等）
**Caption**：Per-task accuracy R[i,j] (acc on task j after learning task i). NaviPath columns are
**flat by construction** (decoupled frozen backbone → Forgetting=0 is an identity, *not* a
contribution); the QPMIL baseline shows mild genuine drift. We surface this to transparently
locate our contribution in the **selection** analysis, not in a trivial F=0.
**Defend**：主動回答「為何剛好 0」，把它定位成恆等式，避免被當賣點質疑。

### Fig S1 — `FigS1_arch.png`（附錄：評估流程圖）
**Caption**：Evaluation protocol. The learned router and three training-free
selectors (random / prototype / semantic) each pick a Top-$K$ subset from the
*same* frozen patch features and feed the *same* frozen QPMIL head; we report
accuracy at budget $K$ and a GO/NO-GO criterion (router beats the best heuristic
at a finite budget). Since only the selected patch set differs, any accuracy gap
is attributable to selection alone.
**用途**：放附錄或 §3.7，正文一句「評估流程見 Fig S1」即可。與 Fig 1 互補
（Fig 1 講「系統怎麼建」，Fig S1 講「我們怎麼公平比」）。

---

# 附錄 B：Tables

## Table 1 — Continual accuracy (backbone level, mean±std, 3 folds)
| Method | Order | ACC | Forgetting | BWT |
|---|---|---|---|---|
| QPMIL baseline | paper | 0.924±0.016 | 0.017±0.022 | −0.017±0.022 |
| QPMIL baseline | reverse | 0.917±0.026 | 0.041±0.023 | −0.041±0.023 |
| NaviPath (decoupled) | paper | 0.879±0.030 | 0.000 | 0.000 |
| NaviPath (decoupled) | reverse | 0.886±0.030 | 0.000 | 0.000 |
（註1：**公平比較**——QPMIL baseline 與 NaviPath 皆 12 epochs/task、Adam lr=1e-3、wd=5e-4
（`main.yaml`；6 份 qpmil JSON 均 `epochs_override:0`）。舊「0.594/5ep」已淘汰。
註2：NaviPath F=0 為 decouple 恆等，見 Fig 6；ACC 為 router-free 全 patch。
註3：QPMIL ACC ≥ NaviPath → 本文貢獻不在 ACC，而在 selection-forgetting 分析。）

## Table 2 — Router patch selection: recent vs old (router@K, mean over 3 folds)
> 「best heur」= max(random, prototype, semantic)。GO = router > best heur（finite budget）。
> Plan B 兩欄：pertask=上界、ewc=replay-free 修法。**目前 pertask/ewc 為 reverse f1；
> f2-3 跑完改 3-fold 平均（esca-OLD 列）。lung-OLD 的 Plan B 需 paper-order，尚未跑。**

| Task | Condition (recency) | @256 | @128 | @64 | @32 | best heur@64 | GO | **pertask@64** | **ewc@64** |
|---|---|---|---|---|---|---|---|---|---|
| esca | RECENT (paper, last) | 0.956 | 0.956 | **0.956** | 0.933 | 0.889 | ✓ | — | — |
| esca | OLD (reverse, first) | 0.511 | 0.400 | **0.333** | 0.333 | 0.822 | ✗ | **0.867** (f1) | 0.400 (f1) |
| lung | RECENT (reverse, last) | 0.904 | 0.915 | **0.922** | 0.918 | 0.897 | ✓ | — | — |
| lung | OLD (paper, first) | 0.512 | 0.453 | **0.397** | 0.353 | 0.813 | ✗ | _[待跑 paper]_ | _[待跑 paper]_ |

**讀法**：同任務 RECENT→OLD，router@64 由 0.92–0.96 崩到 0.33–0.40（且 < best heur）。
**Plan B 結果（esca-OLD, f1）**：
- **pertask 上界 = 0.867，且 @256/128/64/32 全持平 0.867（＝All-patch 水準）、贏 best heur(0.82) → GO**。
  證明訊號仍在、是「遺忘」而非能力不足。
- **ewc = 0.400，仍 < best heur(0.82) → NO-GO**。weight-level 正則不足以修 selection forgetting。

## Table 2-supp — full budget curves（給 appendix，數字見 `collect_results.py` 輸出）
- recent：paper-esca、reverse-lung（GO 3/3）。
- old：reverse-esca、paper-lung（NO-GO 3/3，router < random 全 budget）。
