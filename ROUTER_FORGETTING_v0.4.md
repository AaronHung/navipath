# Router Forgetting — 發現紀錄 + Plan B 設計 + 實驗/寫稿計畫 (v0.4)

> 本檔是「過程紀錄」與「補救設計草稿」，**目前不含任何 code 變更**。
> 目的：日後補救時知道怎麼做、補救完知道怎麼寫稿、怎麼引用實驗與 ablation。
> 相關：[SOP_v0.4.md](SOP_v0.4.md)、[outputs/PROGRESS.md](outputs/PROGRESS.md)。

---

## 0. Decision log（時間序，最重要先看）

- **2026-06-23 00:0x** — TASK 2 reverse f1 跑出 oldtask_budget（esca）= **NO-GO 且崩潰**：
  router@64 = **0.133**，遠輸 random 0.80。判定為 **router 自身的 catastrophic forgetting**（非 bug，見 §2）。
- **決議**：(a) 不中斷，讓 reverse f2/f3 跑完確認複現；(b) 先寫本設計草稿備用；
  (c) 敘事在三 fold 確認後二選一（§3）：**A 誠實分析稿（預設）** / **B 加 router consolidation 救**。
- **待補資料**：reverse f2/f3 esca oldtask（跑中）；paper order 的 oldtask（最舊=lung，尚未跑，見 §5）。

---

## 1. 發現（證據）

### 1.1 控制對照（殺手級）：同一個 esca test set，只差「剛學完 vs 很久前學」

| esca 的 **router@K** | All | 256 | 128 | **64** | 32 | 判定 |
|---|---|---|---|---|---|---|
| esca **剛學完**（paper order，esca 是第 4 個）| 0.867 | 0.933 | 0.933 | **0.933** | 0.933 | GO ✓ |
| esca **很久前學**（reverse，esca 第 1 個，之後又學 3 個）| 0.867 | 0.400 | 0.400 | **0.133** | 0.200 | NO-GO ✗ |

> 同一資料、同一 router 架構，**唯一變因 = recency**。router@64 由 0.93 → 0.13。

### 1.2 reverse f1 完整數字

**eval = tcga_esca（task_index=0，最舊）— oldtask_budget_reverse_f1_task0.json**

| method | All | 256 | 128 | 64 | 32 |
|---|---|---|---|---|---|
| router | 0.867 | 0.400 | 0.400 | **0.133** | 0.200 |
| random | 0.867 | 0.867 | 0.600 | **0.800** | 0.733 |
| prototype | 0.867 | 0.733 | 0.733 | 0.733 | 0.667 |
| semantic | 0.867 | 0.600 | 0.667 | 0.667 | 0.600 |

`router@64 - random@64 = -0.667`；`router@128 - semantic@128 = -0.267` → **NO-GO**。

**eval = tcga_lung（task_index=3，最近）— router_v0_reverse_fold1.json**

| method | All | 256 | 128 | 64 | 32 |
|---|---|---|---|---|---|
| router | 0.853 | 0.895 | 0.884 | **0.895** | 0.884 |
| random | 0.853 | 0.884 | 0.853 | 0.853 | 0.779 |
| semantic | 0.853 | 0.874 | 0.863 | 0.863 | 0.853 |

`router@64 - random@64 = +0.042`；`router@128 - semantic@128 = +0.021` → **GO ✓**。

### 1.3 fold 狀態

- reverse esca oldtask：f1 ✗（崩）；**f2 / f3 跑中（待確認複現）**。
- recent-task GO 已複現：paper f1/2/3 + reverse f1（last task）。

---

## 2. 診斷：為什麼這是「真現象」而非 bug

1. **All budget 四法皆 0.8667** → 不做選擇時 backbone 對 esca 完全正常；崩的只有「router 的選擇」這一步。資料 / eval pipeline 沒問題。
2. **同條件下 heuristics 正常**（random@64=0.80）→ 同一支 eval、同一 backbone predict；只有 router 崩 = **router-specific**。
3. **訓練 log 自證遺忘軌跡**：esca（task_pos 0）loss 一路降到 0.05（router 本來學得好）；之後連續訓練 rcc/brca/lung，router 是**單一共用 MLP**（`Linear(516→256)→GELU→Linear(256→1)`，~132K 參數），打分被改成適配後面任務 → 回看 esca 挑錯。
4. **router@64 < random** 而非 ≈ random → 不是「沒訊號」，是**主動挑到最差 patch**（打分對舊任務已反向/退化）。

> 機制假說（待 §6 figure 驗證）：router 對 esca 的 score 分布在學完後變得退化（近常數）或與「判別性」反相關。

---

## 3. 敘事二選一（三 fold 確認後定案）

- **A. 誠實分析稿（預設、穩、趕得上 7/1）**
  主張：router 對**近期任務**選 patch 有效（GO，已複現）；但對**久遠舊任務**會 catastrophic forgetting，選擇能力崩潰（@64 掉 ~0.8）。
  與既有兩發現連成主線：①trainable MoE 動特徵→干擾(0.73)；②decouple 避開干擾但 expert 對主指標無貢獻；③**連最輕量的 router，patch 選擇能力本身也會被遺忘**。

- **B. A + 提出輕量修法（時間夠才做，升級成「發現問題＋解法」）**
  若 Plan B（§4）能把 esca@64 從 0.13 拉回 ≳ random/heuristic，則從 negative finding 升級為完整方法貢獻。

---

## 4. Plan B 設計草稿（router consolidation）— **設計，尚未實作**

目標：在**不碰 backbone、維持 replay-free 為主**的前提下，讓單一 router 不要忘掉舊任務的 patch 打分。

### B2（**主推**）EWC-on-router（replay-free，正規化）
- 每學完一個任務 t：用該任務 train loader 估 router 參數的 diagonal Fisher `F_t`，記錄當前最優解 `θ_t*`。
- 學任務 t+1 時，loss 加 `λ/2 · Σ F_t · (θ - θ_t*)²`，保護舊任務重要權重。
- **可直接複用** `train_cl_baselines.py` 的 `EWC` 類別模式（已有 Fisher/penalty 邏輯），只是對象換成 `router.parameters()`（132K，極輕）。
- 超參：`λ ∈ {1e2, 1e3, 5e3}` 掃描。

### B1（**上界 / oracle**）Per-task router head
- 每個任務存一份 router state_dict（132K×4，可忽略）；eval task t 時載 head t。
- 意義：**證明「選擇訊號存在、forgetting 才是元凶」**——若 per-task router 在舊任務恢復 GO，等於把 §1.1 的對照變成正式 ablation 上界。
- 限制：class-IL 下 test 需知道 task id；本實驗 budget eval 本來就 per-task（已知 task_index），故可行。當作 upper bound 報，不宣稱可部署。

### B3（便宜 ablation）Partial-freeze + per-task tiny head
- 學完任務 1 後凍結 `mlp[0]`（大 Linear，共享表徵），只為每任務學一個小的最後層 / adapter。
- 介於 naive 與 per-task 之間，成本低。

### B5（備選）Momentum / EMA consolidation
- 對 router 權重做 EMA（慢速更新），減緩覆寫。最簡單但效果通常最弱，列為對照。

### 預計的 code 介面（**尚未動**，實作時才加）
- `train_router_v0.py` 新增：`--router-consol {none,ewc,perTask,freeze,ema}`、`--consol-lam <float>`。
- 產出檔名建議加後綴以免覆蓋：`router_v0_{order}_fold{f}__{consol}.json`、`oldtask_budget_{order}_f{f}_task{t}__{consol}.json`。
- 防覆蓋機制沿用現狀（存在即 skip）。
- `none` = 現行行為（baseline，已跑）。

---

## 5. 實驗矩陣（補救時要跑的）

> 先把 **paper order 的 oldtask 也補上**，讓「recency 對照」對稱（paper 最舊=lung；reverse 最舊=esca）。
> 補 paper oldtask（不需新 code，現有 `--eval-tasks` 即可）：
> `--eval-tasks="0"`（paper task0 = lung）→ `oldtask_budget_paper_f{1,2,3}_task0.json`。

| 變因 | 取值 |
|---|---|
| consol | none(已跑) / ewc(λ掃) / perTask / freeze / (ema) |
| order | paper, reverse |
| fold | 1,2,3 |
| eval | recent(last task) + old(task0) |
| budget | All,256,128,64,32 |

**GO/NO-GO 判準（沿用現行）**：`router@64 - random@64 > 0.02` 或 `router@128 ≳ semantic@128`。
**補救成功判準**：在**舊任務**上，ewc/perTask 的 `router@64` 由 ~0.13 回到 **≥ random@64**（理想 ≥ heuristics 最佳）。

---

## 6. 寫稿 / 引用實驗對照（paper mapping）

| 論文元件 | 來源實驗 / 檔案 | 支撐的 claim |
|---|---|---|
| **Fig 1**（主結果）patch-budget on recent task | `router_v0_*` + `plot_results.py P0` | router 對近期任務選得比 heuristics 好 |
| **Fig 2**（殺手圖）同任務 recent vs old | §1.1 對照（paper-last esca vs reverse-first esca；對稱補 lung） | router 的選擇能力 recency-dependent、會遺忘 |
| **Fig 3**（機制）router score 分布 old vs recent | `plot_results.py P2-lite`（esca slide，比較剛學完 vs 學完後） | 解釋為何挑到最差 patch（score 退化/反向） |
| **Fig 4**（R-matrix）backbone decouple 恆等 | `plot_results.py P1` | 誠實說明 F=0 是 decouple 結果，非主貢獻 |
| **Table 1** ACC/Forgetting（backbone） | `collect_results.py` accuracy 表 | 持續學習主指標 |
| **Table 2** router budget GO/NO-GO × {none,ewc,perTask} × {recent,old} | 補救實驗（§5） | 若走 B：forgetting 可被簡單 consolidation 緩解 |

**寫法骨架**：
1. 觀察：router 對近期任務有效（Fig 1）。
2. 反例：對舊任務崩潰（Fig 2），且非 backbone 問題（All=heuristics、R-matrix Fig 4）。
3. 機制：score 退化（Fig 3）。
4.（若走 B）緩解：EWC-on-router 部分恢復；per-task router 為上界（Table 2）。
5. 結論：在 frozen-FM 上加 trainable 選擇/路由，需正視「選擇本身也會被遺忘」。

---

## 7. 風險 / 待確認

- **R-1 複現性**：esca 崩是否在 reverse f2/f3 一致？（跑中）若 f1 為離群，重評。
- **R-2 對稱性**：paper 最舊（lung）是否也崩？需補 paper oldtask（§5）。若 lung 不崩、esca 才崩，需區分「recency」與「任務難度/樣本數」混淆因子（esca test 僅 15 張，樣本小、方差大——務必三 fold 平均並標 error bar）。
- **R-3 小樣本**：esca test=15 張，單 fold 數字雜訊大；所有結論以三 fold mean±std 為準。
- **R-4 B 不保證成功**：EWC 可能只部分恢復；per-task 一定成功但只是上界。時間不足就走敘事 A。

---

## 8. 立即 TODO（待 f2/f3 push、Mac pull 後）

1. 讀 `outputs/oldtask_budget_reverse_f{1,2,3}_task0.json` → 三 fold 一致性（R-1）。
2. `collect_results.py` 重跑 → oldtask 區塊填上；`plot_results.py` 出 P0(old) + P2-lite。
3. 補 paper oldtask（§5 指令）→ 對稱對照（R-2）。
4. 依三 fold 結果定案敘事 A / B；若 B，按 §4 實作（届時才動 code）。
