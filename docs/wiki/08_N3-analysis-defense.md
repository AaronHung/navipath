# 08 · N3 分析的答辯觀念（append-only）

> N3 = Mac 端把 N2 跑出的 json 收成「完整 4 任務 retention 表 + 結果圖」。
> 這篇收錄「報 N3 時、被問必須答得出來」的觀念。對照：`outputs/seqobs_*.json`、SOP `specs/01_sop_*` §N3。
> 規則：只增不刪。

---

## 0. N3 在做什麼（一句話）
N2 把 4 個任務的 router 都訓練好、存進 NSM 了；**N3 不重訓**，只是把同一個 skill bank 對 `task_index = 0/1/2/3` 分別 eval（esca/rcc/brca/lung），組出 **retention 表**＋畫圖。重點從「單看 esca」升級成「看整條任務流的記憶全貌」。

---

## 1. 必須懂的 CL 三指標
設 `a[i][j]` = 「學完第 i 個任務後、測第 j 個任務」的準確率（i ≥ j 才有意義）。

- **mACC（平均準確率）**：全部學完後，4 個任務 acc 的平均 = `mean_j a[T][j]`。整體好不好。
- **Forgetting（遺忘量，越小越好）**：對每個舊任務，「歷史最佳 acc − 最終 acc」，再平均。量「被後面任務洗掉多少」。
- **BWT（Backward Transfer，越接近 0 / 正越好）**：`mean_{j<T}(a[T][j] − a[j][j])`。學新任務對舊任務是傷害（負）還是幫助（正）。

> 防答辯：nonsm 會出現**大 Forgetting / 很負的 BWT**（如 esca 0.867→0.133）；nsm 把 Forgetting 壓近 0。**對比這兩條就是 N3 的主結果。**

---

## 2. retention 表怎麼讀（核心圖）
列 = 「學到第幾個任務」，欄 = 「測哪個任務」，看**下三角**：

| 學完↓ \ 測→ | esca | rcc | brca | lung |
|---|---|---|---|---|
| esca | a00 | – | – | – |
| rcc | a10 | a11 | – | – |
| brca | a20 | a21 | a22 | – |
| lung | a30 | a31 | a32 | a33 |

- **對角線 a_ii**：剛學完當下的表現（能學會嗎）。
- **同一欄往下掉**（如 esca 欄 a00→a30 變差）= 遺忘。
- **nsm vs nonsm 兩張表並排**：nonsm 下三角崩、nsm 守住 → 一眼看出貢獻。

---

## 3. budget 曲線（Agent 軸的圖）
x = budget(16/32/64/128/All)，y = acc。一張片只准看 K 個 patch 時 acc 怎麼變。
- 看 **acc@K 多快逼近 acc@All**：逼得越快＝「少少幾個 patch 就抓到重點」＝ navigation 有效。
- 同圖疊 **sequential vs one-shot**：兩線分不開（目前狀況）就誠實標「sequential 增益待調參」（見 wiki 07 §7）。

---

## 4. 為何要 3-fold（mean ± std）
esca test 只有 15 張，**單 fold 顆粒度 6.67%（差 1 張就跳 0.067）**，極易被切分運氣影響。跑 3 個 fold 取平均±標準差，數字才有統計意義、reviewer 才不會打「n=1」。**缺的 fold/格子標 `[MISSING]`，不捏造。**

---

## 5. N3 的成功判準（先講好，免得事後挪動門檻）
- **主判準（CL）**：nsm 的舊任務 acc **顯著 > nonsm**，且 Forgetting 明顯較小。→ 已有 fold-1 強訊號。
- **次判準（Agent budget）**：acc@64 接近 acc@All。
- **探索（不設門檻）**：sequential 是否 > one-shot（這版多半打平，列 future tuning）。

## 更新日誌
- 2026-06-27（深夜）：建立本篇（N3 分析答辯觀念）。
