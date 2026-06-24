# 公平性 & Sanity-check 說明（ConSlide → QPMIL pivot）

> 寫給「我自己」看懂、然後能跟老師講清楚、答得出問題的版本。不是正式論文語氣。
> 一句話：**我們不是為了逃避比較才換 backbone；我們是換到一個「每個條件都能被我們控制、且可重現」的設定，讓比較第一次真正公平。**

---

## 0. 先講結論（30 秒版）
- 老師質疑的是：**拿 ConSlide 來比不公平**——因為我們**沒辦法完整複製它的執行條件、也缺它要的資料**，那任何「我們重跑出來的 ConSlide 數字」都是我們的瑕疵版，不能拿來當對照。
- 我們的回應：**pivot 到 QPMIL-VL**。理由不是「ConSlide 太強躲開它」，而是 QPMIL 讓我們能做**受控對照（controlled comparison）**：同一個凍結 backbone、同一份特徵、同一份資料切分、**同樣的訓練預算（每任務 12 epochs、同 lr/wd）**、同一套評估。差異只剩我們真正想研究的那一個變因（patch 選擇）。
- 而且我們**根本不靠「贏 baseline 的準確率」當賣點**（QPMIL 的 ACC 其實 ≥ 我們）。所以「你是不是把 baseline 弄弱才贏」這種攻擊在我們身上不成立。

---

## 1. 老師到底在質疑什麼（把問題講準）
公平性問題的本質是 **「對照組是否在相同條件下產生」**。
- 如果對照組（ConSlide）的數字是我們**自己重跑**的，但我們**無法重現它原本的設定**（架構細節、資料、預處理、訓練條件），那這個數字到底代表「ConSlide 的真實能力」還是「我們沒調好的 ConSlide」？**沒人說得清** → 比較不公平、reviewer 一問就垮。
- 老師的話翻成白話：「你連 ConSlide 都複製不出來、資料也不齊，那你跟它比的數字我憑什麼信？」

這是對的質疑。**所以解法不是硬湊一個 ConSlide 數字，而是換到一個我們能 100% 控制的對照基準。**

---

## 2. 為什麼「比 ConSlide」本來就不公平 / 不可行（三個硬理由）
（來源：我們自己的 `Navipath_moe_plan_v01.md` §1.4 pivot 紀錄）

1. **複現困難、且無法驗證**：ConSlide 是階層式架構（HIT），要拿到 region-level attention 得做 attention rollout。我們很難完整復現，而且 reviewer 會直接問「你的 region-level 是真的嗎？」——**一個我們自己都無法保證正確的重跑，不能當公平對照**。
2. **問題設定不同（buffer-based vs replay-free）**：ConSlide **要儲存舊資料（buffer）**；我們的賣點是 **完全不存舊資料（replay-free）**。兩者根本是**不同規則的比賽**，硬比是 apples-to-oranges。
3. **缺資料 / 缺對齊的執行條件**：ConSlide 需要的資料與預處理我們沒有齊備；而 QPMIL **官方有公開 code + 已備好的 CONCH features**（我們手上的 `can_dataset`），可以照官方設定跑通。

> 結論：與其報一個「我們自己拼湊、無法驗證」的 ConSlide 數字（不公平），不如**把 ConSlide 保留為 cited baseline（引用、說明差異），主線換成可受控的 QPMIL**。這是**提高**嚴謹度，不是規避。

---

## 3. 為什麼 pivot 到 QPMIL 之後，比較就「公平」了
公平 = **受控變因實驗（controlled experiment）**：把所有條件鎖死，只動我們要研究的那一個。
我們的論文做的就是這件事——**唯一變動的是「誰來選 patch」**（learned router vs random/prototype/semantic），其餘全部相同：

| 維度 | baseline 與 ours 是否相同？ | 為何這保證公平 |
|---|---|---|
| 影像編碼器 | 同：凍結 CONCH，特徵抽一次共用 | 沒有人偷換更強的 encoder |
| 預測頭（backbone）| 同：QPMIL prompt-MIL head，**凍結** | 所有 selector 把選到的 patch 餵**同一個** head；準確率差異只能來自「選得好不好」 |
| 資料與切分 | 同：4 個 TCGA cohort、官方 fold splits、兩種 order、3 folds | 不挑對自己有利的 split |
| 訓練預算 | 同：**每任務 12 epochs、Adam lr 1e-3、wd 5e-4** | 不會「baseline 少訓練、ours 多訓練」偷贏 |
| 評估協定 | 同：同 test set、同 budget K、同 GO/NO-GO 判準 | 對所有 selector 一視同仁 |

→ 因為**只有「選擇器」這一個變因在動**，我們觀察到的差異（router 在近期任務 GO、舊任務崩）**只能歸因於選擇本身**。這就是公平比較的定義。

---

## 4. 我實際做了哪些 Sanity-check（可現場給老師看證據）
每一條都對應 repo 裡可查的檔案，不是嘴上說說。

1. **訓練預算對等（最關鍵）**
   - 查 `QPMIL-VL/configs/main.yaml` → `epochs: [12, 12, 12, 12]`、`adam_lr: 0.001`、`adam_weight_decay: 0.0005`。
   - **同一份 config** 同時用於 QPMIL baseline 與 NaviPath backbone。
   - 為何重要：reviewer 最常見的攻擊就是「epoch 不對等」。我們已對齊 → 這刀砍不到。
2. **同特徵、同資料**
   - `main.yaml`：`dataset_names: [tcga_lung, tcga_brca, tcga_rcc, tcga_esca]`、`dataset_label_shift: [0,2,4,6]`、CONCH `feats-l1-s256` 特徵、`total_fold: 10`（我們報 3 folds 平均）。
   - baseline 與 ours 讀**同一批 `.pt` 特徵檔**。
3. **同一個凍結預測頭**
   - 論文 §3.5 / Fig. S1：router 與三個 heuristic（random/prototype/semantic）選出的 patch，全部餵**同一個 frozen backbone**。只有「選哪些 patch」不同。
4. **誠實性檢查（我們主動自首）**
   - `Forgetting = 0` 我們**明講是 decoupled frozen backbone 的結構恆等式（identity），不是成果**（論文 §4.2、§5、Fig. 6 R-matrix 攤開給看）。
   - 我們**不宣稱準確率贏 QPMIL**（QPMIL ACC 0.924/0.917 ≥ 我們 0.879/0.886）。我們的貢獻是**選擇路徑的分析**，不是刷 SOTA。
   - 為何重要：這反而是公平性的最強證據——**我們沒有任何「灌水贏 baseline」的動機**，因為我們本來就沒在比 ACC。

---

## 5. 老師可能追問 + 怎麼答（背起來）
- **Q：你換 backbone 是不是因為 ConSlide 打不贏 / 不敢比？**
  A：不是。ConSlide 是 **buffer-based、要存舊資料**，和我們 **replay-free** 是不同規則；而且它的 region-level attention 我們無法可靠複現。報一個我們自己拼的 ConSlide 數字才**不**公平。我們把它**保留為 cited baseline**（在 Related Work 說明差異），主線換成**能被我們完全控制、可重現**的 QPMIL。這是提高嚴謹度。
- **Q：那你跟 QPMIL 比，是不是把 QPMIL 設弱了？**
  A：相反。我們用**官方 config、官方特徵**，baseline 跟我們**同 12 epochs、同 lr/wd**。而且我們**不靠贏 ACC**（QPMIL ACC 還比我們高）。我們動的只有「選 patch 的人」。
- **Q：Forgetting=0 是不是灌水？**
  A：不是成果，是 identity。backbone 凍結、預測永不吃被改過的特徵，所以跨任務準確率結構上不變。我們**主動攤開**講（Fig. 6），貢獻在 selection 分析。
- **Q：esca 只有 15 張 test，結論可信嗎？**
  A：所以我們做了 **same-task recency 翻轉**，對**樣本最多的 lung**（~760/task）也一樣崩（0.92→0.40），排除「樣本少才崩」。6/6 folds×orders 複現。
- **Q：為什麼不乾脆把 ConSlide 也跑出來放表裡？**
  A：可以放，但只能當「**cited / 盡力複現**」並標明設定差異；不能拿來當我們主張的核心對照，否則就回到那個無法驗證的公平性問題。我們的核心對照是受控的 selector 比較。

---

## 6. 一頁帶走（給老師的投影片骨架）
1. **問題**：ConSlide 對照無法公平（不可重現、buffer-based、缺資料）。
2. **決策**：pivot 到 QPMIL（公開 code + 備好特徵）→ 做**受控對照**。
3. **公平性鎖死**：同 encoder / 同 head / 同資料 splits / **同 12 epochs+lr+wd** / 同評估；**只動「選 patch 的人」**。
4. **誠實**：Forgetting=0 是 identity 不是成果；不比 ACC（QPMIL 還比我們高）。
5. **可重現**：QPMIL-VL 已 vendored、config 與指令都在 `ONBOARDING_runbook.md`，任何人能重跑。
6. **ConSlide 去哪**：保留為 cited baseline（說明 replay-free vs buffer-based 的設定差異）。

> 證據檔：`QPMIL-VL/configs/main.yaml`（epochs/lr/wd）、論文 `paper/paper_body.tex` §3.5/§4.0/§4.2、`Navipath_moe_plan_v01.md` §1.4（pivot 紀錄）、`ONBOARDING_runbook.md`（可重現流程）。
