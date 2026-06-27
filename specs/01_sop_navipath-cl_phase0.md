# SOP · NaviPath-CL Phase-0（6/27 → 7/20）

> 取代 `legacy/SOP_v0.4.md`（舊 selector 進度，已不報告）。
> 老師 bottom line：**在現有 WSI navigation agent 之上，擴增 CL 能力**。7/3 報「agent + CL 的預計架構（含架構圖）」+「到 7/2 跑出的實驗結果」。
> Authoritative 架構：`specs/decisions/ADR-0006-*.md`；定義/歸因：`docs/wiki/`；敘事：`STORYLINE.md`；看板：`site/`。

## 鐵則
- **Mac 改 code + smoke；RunPod 跑實驗。** 結果存 `outputs/`（進 git），檔名照命名規則、**不覆蓋舊檔**。
- 同步：RunPod `git add outputs/ && commit && push` → Mac `pull` → check。
- 每個里程碑結束更新 `specs/worklog/WORKLOG.md` 與 `outputs/PROGRESS.md`。
- 歸因鐵則：**全程凍結同一個 QPMIL backbone**，只動 navigation 層（見 wiki 04）。

---

## 里程碑

### N0 — 架構鎖定（6/27）✅
- 通用架構圖（`site/figs/arch_navipath_cl.svg`）+ 細部圖（§STORYLINE 6）。
- wiki 04（通用化 + 歸因）；舊 v0.4 檔歸檔 `legacy/`；README 重寫。
- **交付**：`site/architecture.html`、`docs/wiki/04`、本 SOP。

### N1 — 實作 Sequential Budgeted Observation（6/28–6/29）★ 新肉
- Mac：把單步 selector 升級為**多步、累積證據**的觀察迴圈。
  - `ObservationState`：已看 patch 聚合特徵 + running proto/text 相似度 + backbone 信心 + coverage。
  - `NavigationPolicy`：依 state 評分 → 每步取 Top-k；跑 R 輪累積到 budget K。
  - frozen QPMIL backbone；oracle `ContextGate`；per-task `NSM`（沿用 `continual_agent.py` 既有零件）。
  - 輸出 `NavigationTrace`（觀察順序）。
- **交付**：`navipath_moe/sequential_observation.py` + smoke test 過（Mac CPU、小樣本）。
- **驗收**：smoke 能 navigate→predict 並印出 trace；不碰 canonical outputs。

### N2 — RunPod 第一批實驗（6/29–6/30）
- 在 frozen QPMIL backbone 上跑：
  - **Budget 效率**：`acc@K`（K=16/32/64/128）vs `acc@All`。
  - **CL 保留**：old-task `acc@K`，有 NSM vs 無 NSM（naive 連續訓練）。
  - **Agentic**：sequential vs one-shot Top-K（同 budget）。
  - **zero-shot navigator baseline**（SPEC-07，`--policy-mode zero_shot`）：不訓練、frozen-FM 文字相似度選 patch；**不需 epochs/skill-bank、很快**，四個任務一次跑完。用來回應 ZeroSlide / 老師 challenge。
- **交付**：`outputs/seqobs_<order>_fold<f>_*.json` + `.log`（router 與 `*_policy-zeroshot.json` 並存，不覆蓋）；push。
- **驗收**：每格 json 非空、含 `task_index` / `budget` / `mode` 欄位。

### N3 — Mac 分析 + 結果圖（7/1）
- pull → 用實際 json 產 `outputs/RESULTS_seqobs_<ts>.md` 與結果圖（budget 曲線、有/無 NSM 對比）。
- **continual vs zero-shot navigator 對比表**：把 router-mode 與 `*_policy-zeroshot.json` 並列，回答「zero-shot navigation 夠不夠？」（wiki 09 G 節）。zero-shot 同時當 retention 表的「零遺忘參考線」。
- **交付**：結果表 + `site/figs/` 更新；缺格標 `[MISSING]`，不捏造。

### N4 — 凍結 pilot + 雙月報告稿（7/2）
- 凍結 7/2 為止的數字；寫 `reports/bimonthly_2026-07-03.md`（架構 + pilot 結果 + 後續計畫）。
- **交付**：報告稿 + 看板 board.html 更新。

### N5 — 7/3 報告
- 報告內容：通用架構（agent + CL）+ 到 7/2 的實驗結果 + Phase-1/2 heads-up。**不報舊 selector，不假裝論文已完成。**

### N6 — 論文初稿（7/3–7/14）
- 把 pilot 做成論文：problem framing（不以 QPMIL 開頭）、method（CNL/NSM/序列觀察）、experiments（budget 效率 + CL 保留 + EWC baseline + sequential vs one-shot ablation + **zero-shot vs continual navigator** 對比，呼應 ZeroSlide）。
- **交付**：`paper/NaviPath-CL_draft_outline.md` → 正文初稿。

### N7 — 老師 review（7/15）
- 交初稿給老師；收回饋。

### N8 — 年度報告（7/16–7/20）
- 依回饋修論文；寫年度報告 `reports/annual_2026-07-20_outline.md` → 定稿。

---

## 你（Aaron）要做的
- N2 / N3：在 RunPod 貼我給的指令、跑完 push；Mac 端我來 pull + check。
- 每個里程碑我會把結果與圖更新到 `site/`，你看板上即可檢視、回饋。
- 圖/敘事要改的，隨時喊停（keep in loop）。

## 模型建議
- 改 code（N1）/ 寫分析（N3）/ 論文（N6）：高階推理模型。
- pull / 跑指令 / commit（N2、N8 操作）：中階即可，省 token。
