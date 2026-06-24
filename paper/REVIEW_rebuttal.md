# Self-critique & Rebuttal Ammunition (v1.0)

> 用途：投稿前自我挑刺 + meeting 時教授/team 會問的問題，先把彈藥備好。
> 每條格式：**[攻擊]** reviewer 會怎麼打 → **[彈藥]** 我們怎麼回（含已有數據/圖表）→ **[殘留風險]** 還沒補的。
> 對應主稿：`paper_body.tex`；數據：`outputs/*.json`。

---

## A. 核心威脅（reviewer 五分鐘內會問的）

### A1. 「Forgetting=0 是不是灌水？」
- **[攻擊]** 你號稱 Forgetting=0，是不是 cherry-pick 或評估假象？
- **[彈藥]** 我們**主動聲明這是 identity 不是貢獻**（Abstract、§4.2、§5「On the honesty of Forgetting=0」、Fig.~\ref{fig:rmatrix} R-matrix columns flat）。預測路徑是 frozen backbone、永不吃 router 改過的特徵（Eq.~\ref{eq:decouple}），所以跨任務 per-task accuracy 結構上不變。我們把它攤開講，正是為了不 over-claim。貢獻在 selection 分析，不在 ACC。
- **[殘留風險]** 無（已轉為敘事優勢）。

### A2. 「Baseline 不公平（epoch 不對等）？」
- **[攻擊]** NaviPath 用 12 epochs，QPMIL baseline 是不是少訓練？
- **[彈藥]** Table~\ref{tab:acc} caption 與 §4.0 明寫 **matched training（12 epochs/task, Adam lr 1e-3, wd 5e-4）**，QPMIL baseline 與我們同設定。而且 **QPMIL ACC ≥ NaviPath**（0.924/0.917 vs 0.879/0.886），我們沒有靠 ACC 賣點，所以不存在「用 epoch 灌贏 baseline」的動機。
- **[殘留風險]** QPMIL 的 12-epoch 重跑數字（0.924/0.917）來自我們的 run；若 reviewer 要原論文數字對照，附錄補一張對照即可。

### A3. 「esca test 只有 15 張，崩潰是樣本少不是遺忘？」
- **[攻擊]** 最舊任務 esca 樣本太少，0.33 可能是雜訊。
- **[彈藥]** 這正是我們設計 **same-task recency-flip**（§4.4, Fig.~\ref{fig:recency}）的原因：對**樣本最多的 lung**（~760 slides/task）做同樣翻轉，recent 0.922 → old 0.397，一樣崩。樣本少被**證偽**為主因。6/6 folds×orders 複現。
- **[殘留風險]** 無（lung 已堵住此洞）。

### A4. 「router 比 random 還差，會不會只是 bug / 沒訓好？」
- **[攻擊]** 低於 random 通常代表實作有問題。
- **[彈藥]** (1) 同一個 router 在 recent 任務上 6/6 GO，訓練是有效的；(2) per-task router 把 old 任務從 0.333 救回 **0.933（3/3 GO）**（Table~\ref{tab:planb}），證明 selection 訊號**仍在 frozen 特徵裡**、實作正確，崩潰是真遺忘而非 bug；(3) Fig.~\ref{fig:mechanism} 顯示是「自信地選錯」（confident mis-prioritization），分數仍結構化，不是亂數。
- **[殘留風險]** 無。

---

## B. 方法/實驗深度問題

### B1. 「為什麼 EWC 救不了？是不是 λ 沒掃好？」
- **[攻擊]** 也許換個 λ / Fisher 估法 EWC 就有效。
- **[彈藥]** §5 給了**機制性解釋**：真正要保護的是 patch 分數的**排序（ranking）**，那是權重的高度非線性全域函數；EWC 罰的是個別權重漂移，是 ranking 的差代理。三 folds EWC@64 都恰好 0.40（極一致），不像「沒掃到」。我們把方向指向 selection-aware（distill 舊任務排序 / function-space 正則）。
- **[殘留風險]** 只掃了 λ∈{部分值}；可補一條 λ-sweep 曲線當附錄，強化「不是調參問題」。→ 見 runbook「未來任務 F1」。

### B2. 「per-task router 是作弊（需要 task-id）？」
- **[攻擊]** 上界要 task identity，現實不可用。
- **[彈藥]** 我們**明說它是 empirical upper bound**（Table~\ref{tab:planb}、§4.6），目的是證明「資訊還在、可恢復」，不是宣稱可部署方法。它界定了問題的可解性，並反襯 EWC 的不足。
- **[殘留風險]** 無（定位清楚）。

### B3. 「只有 4 個 TCGA 任務、單一來源。」
- **[攻擊]** 任務序列太短、泛化存疑。
- **[彈藥]** §5 Limitations (i) 已承認；這是 workshop short paper 的 scope。現象在 2 orders × 3 folds × 2 焦點任務（esca/lung）都複現，已有相當穩健性。
- **[殘留風險]** 真。未來補 colon/更多 cohort 或更長序列。→ runbook「未來任務 F3」。

### B4. 「router 太簡單（scalar v0），結論能推廣到 attention/set selector 嗎？」
- **[攻擊]** 換成 attention-based selector 可能不會遺忘。
- **[彈藥]** 我們**不宣稱所有 selector 都會**；我們**首次**指出並命名這個現象、給乾淨因果測試與機制。§5 (iii) 明列為 limitation/future work。提出問題本身就是貢獻（analysis paper）。
- **[殘留風險]** 真。可當 future work 賣點。

### B5. 「MoE/experts 呢？標題曾有 MoE。」
- **[攻擊]** experts 只活 2/4、對主指標無貢獻。
- **[彈藥]** 已**降級為 ablation**，主線是 router 的 selection forgetting。decoupled 設計下 experts 不碰預測路徑（§4.2 解釋為何非 decoupled 會災難性干擾 0.735/0.950）。不在正文宣稱 MoE 收益。
- **[殘留風險]** 若投稿版本完全不提 MoE 更乾淨；目前僅在 related/ablation 出現。

---

## C. 寫作/呈現層面

- **C1 t-SNE 機制圖只取 1 個 slide/fold1** → 標為 illustrative；§4.5 結論由 budget 曲線（量化）支撐，圖只是直覺。未來可補多 slide 統計。
- **C2 GO/NO-GO 閾值（@64）是否任意** → 我們同時附**完整 budget 曲線**（@256/128/64/32），@64 只是單一摘要點；結論在所有 tight budget 一致。
- **C3 「selection forgetting」是否真新名詞** → related work §2「Gap」已界定與 classifier/representation forgetting 的差異；可在 rebuttal 補一句「若 reviewer 知道既有命名，我們樂於對齊」。

---

## D. 一句話總攻防（meeting 開場可用）

> 我們不是在賣一個贏 SOTA 的方法；我們是**第一個指出**：在 frozen-FM 持續學習中，即使分類器不可能遺忘，**「看哪裡」的選擇器會遺忘**，而且會**自信地選錯**（比亂選還差）。我們用**同任務 recency 翻轉**做了乾淨因果證明、用**最大樣本 lung** 排除了樣本數干擾、用 **per-task 上界（0.33→0.93, 3/3）** 證明可恢復、並指出 **EWC 這類權重級修法不夠**——把問題、證據、與未來方向一次講清楚。

---

## E. 還沒補但「可補強」清單（投稿前若有時間）
1. **EWC λ-sweep 曲線**（堵 B1 殘留）— 1 張附錄圖。
2. **paper-order Plan B（lung 當最舊）** 3-fold — 讓 Table 3 從「esca only」擴成兩任務，泛化更強。
3. **多 slide 機制統計**（堵 C1）。
4. QPMIL 原論文數字對照表（堵 A2 殘留）。

> 以上 1–4 皆「加分非必須」；目前數據已足以支撐全部正文 claim。
