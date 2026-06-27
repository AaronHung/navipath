# 09 · ZeroSlide 對比 與 老師疑慮的回應（append-only）

> 老師轉來 ZeroSlide（MICCAI workshop）+ 兩個 challenge（budget 必要性、selection forgetting 是否 trivial）。這篇是定稿回應。
> 規則：只增不刪。

---

## A. ZeroSlide 在做什麼？跟我們的關係
**ZeroSlide 結論**：在 WSI 的 lifelong learning，用**凍結的病理 VLM（TITAN/CONCH 類）做 zero-shot 分類、且吃「全部 patch」**，就能跟需要訓練的 CL 方法打平，還天生**零遺忘**（因為不訓練、沒參數可洗）。

**關鍵體認：它把「分類這一軸的 CL」幾乎關掉了。** 既然 frozen FM 已能 zero-shot 把分類做好且不遺忘，**再去比「分類 CL」沒有空間**。

→ 所以 ZeroSlide 不是對手，是**我們的動機錨點**：它正好把問題逼到我們這條軸——
> **「分類能 zero-shot；那『該看哪 (navigation/diagnostic search)』也能 zero-shot 嗎？還是需要 continual navigation learning？」**

這正是老師那句 *frozen FM + continual selector（where to look）* 的精確化。

---

## B. 一張對照表

| | ZeroSlide | NaviPath-CL（我們） |
|---|---|---|
| 軸 | **分類** 是否需要 CL | **navigation（看哪）** 是否需要 CL |
| patch | **全吃** | **budget 下選 K 個**（agentic 序列觀察） |
| 是否訓練 | 不訓練（zero-shot） | navigation 層 trainable + NSM 記憶 |
| 遺忘 | 天生沒有（無參數） | **navigation 會遺忘 → 我們用 NSM 解** |
| 產出 | slide label | label **＋ Navigation Trace（可解釋軌跡）** |
| 互補性 | 我們的 backbone 可直接用它那種 frozen FM | — |

**「我們 oneshot 不太好＝輸 ZeroSlide」是誤解**：ZeroSlide 比的是「全 patch 分類準度」；我們比的是「在 budget 下、跨任務還記不記得怎麼選」。**不同題目。** 我們的 nsm vs nonsm（0.867 vs 0.133）是在我們自己的題目上贏，跟 ZeroSlide 不衝突。

---

## C. 回應老師疑慮①：encode 完才選，budget 還省得到算力嗎？
**先承認**：在現行 encode-all → aggregate 流程下，encoder 已對全部 patch 跑完，後段聚合是 O(n·d) 線性，**對 predictor 加 Top-K 幾乎不省時間。以「省 predictor 算力」當主動機，站不住，我們放棄這個 framing。**

**budget 的真正角色（改寫後的主張）：**

1. **臨床可稽核性 / 可解釋（最主要）**：醫師/法規要的是「**短、有序、可驗證**的證據鏈」。budget=K 逼出「最關鍵的 K 個視野 + 觀察順序（Navigation Trace）」；全 patch attention heatmap 糊成一片，不能稽核。**這跟省不省算力無關，是『證據要可讀』。**
2. **下游每-patch 成本高時，K vs n 放大百倍**：被選中的 patch 若要再餵大型 VLM/LLM、agent 逐 patch 推理、高倍重讀、或**人工複閱**，成本就從 encoder 移到下游，budget 變關鍵。**agentic pipeline 正是這種情境。**
3. **指向更有價值的未來設計（select-before-encode）**：若用便宜訊號先決定「哪些 patch 值得送進 encoder」，省的是最貴的 encoder passes。我們現在 encode 完才選不適用，但**本工作為它鋪路**。
4. **臨床直覺**：WSI 太大，醫師看到某 pattern 就會轉向、不會逐格看。budget + sequential 正是把這種「看夠了就停、據已見決定下一步」形式化。

> 一句話版：**budget 不是為了省算力，是為了「可稽核的證據鏈」與「下游昂貴觀察」；真正省算力的版本（select-before-encode）是本工作鋪路的未來式。**

---

## D. 回應老師疑慮②：selection forgetting 是 trivial 嗎？
**承認**：任何 trainable 模組 sequentially train 都會遺忘（就像 ResNet 當 CL backbone 會忘）。**「觀察到遺忘」本身不是貢獻，這點老師對。**

**我們的貢獻不是「發現遺忘」，而是：**
1. **解法**：NSM（per-task skill memory + gate）把 navigation 遺忘救回（0.133→0.867）。
2. **問對問題**：承 ZeroSlide——分類能 zero-shot 不遺忘了，**那 navigation 呢？** 我們證明 **navigation 這個臨床關鍵能力不會自動免疫於遺忘**，需要被當成 CL 問題處理。這是把 ZeroSlide 的問題**延伸到一個它沒碰、且臨床上更要緊的維度**。
3. 因此用詞從「selection forgetting 是新發現」改為「**continual navigation：zero-shot navigation 夠不夠？**」。

---

## E. 由此新增的關鍵 baseline：zero-shot navigation
ZeroSlide 直接催生一個**必須做**的對照組：

- **zero-shot navigator**：不訓練 router，直接用 frozen backbone 的固有語義訊號（CONCH patch-text 相似度）選 K 個 patch。天生零遺忘。
- **continual navigator（我們）**：trainable router + NSM。

**研究問題定版**：*Is zero-shot navigation enough, or do we need continual navigation learning?*
- 若我們 > zero-shot navigator → 證明「learned + 記憶的 navigation」有價值。
- 若打平 → 誠實報告，仍貢獻「navigation 維度的 CL 分析 + zero-shot navigation 這個強 baseline」。

→ 這條已列為後續實驗（見下方「待補做法」）。

---

## F. 釐清「zero-shot navigator」≠ one-shot 的兄弟（用詞陷阱 + 2D 表）

老師/我們最容易混的點：zero-shot、one-shot、sequential 都有「shot」，但**講的是兩件事**。

| 詞 | 這裡的「shot」指 | 屬於哪條軸 |
|---|---|---|
| one-shot / sequential（多步） | **一張片內「看幾輪」** | Agent 軸（怎麼看） |
| **zero-shot** | **訓練用了幾個樣本**（0＝完全不訓練） | 訓練軸（policy 怎麼來） |

→ `zero-shot` 的 zero 是「零訓練」，**不是「零觀察步驟」**。zero-shot navigator 一樣會選 patch、也能 one-shot 或 sequential，只是它的**打分規則沒被訓練過**。所以是 **2D（2×2）**，不是一條線：

| | **one-shot**（一次選 K） | **sequential**（多輪選 K） |
|---|---|---|
| **zero-shot**（不訓練；分數＝CONCH 文字相似度） | 不訓練・一次選 → **新 baseline** | 不訓練・多輪選 → **新 baseline** |
| **continual**（訓練 router + NSM）＝**我們** | 訓練・一次選（≈ 舊 selector） | 訓練・多輪選 → **★ 主打** |

- **橫軸（左右）**＝一張片內看幾輪（Agent 軸）；**縱軸（上下）**＝這套選法要不要訓練（訓練軸）。
- N2 跑的是**下面那排**（continual）。zero-shot navigator＝**把上面那排補上**。
- **ZeroSlide 本人**＝不訓練 + **連選都不選（吃全部 patch）** + 只做分類 → 在這張表**外面更上層**（連 budget 都不要）。我們的表多了一條 Agent 軸（在 budget 下選）。

**同一個架構，換引擎**：Observation State / budget / sequential 迴圈 / Context Gate / NSM 全不動，只換 `NavigationPolicy` 的打分來源——
- continual：分數來自訓練過的 router（MicroRouterV0）。
- zero-shot：分數來自 CONCH 文字相似度（`summary_feats` 已算的 `sim_txt_max`）；NSM 自動變空殼（沒東西可記）。

**要不要訓練/跑 RunPod？** zero-shot navigator **本身不訓練**；只需跑一次 inference 評估（特徵都現成，esca 在 Mac 即可，大任務再上 RunPod）。**不是重寫**，是加一個 `policy_mode="zero_shot"` 分支（見 `specs/features/SPEC-07`）。

**對 defend 的價值**：把老師「為何不像 ZeroSlide 用 zero-shot？」「selection forgetting 是否 trivial？」直接變成一個 ablation——continual > zero-shot 就證明「訓練+記憶的 navigation」值得；打平就誠實報告 + 留下強 baseline。

---

## G. 兩種遺忘、為什麼要做 zero-shot navigator、誰才是 Agent

### G1. 我們的 zero-shot 跟 ZeroSlide「哪裡一樣」
**一樣的是工具，不一樣的是工作。**
- 一樣：兩者都用 **frozen 病理 FM 的文字對齊、零訓練**（靠 FM 內建語義，不學 task-specific 參數）。
- 不一樣：ZeroSlide 拿來做**分類**（看全部 patch → 出診斷，不選）；我們拿來做**選 patch（看哪）**，選完再交給 frozen backbone 分類。
- 故：我們的 zero-shot navigator＝**把 ZeroSlide「不訓練、靠 FM 語義」這招從『分類』搬到『選哪裡看』**。

### G2. 真的有兩種遺忘（遺忘住在「會被訓練的模組」裡）
| 遺忘 | 住哪 | 內容 | 我們的狀態 |
|---|---|---|---|
| **classification forgetting** | 分類頭 | 「這片是什麼癌」決策邊界被洗掉 | **不存在**——backbone 凍結，分類頭不訓練 |
| **navigation forgetting** | router | 「這種癌看哪」策略被洗掉 | **唯一存在**——只有 router 會訓練 |

→ 因為故意凍結 backbone，**整條 pipeline 唯一會遺忘的就是 navigation**；贏了才能歸因給我們。

### G3. ZeroSlide 對抗哪種遺忘？
**classification forgetting**，且解法是「**根本不訓練**（zero-shot 分類）→ 沒參數可洗 → 天生零遺忘」。它**完全沒碰 navigation**（看全部、不選），所以對 navigation forgetting 一句話都沒說。
→ 「他們解了 navigation 遺忘、我們輸給他們」**不會發生，因為 ZeroSlide 沒有 navigator**。

### G4. 真正的對手是「我們自己的 zero-shot navigator」；輸了怎麼辦
比的是 **zero-shot navigator（不訓練的選擇）vs continual navigator（訓練 router + NSM）**：
- **continual 贏** → 訓練+記憶的 navigation 值得，遺忘問題值得解。
- **zero-shot 打平/更好（擔心的「輸」）** → **不是末日，是可發表發現**：如同 ZeroSlide 說「分類不用訓練」，我們說「**navigation 也不用訓練，frozen FM 就知道看哪**」——把它的結論延伸到 agent/navigation。我們提出問題、造 baseline、給答案，本身即貢獻。

**即使打平，我們不只剩「軌跡分析」，還有：**
1. **可吸收未來訊號（最重要）**：zero-shot 文字相似度寫死、學不動；醫師 gaze、偏好回饋（RLHF）、reward 來了用不上。只有可訓練 agent 能學進去 → zero-shot 有天花板。
2. **真正序列/狀態決策**：zero-shot 是固定 per-patch 分數，本質偏 one-shot；state-conditioned「看 A 再決定看 B」要靠可學策略。
3. **換 backbone 穩健**：zero-shot 全靠 FM 文字對齊強（CONCH 有）；對齊弱的 backbone 它就瞎，可訓練 navigator 還能學。
4. **CL 框架本身**：NSM / gate / 序列觀察是可重用通用模組，zero-shot 只是其中「不訓練」的退化模式。

### G5. 誰比較適合「Agent」？→ 可訓練的（continual）
Agent＝「依累積觀察做序列決策、能被 reward 優化、能採取行動」的**策略**。
- zero-shot navigator＝固定啟發式分數，比較像**反射**：能塞進序列迴圈，但不學、不能被回饋改進 → 好 baseline / 初始化，不是會成長的 agent。
- continual navigator＝**可學的 policy**：能 conditioned on state、被 reward 優化、有醫師資料就變強 → Agent 敘事的本體。

**一句話定位**：*zero-shot navigator 是「反射式」強 baseline（證明不訓練能走多遠）；continual navigator 是「可學的 agent」，是唯一能吸收未來回饋、做真正序列決策、且不挑 backbone 的路。我們不賭現在就贏 zero-shot，我們賭「navigation 是 CL 問題，可學的 agent 是它唯一能成長的底座」。*

## 更新日誌
- 2026-06-27（深夜）：建立本篇（ZeroSlide 對比 + budget/forgetting 回應 + zero-shot navigation baseline）。
- 2026-06-28：新增 F 節（zero-shot 用詞陷阱 + 2×2 表 + 同架構換引擎），對應 `specs/features/SPEC-07`。
- 2026-06-28：新增 G 節（兩種遺忘、與 ZeroSlide 同異、輸了怎麼辦、誰適合 Agent）；SPEC-07 程式已實作 + Mac smoke 通過。
