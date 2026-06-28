# 11 · 機制逐層拆解（白話完整版）+ budget 省算力的其他表示 + 術語對照（append-only）

> 給「第一次接觸我們設計的人」的溫和導讀：N2/N3 在做什麼、唯一在訓練的是什麼、NSM 到底存什麼、為何不是「把東西存下來」、便宜記憶怎麼升級、budget 想講省算力該怎麼講、內部黑話如何翻成 reviewer 看得懂的詞。
> 對照實作：`navipath_moe/routers.py`（MicroRouterV0）、`train_router_v0.py`（train_router_one_task）、`navipath_moe/continual_agent.py`（NavigationSkillBank/ContextGate）。
> 規則：只增不刪。

---

## 0. N2 / N3 一句話
- **N2（GPU）**：依序訓練 4 種癌症的 navigation policy（router），每學完一個存快照；再評估「有記憶 / 無記憶 / zero-shot」在不同 budget 下的表現。
- **N3（Mac，純分析）**：把 N2 的 JSON 彙整成 retention 表、CL 指標（mACC/Forgetting）、budget 曲線。

---

## 1. 架構錨點：唯一在訓練的東西 = 一個小 MLP（router）
router＝`MicroRouterV0`＝兩層 MLP：`Linear(516→256) + GELU + Linear(256→1)`，約 **13.2 萬個參數（≈533KB）**。

- 輸入（每個 patch）：`[512 維 CONCH 特徵 Z_i ; 4 個相似度統計]`。
- 輸出：每個 patch 一個「重要度分數」。
- **這個 MLP 就是 policy（「看哪」的策略本體）。**

| 元件 | 訓練？ | 來源 |
|---|---|---|
| patch 特徵 Z_i | ❌ 凍結 | CONCH 預先算好 |
| class-text f_txt、prototype F_p | ❌ 凍結 | QPMIL backbone |
| **router MLP 權重** | ✅ **唯一在學** | 隨機初始化後弱監督訓練 |
| prompt | — | 這版**沒有**（N6 才加） |

---

## 2. router 學什麼、怎麼學、為什麼 CONCH 不夠
**怎麼學（弱監督，只有 slide-level label）**：router 給所有 patch 打分 → 取最高 K 個 → softmax 權重聚合成一個向量 → 丟進**凍結的**分類頭 → 跟片子真實 label 算 cross-entropy → **梯度只回傳調 router**。重複多次後，router 為了讓「它挑的 patch」能分類正確，被迫學會把**判別區打高分**。沒人標病灶，它從「挑了→對不對」的回饋自學。

**為什麼 CONCH 已會看片、router 還能學到東西？**
> CONCH 只會把**單一一個 patch** 變成 embedding（看懂一塊）。它**沒有「在幾千 patch 裡取捨、排序」的能力**——它不是設計來做選擇的。router 學的正是這個「跨 patch 取捨」，是 CONCH 沒有的能力。

---

## 3. ★ NSM 存的是「鑰匙」，不是「訊號」
- **naive（nonsm）**：自始至終**只有一份** router。換 task＝同一組 13.2 萬權重被新 task 梯度**持續覆寫**，esca 調好的被 lung 蓋掉 → 拿去看 esca 就選錯（0.333）。
- **有記憶（NSM）**：每學完一個 task，把**那份 router 權重拷貝一份**存起來（`task_id → state_dict`）。推論時 gate 取對應那份。

**存的具體是什麼（精確）**：每個任務一份 `MicroRouterV0` 的 `state_dict`＝`mlp.0.weight[256,516]`+`mlp.0.bias`+`mlp.2.weight[1,256]`+`mlp.2.bias`。**不是** 64 個 patch、**不是** features、**不是** prototype/prompt。檔案佐證：`skill_bank_*.pt ≈ 2.1MB = 4 × 533KB`。

**「導覽能力」「訊號」的精確對應（修正口語）**：

| 口語 | 精確對應 | 在哪、會不會丟 |
|---|---|---|
| 導覽能力 | router MLP 的「打分函數」 | 13.2 萬權重；naive 會被覆寫而丟 |
| 訊號（判別資訊） | esca 判別 pattern 在 **frozen CONCH embedding + 分類頭** | 在 backbone，**永不丟** |

→ 結論：**遺忘的不是訊號，是「讀取訊號的鑰匙（打分函數）」。** router 存的是鑰匙；訊號本體在 frozen backbone，不需存。4 把鑰匙都用，oracle 按 task 取對應那把。
→ **你問的好點**：未來 task-free 時，鑰匙存著沒問題，**難在「沒給 task_id，怎麼知道開哪把鎖」**——推錯就載錯鑰匙。這就是 oracle 只是上界、task-free gate 是真難題的原因。

---

## 4. ★ 誤會澄清：NSM 是「上界」≠「不是好機制」≠「跟 full-patch 一樣」
- **「上界」是理想天花板，不是爛**：它證明「若記憶不用錢、技能完美保護，遺忘可解到 0.935」＝好消息（訊號還在、可救回）。它唯一不漂亮處：**每任務存整份 router，參數隨任務線性膨脹、不可規模化** → 所以是「要逼近的目標」，不是終點方法。
- **跟 full-patch 完全不同軸**：

| | full-patch（ZeroSlide） | 我們的 NSM |
|---|---|---|
| 管的軸 | 看幾個 patch（budget/agent） | 跨任務記不記得（CL） |
| patch | 看**全部**（不挑） | **只挑 64 個**（要會挑） |
| 「會看哪」 | 沒有（不需選） | **有，且是訓練出來的** |

→ 我們**只看 ~1~2% 的 patch 就追平/超過看全部**（esca 0.911 vs full 0.867），這正是 full-patch 做不到的。

---

## 5. 便宜記憶升級（N6 / 衝 7/20 的核心）
痛點：NSM 每任務存整份 533KB router。三條把它變便宜的路：

| 方法 | 怎麼做（對應我們架構） | why | 每任務存多大 |
|---|---|---|---|
| **Prompt** | 共用一份 router base 凍住，每 task 只學一個小 prompt 向量拼進輸入；gate 取該 task 的 prompt | 共享主幹、可遷移、存最小 | 幾 KB |
| **LoRA** | router 的 `Linear` 凍住，每 task 加低秩增量 ΔW=A·B（r≈4~8），只訓 A,B | PEFT 標準、好訓、數學乾淨 | ~4K 數（小 ~30×） |
| **Replay** | 維持一份 router，訓新 task 時混入舊 task 少量樣本一起訓 | 直接用複習對抗遺忘，不長參數（改存少量資料） | 一個小 buffer |

**升級的意義**：從「每任務存整份 router（上界、不可規模化）」→「存幾 KB 的 prompt/LoRA 或一個小 replay buffer，逼近 0.935 上界」＝把「上界證明」升級成「**可規模化、可部署的 CL navigation 真方法**」。
**7/20 誠實目標**：prompt 或 LoRA **擇一**做出來、證明逼近上界，外加多步路線 A（推論期自適應，Mac 可做）。三種全做＋RL＋task-free gate **不切實際**，列 7/20 之後。

---

## 6. North Star roadmap（7/20 之後）

| 名詞 | 人話：改進什麼 | 途徑 / 訓練 | 需要什麼 data | 可行性 |
|---|---|---|---|---|
| task-free Context Gate | 沒人告訴你這是哪種癌，模型自己認出該載哪把鑰匙 | 小分類器吃 slide 級特徵 → 預測 task → 載對應 skill | 現有特徵（訓練時有 task label） | 高（風險：相似癌種認錯） |
| PEFT 擴展（prompt/LoRA） | 把便宜記憶擴到更多器官/癌種 | 沿用 §5，每新癌種加一組 | 更多 WSI | 高 |
| 多步 RL navigation | 真的邊看邊決定下一步、序列勝單步 | REINFORCE/bandit，reward＝最小 budget 下診斷正確 | 仍只要 slide label（**不需 trajectory**） | 中（RL 變異大、需 GPU） |
| Move/Zoom 多倍率 | 像醫師會放大、平移 | action 加 zoom/pan；需多倍率特徵 | 多倍率 WSI tiles（要重抽） | 中偏難 |
| RLHF / 醫師軌跡 | 導覽貼近真實醫師習慣 | 收集醫師軌跡當 reward/imitation | **新資料：醫師軌跡**（最難取得） | 低/長期 |
| Select-before-encode | 真正省算力：encode 前先篩 | 輕量 pre-scorer 在 encode 前選 | 現有 | 中（budget 真正省算力版） |

---

## 7. ★ budget 想講「省算力」該怎麼講 + 別人怎麼做的
**老師說得對**：我們現在是 encode-all→aggregate，encoder 已跑全部 patch，budget 省不到 encoder 算力。真正能省算力的範式是 **selection 發生在 encoding 之前（select-before-encode / coarse-to-fine）**。相關做法（可引用）：

- **RLogist**（Zhao et al., **AAAI 2023**；code: tencent-ailab/RLogist）：RL agent 模仿病理醫師，**跨倍率找有價值區域，不必對整張片在高倍率密集抽特徵** → 直接省下昂貴的 patch 特徵抽取，且觀察路徑可解釋。**與我們 North Star 幾乎同向，但它單任務、無 CL** → 我們的增量＝**RLogist 式快速觀察 + continual（skill memory）**，正好是老師建議的「frozen FM + continual selector」。
- **Differentiable Patch Selection**（Cordonnier et al., **CVPR 2021**）：用可微分 Top-K，在**下游處理前**就選出最相關 patch 來省高解析影像的記憶/算力，端到端可訓練、無需 bbox 標註。＝select-before-encode 的通用模組。
- **Coarse-to-fine / thumbnail-guided ROI**：先看低倍率縮圖找 ROI，只在 ROI 高倍率 encode（典型省算力路線）。
- **Anytime / dynamic inference**（如 Glance-and-Focus 類）：看到夠確定就停 → 省算力，對應我們多步路線 B（信心早停）。

**我們的概念可改用的表示法（避免被誤讀成「省 predictor 算力」）**：
- **evidence efficiency / diagnostic sufficiency**（用最少證據達到足夠診斷）。
- **test-time observation budget**（推論時觀察預算，呼應醫師看片有限注意力）。
- **interpretable evidence selection / auditable evidence trail**（可稽核證據鏈）。
- **anytime budgeted inference**（可早停的預算化推論）。
- **未來真省算力**：明講「現在 budget 是診斷探針/可稽核；select-before-encode 版才省 encoder 算力，是 roadmap」（誠實、且有 RLogist/Cordonnier 背書方向）。

---

## 8. ★ 術語對照（內部黑話 → reviewer 看得懂）
被很多 reviewer 不熟的詞，建議對外改用右欄：

| 我們內部詞 | reviewer 友善說法 | 白話一句 |
|---|---|---|
| **oracle gate** | *task-aware (oracle) setting* / known task identity | 測試時假設已知這片屬哪任務（理想**上界**） |
| **NSM**（per-task router） | *parameter isolation* / per-task adapters | 每任務存一份專屬參數，互不干擾（CL 標準法之一） |
| **nonsm / naive** | *naive sequential fine-tuning* | 一路微調同一模型、不做任何防遺忘 |
| **navigation / where-to-look** | *patch selection policy* | 決定看哪些 patch 的策略 |
| **navigation skill** | *task-specific policy / adapter* | 某任務專屬的選片技能 |
| **Continual Navigation Layer** | *continual patch-selection module* | 可持續學習的選片模組 |
| **navigation forgetting** | *forgetting of the selection policy* | 選片策略對舊任務退化 |
| **context gate (task-free)** | *task-identity inference / task router* | 自動判斷這片屬哪任務 |
| **budget K** | *observation / patch budget (test-time)* | 推論時最多看幾個 patch |
| **Navigation Trace** | *selected-patch sequence / evidence trail* | 選出的 patch 順序＝證據鏈 |
| **zero-shot navigator** | *training-free patch selection (FM similarity)* | 不訓練、用 FM 文字相似度選片 |

> replay、task-free、PEFT、LoRA、prompt 這些已是社群通用詞，保留即可。

---

## 9. ★ 兩條軸別混淆 + N6→7/20 時間軸（最常被問的 scope）
**便宜記憶（CL 軸）和多步（Agent 軸）是兩件獨立的事，互不相干：**

| | **CL 軸（記憶）** | **Agent 軸（觀察）** |
|---|---|---|
| 管什麼 | 跨任務怎麼記住「每癌種怎麼選片」 | 一張片內怎麼一步步看 |
| pilot 現況 | NSM＝每任務存整份 router（貴、上界） | one-shot 靜態 top-K |
| N6→7/20 升級 | **便宜記憶（建議 LoRA）** 逼近上界 | **多步 route A（已做）+ B** |
| 與對方關係 | **與多步無關**（純 CL 提升） | **與記憶無關**（純 Agent 提升） |
| 之後（North Star） | PEFT 擴展更多癌種 | 多步 RL（route C）、move/zoom |

**N6 同時推這兩條，但它們是獨立工作流。** 便宜記憶＝CL 機制提升，跟 Agentic 無關；多步＝Agentic 提升，跟記憶無關。

**時間軸（誠實可行性）**：

| 項目 | 7/20 前？ | 說明 |
|---|---|---|
| 便宜記憶 PEFT（**LoRA** 擇一） | ✅ 做得到 | CL 軸；router 是純 MLP，LoRA 最自然、每任務 ~4K 數 |
| 多步 **route A（已做）+ B** | ✅ 做得到 | Agent 軸；非 RL、不用訓練 |
| paper-order 對稱 + EWC baseline | ✅ 可選 | 需重開 GPU |
| **多步 RL（route C）** | ❌ 7/20 之後 | RL 變異大、需 reward 設計＋大量 GPU |
| task-free gate | ⚠️ 勉強/未必 | 真難題 |
| move/zoom、RLHF/醫師軌跡 | ❌ 之後 | 需新資料（多倍率 tiles / 醫師軌跡） |

→ **PEFT 是 7/20 目標；多步是 7/20 目標（route A/B 非 RL 版）；多步 RL 不是 7/20，是之後。**

## 10. ★ λ（lambda）白話 — 多樣性旋鈕
多步 route A 裡，每看完一批 patch，對「跟已看過的很像」的 patch **扣分**；扣多重由 λ 決定：
- **λ=0**：不扣 → 退化成 one-shot（只看最高分，可能全擠同一區）。
- **λ 中**：主要看重要度，順便避一點重複。
- **λ 大**：用力避開重複 → 強迫看遍不同區域。
- **太大**：可能為追「新」跳過真病灶 → 所以要 sweep（試 1/2/4）找最佳。
> 合成驗證：λ=0 全擠一群、λ=4 覆蓋四群。不需訓練、不需醫師軌跡，是推論時的搜尋規則。

## 更新日誌
- 2026-06-28：建立本篇（機制逐層拆解白話版＋budget 省算力的其他表示與相關論文 RLogist/Cordonnier＋術語對照表）；同步看板答辯筆記第五、六區。
