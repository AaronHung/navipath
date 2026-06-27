# 06 · 架構 block 白話詞彙表（glossary, append-only）

> 架構圖上每個 block 的人話解釋。被老師/reviewer 問時可直接查、可當 rebuttal 彈藥。
> 規則：只增不刪，未來持續添增。對應圖：`site/architecture.html`、`site/figs/arch_navipath_cl*.svg`。

---

## 一、輸入與 backbone（藍色）

**Whole-Slide Image / Patch Pool**
一張病理切片是 gigapixel 巨圖，切成成千上萬個小方塊（patch）。這堆 patch ＝ patch pool。

**Feature Interface（CONCH / Pathology FM）**
不直接吃像素：先用預訓練模型（CONCH）把每個 patch 變成 512 維向量（feature）。是「翻譯官」，把圖翻成 backbone 與 agent 都看得懂的數字。

**Frozen Diagnostic Backbone（凍結的診斷骨幹）**
真正下診斷的模型（本次＝QPMIL-VL）。**Frozen＝完全不訓練它**，只借它看 patch、出診斷。凍結是故意的 → 贏了才能歸因是「我們的 navigation 贏的」，不是它的功勞。

**Slide-level Prediction**
backbone 看完我們挑的 patch 後，給整張片子的診斷（癌種/分數）。

---

## 二、貢獻層 CNL（綠色）＝ Agent

**Observation State（觀察狀態）**
agent 看單張片子時的「當前筆記/短期記憶」：看過哪些 patch、證據像哪一類、backbone 多有把握、看了幾%。**讓「下一步看哪」取決於已看到的東西 → 序列決策。**

**Navigation Policy（導覽策略）**
決策大腦：依 Observation State 幫每個未看 patch 打分，決定哪些值得看。

**Budgeted Sequential Observation（有預算的序列觀察）**
- *Budgeted*＝只准看 K 個 patch（模擬醫師無法每處細看）。
- *Sequential*＝分多輪看，不是一次挑完。
- 流程：看一點 → 更新 state → 再決定看哪 → 再看，直到用完 K。

**Navigation Trace（導覽軌跡）**
agent 依序看了哪些地方的紀錄。＝可解釋性產出，攤給醫師看「牠怎麼看的」；也是跟 QPMIL「一次吃全部」最大的差異。

---

## 三、CL 記憶（紫色）＝ 持續學習

**Context / Task Gate（情境/任務閘門）**
一個**開關/路由器**：看到新片子，決定**這張要用 NSM 裡的哪一套技能**（像把病例分流到對的專科）。

**oracle（理想答案來源）**
ML 裡 oracle ＝「直接給正確答案的理想來源」，一種合理作弊，用來量**最好情況（upper bound 上界）**。
*oracle gate* ＝ 我們**直接告訴 Gate 這張是哪個癌種**（實驗資料本來就知道答案）→ Gate 一定挑對技能。
- 為何：先拿掉「分流分錯」這個變數，**單純驗證記憶+技能這套有沒有用**。
- 未來（task-free）：不給答案，模型自己從片子推任務 → 完整版，列 future。

**Navigation Skill Memory（NSM，導覽技能記憶）**
**跨任務長期記憶**：學完一個癌種，就把「看這種癌該怎麼看」的技能存進「技能銀行」。CL 的記憶體在此 → **忘記與恢復都發生在這裡**。

**CL Update（持續學習更新）**
管理 NSM 的規則：學新任務時**新增**技能、**凍結**舊技能、（未來）**合併/重播**，避免學新洗掉舊。

> 分工一句話：**Observation State＝單片內短期記憶（Agent）；NSM＝跨任務長期記憶（CL）；Gate＝決定這張片子拿哪套技能。**

---

## 四、虛線框（Future work，這次不做，只標路線）

**Feature Replay Memory（特徵重播記憶）**
未來存「代表性 patch 的特徵向量」，學新任務時拿出來複習防遺忘。**存特徵，跟 NSM 存「技能」不同。**

**Reward / Preference Model（獎勵/偏好模型）**
未來有醫師資料時，評分「agent 看的順序好不好、有沒有看對地方」。

**North Star signals**
長期願景訊號：真實醫師瀏覽軌跡、放大/移動動作、人類回饋（RLHF）。現在沒這些資料 → 列未來。

---

## 五、底部與標籤

**Continual task stream（t = 1 … T）**
任務一個接一個來（癌種 1→2→3…），不是一次全給。＝持續學習的舞台：學後面會不會忘前面。

**Phase-0 instance: QPMIL-VL + CONCH**
提醒：整套是通用框架，**QPMIL 只是這次插進 backbone 槽的一個例子**，非主角。

---

## 六、細部圖的介面契約三呼叫

**encode(WSI) → Z**：給每個 patch 的特徵。
**predict(subset) → logits**：只給**一部分** patch 也能出診斷（我們只給 K 個）。
**task_query(WSI) → q**：（選用）推測任務是哪類，task-free 才用。

> 完整契約與相容範圍見 [05](05_interface-contract-and-compatibility.md)。

## 更新日誌
- 2026-06-27：建立 glossary（架構圖全 block 白話）。
