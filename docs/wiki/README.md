# NaviPath-CL 知識庫 Wiki（append-only）

> 這是我們的**設計 / 技術 / 架構 wiki**——隨時可讀、持續累積。
> 規則：**只增不刪**，舊認知要改時用「更新註記」標日期，不直接覆蓋，避免遺忘脈絡。
> 對外敘事禁用詞：mechanism probe / selection、selector、GO-NO-GO、「比較 shared/EWC/per-task」、compute-saving、forgetting 診斷。

## 索引

| 編號 | 主題 | 一句話 |
|---|---|---|
| [01](01_architecture.md) | 架構與名詞（Agent/CL/State/Gate/NSM） | 我們到底做什麼、Agent 在哪、CL 在哪 |
| [02](02_qpmil-vl-relationship.md) | 與 QPMIL-VL 的關係 | 我們用了它什麼、跟它差在哪、它的 prototype/prompt 為何不是我們的 policy |
| [03](03_success-criteria-experiments.md) | 成功判準與實驗設計 | 什麼數字算 POC 成功、跟誰比、為何不用贏 QPMIL |
| [04](04_generalization-and-attribution.md) | 通用化定位 與 歸因 | 通用 module＝我們的貢獻本體；贏了憑什麼是我們的機制 |
| [05](05_interface-contract-and-compatibility.md) | 介面契約 與 相容性範圍（versioned） | backbone 要給什麼才能插進來、什麼能插什麼不能、隨版本演進 |
| [06](06_glossary.md) | 架構 block 白話詞彙表 | 圖上每個 block 在說什麼（看板 `site/glossary.html`） |
| [07](07_what-we-do-deep-dive.md) | 我們在做什麼（深入）/ N2 驗證什麼 | seq vs oneshot、Agent/CL 兩軸、router 無標註怎麼學會看哪、為何只測 esca |
| [08](08_N3-analysis-defense.md) | N3 分析的答辯觀念 | CL 指標(mACC/Forgetting/BWT)、retention 表怎麼讀、budget 曲線、3-fold |
| [09](09_zeroslide-and-rebuttals.md) | ZeroSlide 對比 與 老師疑慮回應 | budget 必要性、selection forgetting 是否 trivial、zero-shot navigation baseline |
| [10](10_mechanism-defense-and-multistep.md) | 機制底層原理 + 多步路線圖 | 「不只是存下來」怎麼 defend、多步沒醫師 trajectory 怎麼訓練、結果↔主張對應 |
| [11](11_mechanism-walkthrough-and-terms.md) | 機制逐層拆解（白話）+ budget 省算力表示 + 術語對照 | router 學什麼/NSM 存「鑰匙」不是「訊號」、便宜記憶升級、RLogist/Cordonnier、oracle 等黑話翻譯 |

## 相關正式文件
- 架構決策：`specs/decisions/ADR-0006-*.md`（authoritative 架構圖 + 標題 + 對應表）
- pivot 緣由：`specs/decisions/ADR-0001-*.md`
- 敘事錨（待改寫為新架構）：`STORYLINE.md`

## 更新日誌
- 2026-06-27：建立 wiki，新增 01/02/03。
- 2026-06-27（晚）：新增 04（通用化/歸因）、05（介面契約/相容範圍, versioned）；校正「通用 module＝我們的貢獻本體」。
- 2026-06-27（晚）：新增 06（架構 block 白話 glossary），同步看板 `site/glossary.html`。
- 2026-06-27（深夜）：新增 07（N2 深入/我們在做什麼）、08（N3 答辯觀念）、09（ZeroSlide 對比 + 老師疑慮回應）；新增看板分頁「答辯筆記」`site/notes.html`（append-only），並掛進所有頁面導覽。
- 2026-06-28：09 新增 F 節（zero-shot navigator 用詞陷阱 + 2×2 表 + 同架構換引擎）；同步看板答辯筆記；新增 `specs/features/SPEC-07-zero-shot-navigator.md`（policy_mode=zero_shot 實作規格）。
- 2026-06-28：09 新增 G 節（兩種遺忘 classification vs navigation、與 ZeroSlide 同異、輸了怎麼辦、誰適合 Agent）；同步看板。**SPEC-07 程式已實作**（`sequential_observation.py` 加 `policy_mode`、`eval_sequential_observation.py` 加 `--policy-mode zero_shot`），Mac smoke 通過。
- 2026-06-28：**N2 完整跑完（reverse 3-fold）＋ N3 分析出爐**：`analyze_seqobs_n3.py`、`outputs/RESULTS_seqobs_20260628.md`、`site/figs/n3_*.png`；看板新增「D · N2/N3 Pilot 結果」。**新增 10（機制防禦＋多步路線圖）**，回應老師「不只是存下來」與「多步怎麼訓練」；同步看板答辯筆記（第四區）與架構頁說明。
- 2026-06-28：**N4 報告稿改寫**（`reports/bimonthly_2026-07-03.md` 轉新 agent+CL 敘事＋真實數字）。**新增 11（機制逐層拆解白話版）**：router 學什麼/NSM 存「鑰匙」非「訊號」、便宜記憶 prompt/LoRA/replay 升級、North Star roadmap、budget 省算力的其他表示（引 RLogist AAAI'23、Cordonnier CVPR'21）、oracle 等術語對照表；同步看板答辯筆記第五、六區。
