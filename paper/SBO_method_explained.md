# ZeroNav 完整方法說明
## ZeroSlide → ZeroNav → SBO + MMR 全鏈路理解

---

## 0. 老師要的實驗結果 & 訓練完後要跑的指令

### 老師要求的三件事

| # | 老師要求 | 對應指令 | 輸出位置 |
|---|---|---|---|
| 1 | **所提方法 + ZeroSlide 的比較數據**（one-shot vs multi-step） | `eval` | `outputs/zeronav/eval/` |
| 2 | **Router output vs TASK_ID 的相似度**（router 有沒有學到任務相關） | `analyze` | `outputs/zeronav/router_analysis_*.json` |
| 3 | 架構圖（已更新，無 prototype） | 已完成 | `experiment_visualize/figs/` |

### 訓練完後的指令（按順序貼）

**Step A：EVAL（λ sweep，inference only，不重訓）**
```bash
python run_zeronav.py eval \
    --backbone-ckpt outputs/backbone_reverse_fold1.pt \
    --order reverse --fold 1 \
    --lambdas 0.0,0.1,0.25,0.5,0.75,1.0,1.5,2.0 \
    --budget 64 --step-size 16 \
    2>&1 | tee logs/zeronav_eval_f1.log
```

輸出四種模式 × 8 個 λ × 4 個 task 的 F1/AUC：
```
zeronav_multishot   zeronav_oneshot
zeroshot_multishot  zeroshot_oneshot
```

**Step B：ANALYZE（router vs task-ID 相似度，inference only）**
```bash
python run_zeronav.py analyze \
    --backbone-ckpt outputs/backbone_reverse_fold1.pt \
    --order reverse --fold 1 \
    2>&1 | tee logs/zeronav_analyze_f1.log
```

輸出兩個矩陣：
- **Router weight cosine similarity**：Router_0 的參數向量 vs Router_1、2、3 有多像？（應該對角線高，非對角線低）
- **Cross-task accuracy**：Router_i（為 task i 訓練的）拿去評估 task j，準確率如何？（應該對角線最高）

**Step C：git push 結果**
```bash
git add outputs/zeronav/ logs/
git commit -m "results(zeronav): fold1 reverse lambda-sweep + router analysis"
git push origin main
```

**Mac 上 pull 回來查看：**
```bash
cd /Users/aaron/research/01_navipath && git pull origin main
ls outputs/zeronav/eval/
cat outputs/zeronav/router_analysis_reverse_f1.json
```

---

## 1. 全局理解：ZeroSlide → ZeroNav 的進化

### ZeroSlide 做了什麼？

ZeroSlide 是 zero-shot WSI 分類方法：

$$\text{score}_i = \max_{c \in \mathcal{C}} \cos(Z_i, f_c^{\text{txt}})$$

每張 patch 的重要性 = 它和「哪個類別的文字描述最相似」的最大分數。

**優點：** 完全不需要訓練，只要有 CONCH（或 CLIP 類模型）就能用。

**缺點：**
- 靜態公式，不會根據任務特性調整
- 不知道「哪種 patch 對最終分類決策最關鍵」，只知道「和文字描述最像」
- 兩件事看起來相似，但其實不同：文字說「腫瘤有細胞核增大」，但最終分類靠的不一定是核最大的那張 patch

### ZeroNav 做了什麼（在 ZeroSlide 基礎上進步）？

ZeroNav = **「ZeroSlide 的特徵設計 + 任務相關的學習」**

**特徵設計繼承自 ZeroSlide（backbone-agnostic）：**
$$\mathbf{x}_i = \left[ Z_i \;;\; \max_c \cos(Z_i, f_c^{\text{txt}}) \;;\; H_i^{\text{txt}} \right] \in \mathbb{R}^{514}$$

其中第二項為 ZeroSlide score（最大 text-patch cosine similarity），第三項為分類不確定度（text entropy）。

**然後用 MLP 學習如何利用這些特徵：**
$$s_i = \text{TextNavRouter}(\mathbf{x}_i) = \text{MLP}(\mathbf{x}_i)$$

**關鍵差異：**

| | ZeroSlide score | ZeroNav score |
|---|---|---|
| 學習過程 | 無，純公式 | 有，Cross-Entropy 訓練 |
| 能「知道」的事 | 這張 patch 和文字描述有多像 | 這張 patch 對**這個任務的最終分類**有多重要 |
| 對不同任務 | 全部任務同一個公式 | 每個任務一個專屬 Router |
| Prototype 依賴 | 無 | **無**（這是 pivot 的核心） |

---

## 2. 為什麼不直接用全部 N 張 patch？

### WSI 的現實

一張 WSI 切片通常有 3000～10000 個 patch（256×256 pixels）。

| 做法 | 計算量 | 準確度 |
|---|---|---|
| 全部 N 張送進 backbone | $O(N)$ backbone forward pass | Oracle（上界） |
| 隨機抽 64 張 | O(1) | 差，可能全選到正常組織 |
| Top-64（one-shot） | 需要先對 N 張打分，再選 | 中等 |
| ZeroNav SBO（4×16） | 同上，但 4 輪自適應 | 我們的主張：比 top-64 好 |

**Budget 限制（64 張）的原因：**
1. 推論速度：臨床環境不允許處理幾千張
2. 比較公平：所有方法在相同計算量下比較
3. 科學問題：**在有限 budget 下，如何選這 64 張最有效？**

---

## 3. Top-64 的陷阱（為什麼需要 SBO）

### 案例一：腫瘤集中型 WSI

```
WSI 結構：
████ ░░░░░░░░░░░░░░░░░░
████ ░░░░ normal ░░░░░░
████ ░░░░░░░░░░░░░░░░░░
腫瘤（80 patches，score 都很高）
```

- **Top-64**：選了 64/80 個腫瘤 patch，全是同一個區域，高度冗餘
- **SBO Round 2~4**：Round 1 確認腫瘤後，被強制去看邊界、周圍間質、血管侵犯等

→ SBO 提供更完整的病理學依據

### 案例二：多病灶型 WSI（更典型的診斷挑戰）

```
WSI 結構：
░░ 腫瘤 A ░░░░ 發炎區 ░░
░░░░░░░░░░░░░░░░░░░░░░░
░░░░░ 壞死 ░░░ 腫瘤 B ░░
```

腫瘤 A 的 score 高，但**腫瘤 B 的存在**也是確診的關鍵。

- **Top-64**：全選腫瘤 A 附近（score 最高），完全沒看到腫瘤 B
- **SBO**：Round 1 選腫瘤 A，Round 3 被 MMR 逼到腫瘤 B 和壞死區

→ SBO 在需要**全局多種線索**的診斷任務中有明顯優勢

### 什麼時候 Top-64 夠好？

- 任務很簡單：只要找到任何一個腫瘤 patch 就能確診
- WSI 高度均勻：整張切片都差不多，任何 64 張都代表整體
- 腫瘤線索只在一個地方

**λ sweep 的意義：** 讓資料說話——在這個任務/資料集上，top-64（λ=0）和 SBO（λ>0）誰更好。

---

## 4. Upper Bound 的層次結構

```
Oracle（全部 N 張）           ← 真正的 upper bound，不受 budget 限制
      ↑
ZeroSlide（全部 N 張，原論文） ← 若用全 N 張，成績比 top-64 高，但不公平比較
      ↑
zeroshot_oneshot（ZeroSlide score + top-64）  ← 在 budget=64 下的 zero-shot baseline
      ↑
zeronav_oneshot（trained router + top-64）    ← 學習讓選法更準確
      ↑
zeronav_multishot（trained router + SBO MMR） ← 我們的最佳主張（需要最佳 λ）
```

**論文的 claim：** `zeronav_multishot` > `zeronav_oneshot` > `zeroshot_oneshot`（在 budget=64 的限制下）

---

## 5. SBO 四輪選取完整說明

### 公式符號定義

| 符號 | 意義 |
|---|---|
| $Z_i \in \mathbb{R}^{512}$ | patch $i$ 的 CONCH embedding |
| $\hat{Z}_i = Z_i / \|Z_i\|$ | L2-normalized embedding |
| $s_i$ | TextNavRouter 對 patch $i$ 的 importance score |
| $\mathcal{S}_t$ | 第 $t$ 輪選出的 patch 集合 |
| $\lambda$ | 多樣性懲罰強度 |

---

### Round 1：無歷史，自由選最重要的

$$\text{adj}_i^{(1)} = s_i$$
$$\mathcal{S}_1 = \text{top-16}(\text{adj}^{(1)})$$

沒有已選 patch，完全按 router score 排序，取前 16 名。

---

### Round 2：強制避開 Round 1 的區域

更新「每個 patch 和已選集合的最大相似度」：

$$\text{max\_sim\_seen}[i] = \max_{j \in \mathcal{S}_1} \cos(\hat{Z}_i, \hat{Z}_j)$$

計算懲罰後的 adjusted score：

$$\text{adj}_i^{(2)} = s_i - \lambda \cdot \text{max\_sim\_seen}[i]$$

選出下一批（已選的 $\mathcal{S}_1$ 自動排除，因為 score 設為 $-\infty$）：

$$\mathcal{S}_2 = \text{top-16}(\text{adj}^{(2)}) \setminus \mathcal{S}_1$$

**臨床直覺：** 「剛才確認了腫瘤核心，現在去找跟腫瘤完全不一樣的地方——可能是腫瘤邊緣、周圍免疫細胞浸潤、或正常組織對照。」

---

### Round 3 & 4：遞推擴展

**增量更新**（只需對新選的 $\mathcal{S}_{t-1}$ 計算，不重算整個歷史）：

$$\text{max\_sim\_seen}[i] \leftarrow \max\!\Big(\text{max\_sim\_seen}[i],\; \max_{j \in \mathcal{S}_{t-1}} \cos(\hat{Z}_i, \hat{Z}_j)\Big)$$

$$\text{adj}_i^{(t)} = s_i - \lambda \cdot \text{max\_sim\_seen}[i]$$

$$\mathcal{S}_t = \text{top-16}(\text{adj}^{(t)}) \setminus \bigcup_{\tau<t}\mathcal{S}_\tau$$

**Round 3 臨床直覺：** 「前兩輪看了腫瘤核心和腫瘤邊緣，現在去找壞死區域或纖維化間質。」

**Round 4 臨床直覺：** 「三個區域都確認了，去找還沒探索的角落——可能發現第二個病灶，或確認整張切片的背景正常組織。」

---

### 最終預測

$$\text{prediction} = \text{backbone.aggregate\_and\_predict}(\mathcal{S}_1 \cup \mathcal{S}_2 \cup \mathcal{S}_3 \cup \mathcal{S}_4)$$

4 輪 × 16 張 = 64 張送進 backbone，做一次最終預測。

---

### λ 的作用視覺化

```
λ = 0:    [S1: 腫瘤核心16] [S2: 腫瘤核心16] [S3: 腫瘤核心16] [S4: 腫瘤核心16]
          ← 退化為 top-64，全部集中一處

λ = 0.5:  [S1: 腫瘤核心16] [S2: 腫瘤邊緣16] [S3: 發炎區16] [S4: 壞死區16]
          ← 平衡，每輪去新區域但還尊重 score

λ = 2.0:  [S1: 腫瘤核心16] [S2: 最遠的16] [S3: 更遠的16] [S4: 隨機16]
          ← 過度多樣，忽略了 score，選了很多低品質 patch
```

**最佳 λ 在中間**，由 sweep 實驗找到。

---

## 6. TextNavRouter 訓練原理

### 為什麼加入 Entropy 比 ZeroSlide 更好？

ZeroSlide 只用 `max_sim` 打分，但「最高分一樣」不代表「診斷價值一樣」：

```
Patch A：cos = [esca:0.72, rcc:0.10, brca:0.08, lung:0.09]
          max_sim = 0.72   ← ZeroSlide 看到這個
          entropy  = 低    ← 非常確定屬於 esca，診斷價值高

Patch B：cos = [esca:0.71, rcc:0.68, brca:0.67, lung:0.69]
          max_sim = 0.71   ← ZeroSlide 看到幾乎一樣
          entropy  = 高    ← 分不清哪個類別，身份模糊，診斷價值低
```

**ZeroSlide 認為兩張一樣重要，但 Patch A 明顯更有診斷價值。**

Entropy 補充了「確定性」這個維度：

$$H_i^{\text{txt}} = -\sum_c p_{ic} \log p_{ic}, \quad p_{ic} = \text{softmax}\!\left(\cos(Z_i, f_c^{\text{txt}})\right)$$

- **低 entropy** → patch 身份明確，強烈屬於某一類 → 診斷價值高
- **高 entropy** → patch 身份模糊，可能是過渡區或正常組織 → 診斷價值低

Router 將 `Z_i`（形態資訊）、`max_sim`（text alignment 訊號）、`entropy`（確定性）三者合一，學到比靜態公式更完整的「patch 重要性」估計。

---

### 為什麼用 CE Loss？

想讓 Router 學「哪些 patch 對分類有幫助」，最自然的方式：**讓 Router 選出的 patch 能正確分類整張 WSI，分類對了就說明選的好。**

```
Router 選 patch → 拿去分類 → 分類錯了 → 告訴 Router「你選錯了」→ 調整
```

這就是 CE Loss 做的事，只是中間有一個技術問題需要解決。

---

### 為什麼需要 Soft Top-K？

**問題：** 直接 hard top-K 選取的 `argmax` 梯度為 0，無法 backprop 告訴 Router「選錯了」。

**解法：** 用 softmax 做「軟性選擇」——分越高貢獻越大，梯度可以流回 Router。

$$w_i = \frac{\exp(s_i / \tau) \cdot \mathbf{1}[i \in \text{top-K}]}{\sum_{j \in \text{top-K}} \exp(s_j / \tau)}$$

### 梯度如何流回 Router

```
CE Loss
  ↓ ∂L/∂ŷ
backbone output（frozen）
  ↓ ∂L/∂Z_bag
加權平均 Z_bag = Σ w_i Z_i
  ↓ ∂L/∂w_i
softmax weights w_i（可微！）
  ↓ ∂L/∂s_i
Router MLP scores s_i  ← 梯度到這裡，更新 MLP 參數
```

Backbone（CONCH）全程凍結，**只有 TextNavRouter 的 MLP 參數被更新**。

### 設計選擇總結

| 設計選擇 | 理由 |
|---|---|
| **輸入 = Z + max\_sim + entropy** | Z 提供形態資訊；max\_sim 繼承 ZeroSlide 的 text alignment 訊號；entropy 補充「確定性」 |
| **Soft Top-K** 而非 hard selection | 讓梯度可以流過「選取」動作，使 Router 可訓練 |
| **CE Loss** 而非重建 loss / 對比 loss | 目標直接就是「分類準確」，和最終評估指標一致，無 proxy task 偏差 |
| **只訓練 Router，凍結 CONCH** | Router 是輕量 MLP（幾 KB），訓練快；凍結 CONCH 保持 text-patch alignment 不被破壞 |

---

### 完整訓練流程

```
對一張 WSI 的所有 N 個 patch：
    1. CONCH 提取特徵 Z_i（凍結，不更新）
    2. 計算 text-patch 統計量（max_sim, entropy）
    3. TextNavRouter 打出 score s_i（這裡有 grad）
    4. Soft top-K：取最高分的 64 張，計算加權平均
    5. 加權平均特徵送進 backbone classifier（凍結）
    6. 得到預測 logits
    7. Cross-Entropy loss
    8. Backprop 只更新 TextNavRouter 的 MLP weights
```

### Entropy 項 $H_i^{\text{txt}}$ 的作用（人話）

$$H_i^{\text{txt}} = -\sum_{c} p_{ic} \log p_{ic}, \quad p_{ic} = \text{softmax}_c(\cos(Z_i, f_c^{\text{txt}}))$$

- **低 entropy patch**（$H$ 接近 0）：這張 patch 和某個特定類別高度相似，「身份明確」
- **高 entropy patch**（$H$ 接近 $\log C$）：這張 patch 和所有類別差不多，「身份模糊」

**直覺：** 身份明確的 patch 更有診斷價值（如「這張明顯是腫瘤細胞」），身份模糊的 patch 可能是噪音或過渡區。

Router 把 entropy 作為輸入特徵，學習「什麼樣的 entropy 值對這個任務重要」——有些任務需要找最確定的 patch，有些任務邊界曖昧的 patch 反而是關鍵。

---

## 7. 四種選取策略比較

### 方法一：MMR（Maximal Marginal Relevance）— 已實作，預設

**核心機制：** 懲罰「和**任一**已選 patch 最像的」patch

$$\text{adj}_i^{(t)} = s_i - \lambda \cdot \max_{j \in \mathcal{S}_{<t}} \cos(\hat{Z}_i, \hat{Z}_j)$$

**優點：**
- 只要有一個已選 patch 在某個區域，後續就不會再去那個區域
- 對多病灶、多模態組織最有效
- 實現簡單，增量更新 $O(N \cdot k)$

**臨床意義：** 像一個有系統的病理科醫師，確認完一個區域後立刻移去下一個未探索的區域。

---

### 方法二：Centroid — 已實作，較弱 baseline

**核心機制：** 懲罰「和所有已選 patch 的平均向量最像的」patch

$$\text{adj}_i^{(t)} = s_i - \lambda \cdot \cos\!\left(\hat{Z}_i,\; \frac{1}{|\mathcal{S}_{<t}|}\sum_{j \in \mathcal{S}_{<t}} \hat{Z}_j\right)$$

**限制：**
- 如果已選 patch 分布在兩個截然不同的群組，平均向量可能落在「虛空中」（既不像群組A也不像群組B）
- 懲罰訊號被稀釋，後期幾乎失去方向感

**預期效果：** < MMR，尤其在 patch 分布多群組的情況下

---

### 方法三：Coverage（特徵空間覆蓋率）— 未來方向

**概念：** 不用 cosine similarity 度量距離，改用「特徵空間的覆蓋幾何」

每個已選 patch 在特徵空間覆蓋半徑為 $r$ 的球體。
下一輪優先選落在所有球體外面的 patch（未被覆蓋的區域）。

**潛力：**
- 對稀疏病灶（腫瘤 patch 極少）更公平——不會因為一個小腫瘤的所有 patch 都被一個 MMR round 搞定而忽略更遠的稀有 patch
- 能精確量化「已探索多少比例的特徵空間」
- 在 batch-level coverage 的 continual learning 設定中，未來可以用 coverage 來保證每個任務的「特徵空間探索完整性」

**和 MMR 的差異：**
MMR 是「不要和某張 patch 太像」（點對點）。
Coverage 是「不要落在已探索的特徵球體內」（點對幾何集合）。

**為什麼可能優於 MMR：** 當 round 1 選出的 16 個 patch 高度集中時，它們的「覆蓋球」幾乎重疊，MMR 可能高估了這個區域的探索程度。Coverage 則精確計算真正未覆蓋的區域大小。

---

### 方法四：DPP（Determinantal Point Process）— 未來方向

**概念：** 不貪婪，一次性求解「全局最優的多樣性子集」

$$P(\mathcal{S}) \propto \det(L_\mathcal{S}), \quad L_{ij} = s_i \cdot \cos(\hat{Z}_i, \hat{Z}_j) \cdot s_j$$

矩陣行列式越大 = 子集越多樣（向量空間跨度越大）且 score 越高。

**DPP 的直覺：** 想像所有 patch 是空間中的向量。行列式代表這些向量「張開的空間體積」——選出的 patch 越不一樣、越分散，行列式越大。DPP 直接最大化這個體積，同時考慮每個 patch 的品質（$s_i$）。

**潛力：**
- 全局最優，不是每步貪婪近似
- 在 budget 小（K=16, 32）時，DPP 的優勢比 MMR 更顯著
- 可以精確控制「多樣性 vs 品質」的數學 tradeoff
- 未來在 continual learning 中，DPP 可以保證每個任務的 router 學習到的 patch 子集，最大化跨任務的特徵空間覆蓋

**缺點：**
- 精確求解是 $O(N^3)$（矩陣行列式）
- N=3000 時需要近似（Nyström DPP、隨機採樣 DPP）
- 預期提升 0.5%～1%，但計算成本高 10 倍以上

**和 MMR 的差異：**

| | MMR | DPP |
|---|---|---|
| 最優性 | 貪婪逐輪近似 | 全局最優（近似版也接近全局） |
| 速度 | $O(NK)$ 快 | $O(N^2K)$ 慢 |
| 對 K 的敏感性 | K 大時貪婪累積誤差較小 | K 小時優勢最大 |
| 實作難度 | 5 行程式碼 | 需要矩陣分解庫 |

---

## 8. 為什麼不需要重訓就能換選取策略？

**核心原理：** TextNavRouter 只負責輸出 $s_i$（靜態 importance score）。訓練時 Router 完全不知道有 SBO 的存在——Loss 是基於 soft top-K aggregation 算的，和 MMR/centroid/DPP 無關。

```
訓練時：  Z → Router → s_i → soft top-K → classify → CE Loss
推論時：  Z → Router → s_i → [MMR/centroid/coverage/DPP] → 64 patches → classify
```

Router 學的是「哪張 patch 本身重要」。
選取策略學的是「在已有重要性排名的基礎上，如何挑出多樣的 64 張」。

**兩者獨立**，互不影響訓練。這正是「backbone-agnostic」設計的彈性所在。

---

## 9. 總結對比表

| 方法 | 懲罰依據 | 最優性 | 速度 | 需要重訓？ | 預期效果 | 臨床直覺 |
|---|---|---|---|---|---|---|
| **Top-64**（baseline）| 無 | 非最優 | 最快 | 否 | ★★★ | 直接看最可疑的地方 |
| **Centroid** | 與平均向量相似度 | 貪婪 | 快 | 否 | ★★★ | 避開「重心」附近 |
| **MMR**（預設） | 與任一已選 patch 最大相似度 | 貪婪 | 快 | 否 | ★★★★ | 系統性探索新區域 |
| **Coverage**（未來） | 特徵空間覆蓋幾何 | 貪婪 | 中 | 否 | ★★★★ | 確保探索每個「角落」 |
| **DPP**（未來） | 行列式最大化（全局） | 全局最優 | 慢 | 否 | ★★★★★ | 數學上最優的多樣性 |

> 所有方法都**不需要重訓 Router**。

---

*ZeroNav — SBO 方法完整說明*
*Last updated: 2026-07-02*
