"""N3 — 彙整 N2 的 seqobs JSON，產出 retention 表 / CL 指標 / zero-shot 對比 + 圖。

讀 outputs/seqobs_reverse_f{F}_task{T}[_policy-zeroshot].json（F=1..3, T=0..3）。
任務順序 reverse：0=esca(最舊) 1=rcc 2=brca 3=lung(最新)。

定義（oracle gate）：
  nsm   = 用「剛學完該任務的 router」(skill_bank[T]) -> 近似 a[T][T]，無遺忘上界 / 我們的方法
  nonsm = 用「學完所有任務後的最終 router」      -> a[last][T]，naive continual（會忘舊）
  zero  = 不訓練，frozen-FM 文字相似度選 patch（天生零遺忘 baseline）

CL 指標（對 budget B、用 seq 模式）：
  mACC      = mean_T acc[T]
  Forgetting= mean_{T<last} (nsm[T] - nonsm[T])     （越小越好；nonsm 相對自身剛學完掉多少）
  BWT       = -Forgetting

用法：python analyze_seqobs_n3.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

OUT = "outputs"
FIGS = "site/figs"
FOLDS = (1, 2, 3)
TASKS = (0, 1, 2, 3)
TASK_NAME = {0: "esca", 1: "rcc", 2: "brca", 3: "lung"}
BUDGET_KEYS = ("All", "128", "64", "32", "16")
HEADLINE_B = "64"


def load(fold, task, zero=False):
    tag = "_policy-zeroshot" if zero else ""
    path = os.path.join(OUT, f"seqobs_reverse_f{fold}_task{task}{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["results"]


def acc(results, mode, b):
    """results[mode][b]；缺值回 nan。"""
    if results is None or mode not in results:
        return float("nan")
    return float(results[mode].get(b, "nan"))


def collect(mode, b, zero=False):
    """回傳 shape [n_fold, n_task] 的 acc 陣列。"""
    arr = np.full((len(FOLDS), len(TASKS)), np.nan)
    for fi, F in enumerate(FOLDS):
        for ti, T in enumerate(TASKS):
            arr[fi, ti] = acc(load(F, T, zero), mode, b)
    return arr


def ms(x):
    """mean±std 字串（忽略 nan）。"""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return "[MISSING]"
    return f"{x.mean():.3f}±{x.std():.3f}"


def main():
    os.makedirs(FIGS, exist_ok=True)
    lines = []
    P = lines.append
    P(f"# N3 結果彙整 — Sequential Budgeted Observation (reverse, folds {list(FOLDS)})")
    P("")
    P(f"> 自動產出：`analyze_seqobs_n3.py`，{datetime.now():%Y-%m-%d %H:%M}。")
    P("> 任務序：0=esca(最舊)→1=rcc→2=brca→3=lung(最新)。oracle gate。seq 模式為主。")
    P("> nsm=每任務專屬 skill（我們/無遺忘上界）；nonsm=最終單一 router（naive continual，會忘）；")
    P("> zero=zero-shot navigator（不訓練、frozen-FM 文字相似度）。")
    P("")

    # ── 1. headline retention 表（budget=64, seq）──────────────────────────
    nsm = collect("nsm_seq", HEADLINE_B)
    nonsm = collect("nonsm_seq", HEADLINE_B)
    zero = collect("zeroshot_seq", HEADLINE_B, zero=True)
    P(f"## 1. Retention（budget={HEADLINE_B}, seq）— acc mean±std over {len(FOLDS)} folds")
    P("")
    P("| 任務 | nsm（我們） | nonsm（會忘） | zero-shot | nsm−nonsm（遺忘修復） |")
    P("|---|---|---|---|---|")
    for ti, T in enumerate(TASKS):
        gap = nsm[:, ti] - nonsm[:, ti]
        P(f"| {T} {TASK_NAME[T]} | {ms(nsm[:,ti])} | {ms(nonsm[:,ti])} | "
          f"{ms(zero[:,ti])} | {ms(gap)} |")
    P("")

    # ── 2. CL 指標（budget=64, seq）────────────────────────────────────────
    old = [0, 1, 2]  # 舊任務（非最後一個）
    macc_nsm = np.nanmean(nsm, axis=1)
    macc_nonsm = np.nanmean(nonsm, axis=1)
    macc_zero = np.nanmean(zero, axis=1)
    forget = np.nanmean((nsm - nonsm)[:, old], axis=1)   # per fold
    P(f"## 2. CL 指標（budget={HEADLINE_B}, seq）")
    P("")
    P("| 指標 | nsm（我們） | nonsm（baseline） | zero-shot |")
    P("|---|---|---|---|")
    P(f"| mACC（4 任務平均，越高越好） | {ms(macc_nsm)} | {ms(macc_nonsm)} | {ms(macc_zero)} |")
    P(f"| Forgetting（舊任務，越小越好） | {ms(np.zeros_like(forget))} | {ms(forget)} | {ms(np.zeros_like(forget))} |")
    P(f"| BWT（越接近 0/正越好） | {ms(np.zeros_like(forget))} | {ms(-forget)} | {ms(np.zeros_like(forget))} |")
    P("")
    P("> nsm/zero 天生零遺忘（各任務用自己的 skill / 不訓練）；nonsm 的 Forgetting 即「naive 連續訓練 router」對舊任務的衰退。")
    P("")

    # ── 3. esca（最舊任務）budget 曲線數字 ────────────────────────────────
    P("## 3. 最舊任務 esca 的 budget 曲線（mean±std, seq）")
    P("")
    P("| budget | nsm（我們） | nonsm（會忘） | zero-shot |")
    P("|---|---|---|---|")
    for b in BUDGET_KEYS:
        n = collect("nsm_seq", b)[:, 0]
        nn = collect("nonsm_seq", b)[:, 0]
        z = collect("zeroshot_seq", b, zero=True)[:, 0]
        P(f"| {b} | {ms(n)} | {ms(nn)} | {ms(z)} |")
    P("")

    # ── 4. seq vs one-shot（檢查 agentic 差異）────────────────────────────
    P("## 4. sequential vs one-shot（nsm, budget=64）")
    P("")
    seqv = collect("nsm_seq", HEADLINE_B)
    onev = collect("nsm_oneshot", HEADLINE_B)
    P("| 任務 | nsm_seq | nsm_oneshot | 差 |")
    P("|---|---|---|---|")
    for ti, T in enumerate(TASKS):
        P(f"| {T} {TASK_NAME[T]} | {ms(seqv[:,ti])} | {ms(onev[:,ti])} | {ms(seqv[:,ti]-onev[:,ti])} |")
    P("")
    P("> seq≈one-shot 屬預期（budget≥step 時單輪、redundancy 影響小）；agentic 增益待調參，見 wiki 07 §7。")
    P("")

    md_path = os.path.join(OUT, f"RESULTS_seqobs_{datetime.now():%Y%m%d}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[N3] wrote {md_path}")

    # ── 圖 ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[N3] matplotlib 不可用，跳過圖：{e}")
        return

    # 圖 A：esca budget 曲線（nsm vs nonsm vs zero）
    xs = [16, 32, 64, 128]
    def curve(mode, zero=False):
        return [np.nanmean(collect(mode, str(b), zero)[:, 0]) for b in xs]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, curve("nsm_seq"), "o-", label="continual (NSM, ours)", color="#15695a")
    plt.plot(xs, curve("nonsm_seq"), "s--", label="naive continual (forgets)", color="#b03a3a")
    plt.plot(xs, curve("zeroshot_seq", zero=True), "^:", label="zero-shot navigator", color="#7a5bb0")
    plt.axhline(np.nanmean(collect("nsm_seq", "All")[:, 0]), color="#999", ls=":", lw=1, label="acc@All")
    plt.xlabel("budget K (patches)"); plt.ylabel("acc (esca, oldest task)")
    plt.title("Navigation retention on oldest task (esca)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    a = os.path.join(FIGS, "n3_esca_budget_curve.png")
    plt.savefig(a, dpi=140); plt.close()
    print(f"[N3] wrote {a}")

    # 圖 B：retention bar（budget=64，每任務 nsm vs nonsm vs zero）
    x = np.arange(len(TASKS)); w = 0.26
    plt.figure(figsize=(7, 4))
    plt.bar(x - w, np.nanmean(nsm, axis=0), w, label="continual (NSM, ours)", color="#15695a")
    plt.bar(x, np.nanmean(nonsm, axis=0), w, label="naive continual", color="#b03a3a")
    plt.bar(x + w, np.nanmean(zero, axis=0), w, label="zero-shot", color="#7a5bb0")
    plt.xticks(x, [f"{T}:{TASK_NAME[T]}" for T in TASKS])
    plt.ylabel(f"acc @ budget {HEADLINE_B}"); plt.ylim(0, 1)
    plt.title("Per-task retention after learning all tasks")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    b = os.path.join(FIGS, "n3_retention_bar.png")
    plt.savefig(b, dpi=140); plt.close()
    print(f"[N3] wrote {b}")


if __name__ == "__main__":
    main()
