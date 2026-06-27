"""SPEC-02 — Mechanism-selection table for continual WSI navigation.

掃 outputs/oldtask_budget_{order}_f{fold}_task{t}{suffix}.json，依機制
(shared / ewc / pertask) × budget 聚合 router 在「舊任務」上的 acc（跨 fold mean），
輸出 STORYLINE §4 的機制選擇主表。純 stdlib，無外部相依。

用法:
  python tools/mechanism_table.py --outputs outputs -o outputs/MECHANISM_SELECTION.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

# 檔名後綴 -> 機制顯示名
SUFFIX_TO_MECH = {"": "shared", "__ewc": "ewc", "__pertask": "pertask"}
MECH_ORDER = ["shared", "ewc", "pertask"]
FNAME_RE = re.compile(
    r"oldtask_budget_(?P<order>\w+?)_f(?P<fold>\d+)_task(?P<task>\d+)(?P<suffix>__\w+)?\.json$"
)


def _budget_keys(budgets):
    return ["All" if b == 0 else str(b) for b in budgets]


def load_records(outputs_dir):
    """回傳 dict[(order, eval_task)] -> dict[mech] -> list[(fold, results, budgets)]"""
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(outputs_dir, "oldtask_budget_*.json"))):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        suffix = m.group("suffix") or ""
        mech = SUFFIX_TO_MECH.get(suffix)
        if mech is None:
            continue
        with open(path) as f:
            d = json.load(f)
        key = (d["order"], d.get("eval_task", f"task{m.group('task')}"))
        groups[key][mech].append((int(m.group("fold")), d["results"], d["budgets"]))
    return groups


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(x):
    return f"{x:.3f}" if x is not None else "—"


def mech_row(records, bkeys):
    """records: list[(fold, results, budgets)] -> {bkey: mean router acc}, n_folds, go_folds"""
    per_budget = defaultdict(list)
    go_folds = 0
    for _fold, results, _budgets in records:
        for bk in bkeys:
            per_budget[bk].append(results.get("router", {}).get(bk))
        # GO 判定：router@64 > best heuristic@64
        b64 = "64"
        r = results.get("router", {}).get(b64)
        heur = [results.get(h, {}).get(b64) for h in ("random", "prototype", "semantic")]
        heur = [h for h in heur if h is not None]
        if r is not None and heur and r > max(heur):
            go_folds += 1
    return {bk: _mean(per_budget[bk]) for bk in bkeys}, len(records), go_folds


def best_heur_row(all_records, bkeys):
    """跨所有機制共享的 heuristic（用任一機制檔內的 heuristic；取 shared 優先）。"""
    # heuristic 不隨機制變，挑 fold 數最多的一組
    recs = max(all_records.values(), key=len) if all_records else []
    per_budget = defaultdict(list)
    for _fold, results, _budgets in recs:
        for bk in bkeys:
            vals = [results.get(h, {}).get(bk) for h in ("random", "prototype", "semantic")]
            vals = [v for v in vals if v is not None]
            if vals:
                per_budget[bk].append(max(vals))
    return {bk: _mean(per_budget[bk]) for bk in bkeys}


def render(groups):
    lines = ["# Mechanism Selection for Continual WSI Navigation",
             "",
             "> 自動產出（`tools/mechanism_table.py`）。值 = router 在**舊任務**上的 acc，跨 fold mean。",
             "> 對應 STORYLINE §4 / ADR-0003。best-heuristic = max(random, prototype, semantic)。",
             ""]
    if not groups:
        lines.append("(no data found)")
        return "\n".join(lines) + "\n"

    for (order, eval_task), by_mech in sorted(groups.items()):
        # 用任一機制的 budgets 當欄位
        any_rec = next(iter(by_mech.values()))
        budgets = any_rec[0][2]
        bkeys = _budget_keys(budgets)
        lines.append(f"## order={order}, old task={eval_task}")
        lines.append("")
        header = "| Mechanism | " + " | ".join(bkeys) + " | n_folds | GO |"
        sep = "|" + "---|" * (len(bkeys) + 3)
        lines.append(header)
        lines.append(sep)
        for mech in MECH_ORDER:
            if mech not in by_mech:
                lines.append(f"| {mech} | " + " | ".join("(no data)" for _ in bkeys) + " | 0 | — |")
                continue
            row, n, go = mech_row(by_mech[mech], bkeys)
            cells = " | ".join(_fmt(row[bk]) for bk in bkeys)
            lines.append(f"| {mech} | {cells} | {n} | {go}/{n} |")
        # best-heuristic 參考列
        bh = best_heur_row(by_mech, bkeys)
        cells = " | ".join(_fmt(bh[bk]) for bk in bkeys)
        lines.append(f"| best-heuristic | {cells} | — | — |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("-o", "--out", default="outputs/MECHANISM_SELECTION.md")
    args = ap.parse_args()
    groups = load_records(args.outputs)
    md = render(groups)
    with open(args.out, "w") as f:
        f.write(md)
    print(md)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
