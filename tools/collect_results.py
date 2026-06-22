#!/usr/bin/env python3
"""Collect NaviPath-MoE experiment JSONs into paper-ready summary tables.

Scans an outputs/ directory, aggregates across folds, and emits Markdown
tables (and optional CSV). No external dependencies — stdlib only.

Recognised JSON file patterns (written by the training / eval scripts):
  qpmil_{order}_fold{n}.json            -> CL baseline (QPMIL), has "summary"
  navipath_full_{order}_fold{n}.json    -> NaviPath full,        has "summary" (+ "budget")
  navipath_micro_{order}_fold{n}.json   -> NaviPath micro,       has "summary" (+ "budget")
  router_v0_{order}_fold{n}.json        -> router go/no-go on LAST task ("results","go")
  oldtask_budget_{order}_f{fold}_task{t}.json -> router budget on an OLD task
  routing_drift_{order}_fold{n}.json    -> per-task expert weights ("table")

Usage:
  python tools/collect_results.py                         # scan ./outputs, print markdown
  python tools/collect_results.py --outputs outputs       # explicit dir
  python tools/collect_results.py -o outputs/RESULTS_SUMMARY.md   # also write file
  python tools/collect_results.py --csv outputs/csv       # also dump per-table CSVs
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

# Heuristics compared against the router, in display order.
HEURISTICS = ["random", "prototype", "semantic"]
# GO threshold: router must beat the BEST heuristic by at least this margin
# at at least one finite budget to be flagged GO.
GO_MARGIN = 0.0

# ----------------------------------------------------------------------------
# filename parsing
# ----------------------------------------------------------------------------
_PATTERNS = [
    ("oldtask_budget", re.compile(r"^oldtask_budget_(?P<order>\w+?)_f(?P<fold>\d+)_task(?P<task>\d+)\.json$")),
    ("router_v0",      re.compile(r"^router_v0_(?P<order>\w+?)_fold(?P<fold>\d+)\.json$")),
    ("routing_drift",  re.compile(r"^routing_drift_(?P<order>\w+?)_fold(?P<fold>\d+)\.json$")),
    ("navipath_full",  re.compile(r"^navipath_full_(?P<order>\w+?)_fold(?P<fold>\d+)\.json$")),
    ("navipath_micro", re.compile(r"^navipath_micro_(?P<order>\w+?)_fold(?P<fold>\d+)\.json$")),
    ("qpmil",          re.compile(r"^qpmil_(?P<order>\w+?)_fold(?P<fold>\d+)\.json$")),
]


def classify(fname: str):
    """Return (kind, meta_dict) or (None, None) if unrecognised."""
    base = os.path.basename(fname)
    for kind, pat in _PATTERNS:
        m = pat.match(base)
        if m:
            return kind, m.groupdict()
    return None, None


# ----------------------------------------------------------------------------
# small stat helpers
# ----------------------------------------------------------------------------
def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def fmt_ms(m, s, pct=False):
    if m is None:
        return "—"
    if pct:
        return f"{m*100:.1f}±{s*100:.1f}"
    return f"{m:.3f}±{s:.3f}"


def budget_keys(results: dict):
    """Return finite budget keys (exclude 'All'/'0') sorted descending int."""
    keys = set()
    for arm in results.values():
        if isinstance(arm, dict):
            keys.update(arm.keys())
    fin = []
    for k in keys:
        if k in ("All", "0"):
            continue
        try:
            fin.append(int(k))
        except ValueError:
            pass
    return [str(x) for x in sorted(fin, reverse=True)]


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
def load_all(outputs_dir: str):
    records = []
    for path in sorted(glob.glob(os.path.join(outputs_dir, "*.json"))):
        kind, meta = classify(path)
        if kind is None:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] skip unreadable {os.path.basename(path)}: {e}")
            continue
        records.append({"kind": kind, "meta": meta, "data": data,
                        "file": os.path.basename(path)})
    return records


# ----------------------------------------------------------------------------
# table 1: accuracy / forgetting
# ----------------------------------------------------------------------------
def table_accuracy(records):
    # (method, order) -> list of summary dicts
    METHODS = [("qpmil", "QPMIL baseline"),
               ("navipath_full", "NaviPath (full)"),
               ("navipath_micro", "NaviPath (micro)")]
    agg = defaultdict(list)
    folds = defaultdict(set)
    for r in records:
        if r["kind"] not in {m for m, _ in METHODS}:
            continue
        summ = r["data"].get("summary")
        if summ:
            key = (r["kind"], r["meta"]["order"])
            agg[key].append(summ)
            folds[key].add(r["meta"]["fold"])

    lines = ["## Accuracy / Forgetting (mean±std across folds)", "",
             "| Method | Order | n | ACC | Forgetting | BWT | UpperBoundRatio |",
             "|---|---|---|---|---|---|---|"]
    rows_csv = [["method", "order", "n_folds", "ACC_mean", "ACC_std",
                 "Forgetting_mean", "Forgetting_std", "BWT_mean", "BWT_std"]]
    for method, label in METHODS:
        for order in ("paper", "reverse"):
            key = (method, order)
            if key not in agg:
                continue
            summ = agg[key]
            acc_m, acc_s = mean_std([s.get("ACC") for s in summ])
            fg_m, fg_s = mean_std([s.get("Forgetting") for s in summ])
            bw_m, bw_s = mean_std([s.get("BWT") for s in summ])
            ub_m, ub_s = mean_std([s.get("UpperBoundRatio") for s in summ])
            n = len(folds[key])
            lines.append(
                f"| {label} | {order} | {n} | {fmt_ms(acc_m, acc_s)} | "
                f"{fmt_ms(fg_m, fg_s)} | {fmt_ms(bw_m, bw_s)} | {fmt_ms(ub_m, ub_s)} |")
            rows_csv.append([method, order, n, acc_m, acc_s, fg_m, fg_s, bw_m, bw_s])
    return "\n".join(lines), rows_csv


# ----------------------------------------------------------------------------
# budget tables (router_v0 = last task; oldtask_budget = old tasks)
# ----------------------------------------------------------------------------
def _aggregate_budget(records, kind):
    """(order[,task]) -> {arm: {budget: [vals across folds]}}."""
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    folds = defaultdict(set)
    go_flags = defaultdict(list)
    for r in records:
        if r["kind"] != kind:
            continue
        res = r["data"].get("results")
        if not res:
            continue
        order = r["meta"]["order"]
        if kind == "oldtask_budget":
            eval_task = r["data"].get("eval_task", f"task{r['meta'].get('task','?')}")
            key = (order, eval_task)
        else:
            key = (order, r["data"].get("eval_task", "last"))
        folds[key].add(r["meta"]["fold"])
        if "go" in r["data"]:
            go_flags[key].append(bool(r["data"]["go"]))
        for arm, bud in res.items():
            if not isinstance(bud, dict):
                continue
            for b, v in bud.items():
                agg[key][arm][b].append(v)
    return agg, folds, go_flags


def _budget_section(title, note, agg, folds, go_flags):
    if not agg:
        return f"## {title}\n\n_(no data found)_\n", []
    lines = [f"## {title}", "", note, ""]
    rows_csv = [["order", "eval_task", "budget", "router", "random",
                 "prototype", "semantic", "router_minus_best_heuristic"]]
    for key in sorted(agg.keys()):
        order, eval_task = key
        results = agg[key]
        bkeys = budget_keys(results)
        n = len(folds[key])
        go_list = go_flags.get(key, [])
        go_txt = ""
        if go_list:
            go_txt = f" — GO={sum(go_list)}/{len(go_list)} folds"
        lines.append(f"### order={order}, eval_task={eval_task} (n={n} folds){go_txt}")
        lines.append("")
        header = "| Budget | router | " + " | ".join(HEURISTICS) + " | Δ(router−best heur) | GO |"
        lines.append(header)
        lines.append("|" + "---|" * (3 + len(HEURISTICS)))
        for b in ["All"] + bkeys:
            router_m, _ = mean_std(results.get("router", {}).get(b, []))
            heur_means = {h: mean_std(results.get(h, {}).get(b, []))[0] for h in HEURISTICS}
            best_heur = max([v for v in heur_means.values() if v is not None], default=None)
            if router_m is not None and best_heur is not None:
                delta = router_m - best_heur
                delta_txt = f"{delta:+.3f}"
                go = "✓" if (b != "All" and delta > GO_MARGIN) else ""
            else:
                delta_txt, go, delta = "—", "", None
            cells = [b, f"{router_m:.3f}" if router_m is not None else "—"]
            for h in HEURISTICS:
                hv = heur_means[h]
                cells.append(f"{hv:.3f}" if hv is not None else "—")
            cells += [delta_txt, go]
            lines.append("| " + " | ".join(cells) + " |")
            if b != "All":
                rows_csv.append([order, eval_task, b, router_m,
                                 heur_means["random"], heur_means["prototype"],
                                 heur_means["semantic"], delta])
        lines.append("")
    return "\n".join(lines), rows_csv


def table_router_budget(records):
    agg, folds, go = _aggregate_budget(records, "router_v0")
    return _budget_section(
        "Router patch-budget on LAST task (router_v0)",
        "Router vs heuristics, accuracy at each patch budget (mean across folds). "
        "Δ>0 at a finite budget = router selects more informative patches than the "
        "best heuristic. GO column flags those budgets.",
        agg, folds, go)


def table_oldtask_budget(records):
    agg, folds, go = _aggregate_budget(records, "oldtask_budget")
    return _budget_section(
        "OLD-task patch-budget (生死表 / oldtask_budget)",
        "The key COMPAYL claim: under a tight patch budget on PREVIOUS tasks, does the "
        "router still beat random/prototype/semantic? Δ>0 = yes.",
        agg, folds, go)


# ----------------------------------------------------------------------------
# inventory
# ----------------------------------------------------------------------------
def table_inventory(records):
    counts = defaultdict(lambda: defaultdict(set))
    for r in records:
        counts[r["kind"]][r["meta"]["order"]].add(r["meta"]["fold"])
    lines = ["## Inventory (files found)", "",
             "| Kind | Order | folds present |", "|---|---|---|"]
    for kind in sorted(counts):
        for order in sorted(counts[kind]):
            fs = ",".join(sorted(counts[kind][order], key=lambda x: int(x)))
            lines.append(f"| {kind} | {order} | {fs} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs", help="outputs dir to scan")
    ap.add_argument("-o", "--out", default="", help="write markdown to this file too")
    ap.add_argument("--csv", default="", help="dir to dump per-table CSVs")
    args = ap.parse_args()

    if not os.path.isdir(args.outputs):
        raise SystemExit(f"[error] outputs dir not found: {args.outputs}")

    records = load_all(args.outputs)
    if not records:
        raise SystemExit(f"[error] no recognised result JSONs in {args.outputs}")

    acc_md, acc_csv = table_accuracy(records)
    router_md, router_csv = table_router_budget(records)
    old_md, old_csv = table_oldtask_budget(records)
    inv_md = table_inventory(records)

    doc = "\n\n".join([
        "# NaviPath-MoE — Collected Results",
        f"_scanned `{args.outputs}` — {len(records)} result files_",
        inv_md, acc_md, router_md, old_md,
    ]) + "\n"

    print(doc)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(doc)
        print(f"[saved] {args.out}")

    if args.csv:
        os.makedirs(args.csv, exist_ok=True)
        for name, rows in [("accuracy", acc_csv),
                           ("router_budget", router_csv),
                           ("oldtask_budget", old_csv)]:
            p = os.path.join(args.csv, f"{name}.csv")
            with open(p, "w", newline="") as f:
                _csv.writer(f).writerows(rows)
            print(f"[saved] {p}")


if __name__ == "__main__":
    main()
