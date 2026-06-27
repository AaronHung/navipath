# NaviPath-MoE — Collected Results

_scanned `outputs` — 28 result files_

## Inventory (files found)

| Kind | Order | folds present |
|---|---|---|
| navipath_full | paper | 1,2,3 |
| navipath_full | reverse | 1,2,3 |
| navipath_micro | paper | 1 |
| navipath_micro | reverse | 1 |
| oldtask_budget | paper | 1,2,3 |
| oldtask_budget | reverse | 1,2,3 |
| qpmil | paper | 1,2,3 |
| qpmil | reverse | 1,2,3 |
| router_v0 | paper | 1,2,3 |
| router_v0 | reverse | 1,2,3 |
| routing_drift | paper | 1 |
| routing_drift | reverse | 1 |

## Accuracy / Forgetting (mean±std across folds)

| Method | Order | n | ACC | Forgetting | BWT | UpperBoundRatio |
|---|---|---|---|---|---|---|
| QPMIL baseline | paper | 3 | 0.924±0.016 | 0.017±0.022 | -0.017±0.022 | 1.018±0.018 |
| QPMIL baseline | reverse | 3 | 0.917±0.026 | 0.041±0.023 | -0.041±0.023 | — |
| NaviPath (full) | paper | 3 | 0.879±0.030 | 0.000±0.000 | 0.000±0.000 | 0.968±0.033 |
| NaviPath (full) | reverse | 3 | 0.886±0.030 | 0.000±0.000 | 0.000±0.000 | 0.976±0.033 |
| NaviPath (micro) | paper | 1 | 0.857±0.000 | 0.000±0.000 | 0.000±0.000 | 0.943±0.000 |
| NaviPath (micro) | reverse | 1 | 0.852±0.000 | 0.000±0.000 | 0.000±0.000 | 0.939±0.000 |

## Router patch-budget on LAST task (router_v0)

Router vs heuristics, accuracy at each patch budget (mean across folds). Δ>0 at a finite budget = router selects more informative patches than the best heuristic. GO column flags those budgets.

### order=paper, eval_task=tcga_esca (n=3 folds) — GO=3/3 folds

| Budget | router | random | prototype | semantic | Δ(router−best heur) | GO |
|---|---|---|---|---|---|
| All | 0.933 | 0.933 | 0.933 | 0.933 | +0.000 |  |
| 256 | 0.956 | 0.956 | 0.844 | 0.889 | +0.000 |  |
| 128 | 0.956 | 0.889 | 0.756 | 0.822 | +0.067 | ✓ |
| 64 | 0.956 | 0.889 | 0.711 | 0.867 | +0.067 | ✓ |
| 32 | 0.933 | 0.756 | 0.667 | 0.867 | +0.067 | ✓ |

### order=reverse, eval_task=tcga_lung (n=3 folds) — GO=3/3 folds

| Budget | router | random | prototype | semantic | Δ(router−best heur) | GO |
|---|---|---|---|---|---|
| All | 0.892 | 0.892 | 0.892 | 0.892 | +0.000 |  |
| 256 | 0.904 | 0.908 | 0.867 | 0.901 | -0.004 |  |
| 128 | 0.915 | 0.885 | 0.874 | 0.904 | +0.010 | ✓ |
| 64 | 0.922 | 0.882 | 0.831 | 0.897 | +0.025 | ✓ |
| 32 | 0.918 | 0.814 | 0.814 | 0.891 | +0.026 | ✓ |


## OLD-task patch-budget (生死表 / oldtask_budget)

The key COMPAYL claim: under a tight patch budget on PREVIOUS tasks, does the router still beat random/prototype/semantic? Δ>0 = yes.

### order=paper, eval_task=tcga_lung (n=3 folds) — GO=0/3 folds

| Budget | router | random | prototype | semantic | Δ(router−best heur) | GO |
|---|---|---|---|---|---|
| All | 0.764 | 0.764 | 0.764 | 0.764 | +0.000 |  |
| 256 | 0.512 | 0.767 | 0.764 | 0.810 | -0.298 |  |
| 128 | 0.453 | 0.771 | 0.718 | 0.793 | -0.341 |  |
| 64 | 0.397 | 0.783 | 0.705 | 0.813 | -0.417 |  |
| 32 | 0.353 | 0.773 | 0.637 | 0.816 | -0.463 |  |

### order=reverse, eval_task=tcga_esca (n=3 folds) — GO=0/3 folds

| Budget | router | random | prototype | semantic | Δ(router−best heur) | GO |
|---|---|---|---|---|---|
| All | 0.867 | 0.867 | 0.867 | 0.867 | +0.000 |  |
| 256 | 0.511 | 0.844 | 0.822 | 0.778 | -0.333 |  |
| 128 | 0.400 | 0.689 | 0.800 | 0.800 | -0.400 |  |
| 64 | 0.333 | 0.822 | 0.778 | 0.778 | -0.489 |  |
| 32 | 0.333 | 0.600 | 0.689 | 0.756 | -0.422 |  |

