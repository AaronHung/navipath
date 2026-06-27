# Mechanism Selection for Continual WSI Navigation

> 自動產出（`tools/mechanism_table.py`）。值 = router 在**舊任務**上的 acc，跨 fold mean。
> 對應 STORYLINE §4 / ADR-0003。best-heuristic = max(random, prototype, semantic)。

## order=paper, old task=tcga_lung

| Mechanism | All | 256 | 128 | 64 | 32 | n_folds | GO |
|---|---|---|---|---|---|---|---|
| shared | 0.764 | 0.512 | 0.453 | 0.397 | 0.353 | 3 | 0/3 |
| ewc | (no data) | (no data) | (no data) | (no data) | (no data) | 0 | — |
| pertask | (no data) | (no data) | (no data) | (no data) | (no data) | 0 | — |
| best-heuristic | 0.764 | 0.813 | 0.793 | 0.813 | 0.816 | — | — |

## order=reverse, old task=tcga_esca

| Mechanism | All | 256 | 128 | 64 | 32 | n_folds | GO |
|---|---|---|---|---|---|---|---|
| shared | 0.867 | 0.511 | 0.400 | 0.333 | 0.333 | 3 | 0/3 |
| ewc | 0.867 | 0.622 | 0.422 | 0.400 | 0.267 | 3 | 0/3 |
| pertask | 0.867 | 0.933 | 0.933 | 0.933 | 0.933 | 3 | 3/3 |
| best-heuristic | 0.867 | 0.867 | 0.822 | 0.844 | 0.800 | — | — |

