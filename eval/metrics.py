"""Continual-learning evaluation metrics (研究計劃 §7.5).

全部沿用 QPMIL 的定義以保持公平比較。核心輸入是 accuracy matrix R:
  R[t, i] = 在學完任務 t 之後，於任務 i 測試集上的準確率 (t >= i)，下三角有效。
T = 任務數。
"""
from __future__ import annotations

import numpy as np


def average_accuracy(R: np.ndarray) -> float:
    """ACC：學完最後一個任務後，對所有已學任務的平均準確率（R 最後一列平均）。"""
    T = R.shape[0]
    return float(R[T - 1, :T].mean())


def forgetting(R: np.ndarray) -> float:
    """Forgetting：每個舊任務 i 的 (歷史最佳 - 最終) 的平均（i < T-1）。"""
    T = R.shape[0]
    if T < 2:
        return 0.0
    fk = []
    for i in range(T - 1):
        best = R[i:T - 1, i].max()        # 學完任務 i..T-2 期間對 i 的最佳
        fk.append(best - R[T - 1, i])
    return float(np.mean(fk))


def backward_transfer(R: np.ndarray) -> float:
    """BWT：學完最後任務後，舊任務相對「剛學完時」的平均變化（通常為負）。"""
    T = R.shape[0]
    if T < 2:
        return 0.0
    bwt = [R[T - 1, i] - R[i, i] for i in range(T - 1)]
    return float(np.mean(bwt))


def upper_bound_ratio(acc: float, joint_train_acc: float) -> float:
    """ACC / JointTrain 上界。"""
    return float(acc / joint_train_acc) if joint_train_acc > 0 else 0.0


def summarize(R: np.ndarray, joint_train_acc: float | None = None) -> dict:
    acc = average_accuracy(R)
    out = {
        "ACC": round(acc, 4),
        "Forgetting": round(forgetting(R), 4),
        "BWT": round(backward_transfer(R), 4),
    }
    if joint_train_acc is not None:
        out["UpperBoundRatio"] = round(upper_bound_ratio(acc, joint_train_acc), 4)
    return out


# ---- patch-budget 導航評估 (研究計劃 主表 B / §7.2) -------------------------

def patch_budget_table(eval_fn, budgets=(0, 256, 128, 64, 32)) -> dict:
    """對每個 patch budget K 跑一次評估，回傳 {K: ACC}。

    eval_fn(k) -> 在「每張切片只用 router 選出的 Top-K patch」下的 ACC。
    k=0 代表全選（ACC@All）。實作時把 k 傳進 model.forward 的 top_k_select。
    """
    return {("All" if k == 0 else k): round(float(eval_fn(k)), 4) for k in budgets}


if __name__ == "__main__":
    # smoke test：4 任務的假 accuracy matrix
    R = np.array([
        [0.90, 0.0, 0.0, 0.0],
        [0.85, 0.88, 0.0, 0.0],
        [0.82, 0.84, 0.91, 0.0],
        [0.80, 0.83, 0.87, 0.89],
    ])
    print(summarize(R, joint_train_acc=0.908))
