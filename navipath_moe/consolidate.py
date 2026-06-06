"""Replay-free dual-importance expert consolidation (研究計劃 §5.5).

完全不儲存任何過去切片 / patch。每個任務結束後：
  I_cur_e = 該任務中 expert e 的平均路由使用度（已正規化）
  I_old_e = EMA(歷史各任務的 I_cur_e)
  m_e     = sigmoid(a * I_old_e - b * I_cur_e)   # 保護程度（越大越凍結）
  theta_e <- m_e * theta_e^old + (1 - m_e) * phi_e^new

直覺：
  舊任務重要、新任務不重要 -> m_e 大 -> 保護（近凍結）
  舊任務不重要、新任務重要 -> m_e 小 -> 放開更新
"""
from __future__ import annotations

import copy
import torch


class ExpertImportance:
    """追蹤每個 expert 的歷史 (I_old) 與當前任務 (I_cur) 重要度。"""

    def __init__(self, num_experts: int, ema_decay: float = 0.7):
        self.E = num_experts
        self.ema = ema_decay
        self.I_old = torch.zeros(num_experts)
        self._acc = torch.zeros(num_experts)       # 當前任務累加
        self._count = 0

    def reset_current(self):
        self._acc = torch.zeros(self.E)
        self._count = 0

    @torch.no_grad()
    def observe(self, w: torch.Tensor):
        """w:[n,E] 一個 batch / 一張切片的路由權重。累加到當前任務統計。"""
        self._acc += w.detach().float().mean(dim=0).cpu()
        self._count += 1

    def current(self) -> torch.Tensor:
        """I_cur，正規化成分布。"""
        if self._count == 0:
            return torch.ones(self.E) / self.E
        cur = self._acc / self._count
        return cur / (cur.sum() + 1e-8)

    def end_task(self) -> torch.Tensor:
        """任務結束：回傳本任務 I_cur，並把它 EMA 進 I_old。"""
        cur = self.current()
        self.I_old = self.ema * self.I_old + (1 - self.ema) * cur
        self.reset_current()
        return cur


@torch.no_grad()
def consolidate(expert_bank, importance: ExpertImportance,
                old_state: list[dict], a: float = 4.0, b: float = 2.0):
    """把 expert 參數朝舊參數方向回拉，保護程度由 m_e 決定。

    expert_bank : ExpertBank（已含本任務訓練後的新參數 phi）
    old_state   : 上一任務結束時各 expert 的 state_dict 快照（list 長度 E）
    回傳 m: [E] 供 log / 視覺化。
    """
    I_old = importance.I_old
    I_cur = importance.current()
    m = torch.sigmoid(a * I_old - b * I_cur)        # [E]

    for e in range(expert_bank.num_experts):
        new_sd = expert_bank.experts[e].state_dict()
        old_sd = old_state[e]
        me = float(m[e])
        merged = {k: me * old_sd[k] + (1 - me) * new_sd[k] for k in new_sd}
        expert_bank.experts[e].load_state_dict(merged)
    return m


def snapshot_experts(expert_bank) -> list[dict]:
    """深拷貝目前各 expert 的 state_dict（任務開始/結束時呼叫）。"""
    return [copy.deepcopy(expert_bank.experts[e].state_dict())
            for e in range(expert_bank.num_experts)]
