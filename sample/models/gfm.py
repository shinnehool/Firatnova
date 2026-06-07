"""Gated Fusion Mechanism（门控融合机制）。

来自 DiffFNO 的设计：
  G = σ(Conv1x1([v_a, v_w]))
  v_fused = G ⊙ v_a + (1 - G) ⊙ v_w

自适应融合 Attention 分支（空间局部）和 WFNO 分支（频域全局）。
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """门控融合：动态平衡两个分支的特征。"""

    def __init__(self, dim: int, hidden_mult: int = 1):
        """
        Args:
            dim: 特征维度
            hidden_mult: gate 投影的隐藏层倍数
        """
        super().__init__()
        inner_dim = dim * hidden_mult if hidden_mult > 0 else dim
        self.gate_conv = nn.Sequential(
            nn.Conv1d(dim * 2, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(inner_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, v_a: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_a: Attention 分支输出 (B, L, D)
            v_w: WFNO 分支输出 (B, L, D)
        Returns:
            融合特征 (B, L, D)
        """
        # concat 后做 1D 卷积（沿通道维）
        cat = torch.cat([v_a, v_w], dim=-1)           # (B, L, 2D)
        cat = cat.transpose(-1, -2)                    # (B, 2D, L)
        G = self.gate_conv(cat).transpose(-1, -2)      # (B, L, 1)

        return G * v_a + (1.0 - G) * v_w
