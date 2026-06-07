"""Attention-based Neural Operator（Galerkin 注意力神经算子）。

使用线性 Galerkin Attention：O = Q · (K^T V) / N
复杂度 O(N·d²)，而非标准 softmax attention 的 O(N²·d)。

来自 SRNO / HiNOTE 的设计。
"""

import torch
import torch.nn as nn


class GalerkinAttention(nn.Module):
    """单头 Galerkin 线性注意力。"""

    def __init__(self, dim: int, hidden_mult: int = 2):
        """
        Args:
            dim: token 维度
            hidden_mult: Q/K 投影的中间维度倍数
        """
        super().__init__()
        inner_dim = dim * hidden_mult
        self.q_proj = nn.Linear(dim, inner_dim)
        self.k_proj = nn.Linear(dim, inner_dim)
        self.v_proj = nn.Linear(dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape

        Q = self.q_proj(x)  # (B, L, inner_dim)
        K = self.k_proj(x)  # (B, L, inner_dim)
        V = self.v_proj(x)  # (B, L, inner_dim)

        # Galerkin: (K^T @ V) first, then Q @ (K^T V)
        KV = K.transpose(-1, -2) @ V  # (B, inner_dim, inner_dim)
        out = Q @ KV / L                # (B, L, inner_dim)

        return self.out_proj(out)       # (B, L, D)


class AttnO(nn.Module):
    """Attention-based Neural Operator。

    Galerkin Attention + 非线性 + 残差，共享于 PiT 内部。
    """

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = GalerkinAttention(dim, hidden_mult)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # Sub-block 1: Galerkin Attention + residual
        x = x + self.attn(self.norm(x))

        # Sub-block 2: MLP + residual
        x = x + self.mlp(self.norm2(x))

        return x
