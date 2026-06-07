"""DiT Block：MHSA ∥ WFNO → Gate → MLP（含两层残差）。

条件注入方式：全局 AdaLN（所有 patch token 共享同一组 scale/shift/gate）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wfno import WFNO
from .gfm import GatedFusion


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """AdaLN 调制：x * (1 + scale) + shift。"""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT Block。

    结构：
      Sub-block 1: RMSNorm → AdaLN → MHSA ∥ WFNO → Gate → 残差
      Sub-block 2: RMSNorm → AdaLN → MLP → 残差
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 wfno_modes: int = 12, wfno_alpha: float = 0.7,
                 gate_hidden_mult: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # ---- Sub-block 1 components ----
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        self.wfno = WFNO(hidden_dim, modes=wfno_modes, alpha=wfno_alpha)
        self.gate1 = GatedFusion(hidden_dim, hidden_mult=gate_hidden_mult)

        # ---- Sub-block 2 components ----
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ---- AdaLN 投影（从 c 到 scale/shift/gate）----
        # Sub-block 1 的参数
        self.adaLN_proj1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),  # scale, shift, gate
        )
        # Sub-block 2 的参数
        self.adaLN_proj2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )

        # AdaLN-Zero 初始化：scale/shift 置零，gate 置小正值
        nn.init.constant_(self.adaLN_proj1[-1].weight, 0)
        nn.init.constant_(self.adaLN_proj1[-1].bias, 0)
        nn.init.constant_(self.adaLN_proj2[-1].weight, 0)
        nn.init.constant_(self.adaLN_proj2[-1].bias, 0)
        # Gate bias 初始化为 1.0，完全激活 Attention/WFNO 通路
        gate_bias_start = 2 * hidden_dim
        nn.init.constant_(self.adaLN_proj1[-1].bias[gate_bias_start:], 1.0)
        nn.init.constant_(self.adaLN_proj2[-1].bias[gate_bias_start:], 1.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) patch tokens
            c: (B, D) 全局条件向量
        Returns:
            (B, L, D)
        """
        # ---- Sub-block 1: MHSA ∥ WFNO → Gate ----
        # AdaLN params
        params1 = self.adaLN_proj1(c)                                   # (B, 3D)
        scale1, shift1, gate_alpha1 = params1.chunk(3, dim=-1)         # each (B, D)

        x_mod = modulate(self.norm1(x), scale1, shift1)

        # MHSA
        v_attn, _ = self.attn(x_mod, x_mod, x_mod)
        # WFNO: 先 reshape 到 2D
        B, L, D = x_mod.shape
        h = int(L ** 0.5)
        w = L // h
        if h * w != L:
            raise ValueError(f"Cannot reshape {L} tokens to 2D grid (h={h}, w={w})")
        x_2d = x_mod.reshape(B, h, w, D)
        v_wfno = self.wfno(x_2d).reshape(B, L, D)

        # Gated Fusion
        v_fused = self.gate1(v_attn, v_wfno)
        x = x + gate_alpha1.unsqueeze(1) * v_fused                    # 残差1

        # ---- Sub-block 2: MLP ----
        params2 = self.adaLN_proj2(c)                                  # (B, 3D)
        scale2, shift2, gate_alpha2 = params2.chunk(3, dim=-1)         # each (B, D)

        x_mod2 = modulate(self.norm2(x), scale2, shift2)
        v_mlp = self.mlp(x_mod2)
        x = x + gate_alpha2.unsqueeze(1) * v_mlp                       # 残差2

        return x
