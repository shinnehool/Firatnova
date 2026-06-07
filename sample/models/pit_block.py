"""PiT Block：PixelTokenCompaction → AttnO ∥ WFNO → Gate → Expand → MLP（含两层残差）。

条件注入方式：Pixel-wise AdaLN（每个像素独立的 scale/shift/gate，由 Scond 生成）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attno import AttnO
from .wfno import WFNO
from .gfm import GatedFusion


class PixelTokenCompaction(nn.Module):
    """像素 Token 压缩：通过多级 strided Conv2d 将 H×W 像素压缩为 h×w compact token。"""

    def __init__(self, p: int, d_pix: int, d_cmp: int):
        super().__init__()
        self.p = p
        n_stages = int(math.log2(p))
        layers = []
        in_ch = d_pix
        for i in range(n_stages):
            out_ch = min(in_ch * 2, d_cmp) if i < n_stages - 1 else d_cmp
            layers.append(nn.Conv2d(in_ch, out_ch, 3, 2, 1))
            layers.append(nn.GELU())
            in_ch = out_ch
        self.compress = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, D_pix) 像素级特征
        Returns:
            (B, L, D_cmp) compacted tokens
        """
        x = x.permute(0, 3, 1, 2)          # (B, D_pix, H, W)
        x = self.compress(x)                # (B, D_cmp, h, w)
        B, C, h, w = x.shape
        x = x.reshape(B, C, h * w).transpose(1, 2)  # (B, L, D_cmp)
        return x


class PixelTokenExpand(nn.Module):
    """像素 Token 展开：通过多级 Upsample+Conv2d 将 compact token 还原为像素 token。"""

    def __init__(self, p: int, d_pix: int, d_cmp: int):
        super().__init__()
        self.p = p
        n_stages = int(math.log2(p))
        layers = []
        in_ch = d_cmp
        for i in range(n_stages):
            out_ch = max(in_ch // 2, d_pix) if i < n_stages - 1 else d_pix
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
            layers.append(nn.GELU())
            in_ch = out_ch
        self.expand = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: (B, L, D_cmp) compacted tokens
            h, w: patch 网格尺寸
        Returns:
            (B, H, W, D_pix) 展开后的像素特征
        """
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, h, w)  # (B, D_cmp, h, w)
        x = self.expand(x)                           # (B, D_pix, H, W)
        x = x.permute(0, 2, 3, 1)                    # (B, H, W, D_pix)
        return x


class PixelWiseAdaLN(nn.Module):
    """Pixel-wise AdaLN：从 Scond 为每个像素生成独立的调制参数。

    先用小型 MLP 将 Scond 映射为 per-patch 调制参数，再 bilinear upsample
    到像素分辨率。相比原来的 p² 膨胀投影，参数量降低 ~40× 且保证 patch 边界
    平滑连续。

    使用 AdaLN-Zero 初始化：最终投影层权重和偏置置零。
    """

    def __init__(self, d_semantic: int, d_pix: int, p: int, d_intermediate: int = 384):
        super().__init__()
        self.d_pix = d_pix
        self.p = p
        self.n_params = 6 * d_pix  # scale1, shift1, gate1, scale2, shift2, gate2

        self.proj = nn.Sequential(
            nn.Linear(d_semantic, d_intermediate),
            nn.SiLU(),
            nn.Linear(d_intermediate, self.n_params),
        )

        # AdaLN-Zero: 最终投影置零
        nn.init.constant_(self.proj[-1].weight, 0)
        nn.init.constant_(self.proj[-1].bias, 0)
        # Gate 初始化为 1.0（完全激活 Attention/WFNO 通路）
        gate1_start = 2 * d_pix
        gate1_end = 3 * d_pix
        gate2_start = 5 * d_pix
        gate2_end = 6 * d_pix
        nn.init.constant_(self.proj[-1].bias[gate1_start:gate1_end], 1.0)
        nn.init.constant_(self.proj[-1].bias[gate2_start:gate2_end], 1.0)

    def forward(self, s_cond: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            s_cond: (B, L, D_semantic) 语义条件
            h, w: patch 网格尺寸
        Returns:
            (B, H, W, 6, D_pix) 调制参数
        """
        B, L, D_sem = s_cond.shape
        p = self.p

        # Project to per-patch params
        params = self.proj(s_cond)                     # (B, L, 6*D_pix)
        params = params.reshape(B, h, w, 6, self.d_pix)  # (B, h, w, 6, D_pix)

        # Bilinear upsample to pixel grid
        params = params.permute(0, 3, 4, 1, 2)         # (B, 6, D_pix, h, w)
        params = params.reshape(B, 6 * self.d_pix, h, w)
        params = F.interpolate(params, scale_factor=p, mode='bilinear', align_corners=False)
        params = params.reshape(B, 6, self.d_pix, h * p, w * p)
        params = params.permute(0, 3, 4, 1, 2)         # (B, H, W, 6, D_pix)

        return params


class PiTBlock(nn.Module):
    """PiT Block（Pixel Transformer Block）。

    结构：
      Sub-block 1: GroupNorm → PixelAdaLN → Compaction → AttnO ∥ WFNO → Gate → Expand → 残差
      Sub-block 2: GroupNorm → PixelAdaLN → MLP(per-pixel) → 残差
    """

    def __init__(self, d_pix: int, d_cmp: int, d_semantic: int,
                 p: int = 16, wfno_modes: int = 12, wfno_alpha: float = 0.7,
                 attno_hidden_mult: int = 2, gate_hidden_mult: int = 1):
        super().__init__()
        self.p = p

        # ---- Sub-block 1 ----
        self.norm1 = nn.GroupNorm(num_groups=min(8, d_pix), num_channels=d_pix)
        self.adaln1 = PixelWiseAdaLN(d_semantic, d_pix, p)
        self.compaction = PixelTokenCompaction(p, d_pix, d_cmp)
        self.attno = AttnO(d_cmp, hidden_mult=attno_hidden_mult)
        self.wfno = WFNO(d_cmp, modes=wfno_modes, alpha=wfno_alpha)
        self.gate1 = GatedFusion(d_cmp, hidden_mult=gate_hidden_mult)
        self.expand = PixelTokenExpand(p, d_pix, d_cmp)

        # ---- Sub-block 2 ----
        self.norm2 = nn.GroupNorm(num_groups=min(8, d_pix), num_channels=d_pix)
        self.adaln2 = PixelWiseAdaLN(d_semantic, d_pix, p)
        self.mlp = nn.Sequential(
            nn.Conv2d(d_pix, d_pix * 6, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_pix * 6, d_pix, 3, padding=1),
        )
        # 平滑 expanded 特征的 patch 边界
        self.expand_smooth = nn.Conv2d(d_pix, d_pix, 3, padding=1)

    def forward(self, x: torch.Tensor, s_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D_pix, H, W) 像素特征
            s_cond: (B, L, D) 语义条件（来自 DiT）
        Returns:
            (B, D_pix, H, W) 更新后的像素特征
        """
        B, D_pix, H, W = x.shape
        p = self.p
        h, w = H // p, W // p
        L = h * w

        # ---- Sub-block 1: AttnO ∥ WFNO → Gate ----
        x_nchw = x  # (B, D_pix, H, W)
        x_nhwc = x_nchw.permute(0, 2, 3, 1)  # (B, H, W, D_pix)

        # Pixel-wise AdaLN
        params1 = self.adaln1(s_cond, h, w)              # (B, H, W, 6, D_pix)
        scale1 = params1[..., 0, :]                       # (B, H, W, D_pix)
        shift1 = params1[..., 1, :]
        gate1 = params1[..., 2, :]                        # (B, H, W, D_pix)

        x_norm1 = self.norm1(x_nchw).permute(0, 2, 3, 1)  # (B, H, W, D_pix)
        x_mod1 = x_norm1 * (1.0 + scale1) + shift1

        # Compaction
        x_cmp = self.compaction(x_mod1)                    # (B, L, D_cmp)

        # AttnO
        v_attno = self.attno(x_cmp)                        # (B, L, D_cmp)

        # WFNO（在 compacted token 的 2D grid 上）
        x_cmp_2d = x_cmp.reshape(B, h, w, -1)
        v_wfno = self.wfno(x_cmp_2d).reshape(B, L, -1)     # (B, L, D_cmp)

        # Gate
        v_fused = self.gate1(v_attno, v_wfno)               # (B, L, D_cmp)

        # Expand 回像素空间
        v_expanded = self.expand(v_fused, h, w)             # (B, H, W, D_pix)
        # 平滑 patch 边界
        v_expanded = self.expand_smooth(
            v_expanded.permute(0, 3, 1, 2)                  # (B, D_pix, H, W)
        ).permute(0, 2, 3, 1)                                # (B, H, W, D_pix)

        # 残差1
        x_nhwc = x_nhwc + gate1 * v_expanded

        # ---- Sub-block 2: MLP ----
        params2 = self.adaln2(s_cond, h, w)                 # (B, H, W, 6, D_pix)
        scale2 = params2[..., 3, :]
        shift2 = params2[..., 4, :]
        gate2 = params2[..., 5, :]

        x_nchw = x_nhwc.permute(0, 3, 1, 2)                # (B, D_pix, H, W)
        x_norm2 = self.norm2(x_nchw).permute(0, 2, 3, 1)   # (B, H, W, D_pix)
        x_mod2 = x_norm2 * (1.0 + scale2) + shift2

        v_mlp = self.mlp(x_mod2.permute(0, 3, 1, 2))       # (B, D_pix, H, W)
        v_mlp = v_mlp.permute(0, 2, 3, 1)                  # (B, H, W, D_pix)

        # 残差2
        x_nhwc = x_nhwc + gate2 * v_mlp

        # 输出 (B, D_pix, H, W)
        return x_nhwc.permute(0, 3, 1, 2)
