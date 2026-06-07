"""PiT（Pixel Transformer）— Pixel‑level 纹理通路。

由 M 个 PiTBlock 堆叠而成，接收 Scond 生成速度场 v_θ。
"""

import torch
import torch.nn as nn

from .pit_block import PiTBlock


class PiT(nn.Module):
    """Pixel‑level PiT。

    输入 x_t 像素 + Scond → M 层 PiTBlock → 输出速度场。
    """

    def __init__(self,
                 img_size: int = 256,
                 patch_size: int = 16,
                 in_channels: int = 1,
                 d_pix: int = 16,
                 d_cmp: int = 128,
                 d_semantic: int = 768,
                 n_layers: int = 4,
                 wfno_modes: int = 12,
                 wfno_alpha: float = 0.7,
                 attno_hidden_mult: int = 2,
                 gate_hidden_mult: int = 1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_pix = d_pix
        self.out_channels = in_channels

        # 像素嵌入
        self.pixel_embed = nn.Conv2d(in_channels, d_pix, 3, padding=1)

        # PiT blocks
        self.blocks = nn.ModuleList([
            PiTBlock(
                d_pix=d_pix,
                d_cmp=d_cmp,
                d_semantic=d_semantic,
                p=patch_size,
                wfno_modes=min(wfno_modes, d_cmp),
                wfno_alpha=wfno_alpha,
                attno_hidden_mult=attno_hidden_mult,
                gate_hidden_mult=gate_hidden_mult,
            )
            for _ in range(n_layers)
        ])

        # 输出投影：D_pix → C
        self.output_proj = nn.Sequential(
            nn.Conv2d(d_pix, d_pix * 2, 1),
            nn.GELU(),
            nn.Conv2d(d_pix * 2, in_channels, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_t: torch.Tensor, s_cond: torch.Tensor,
                lr_spatial: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_t: (B, C, H, W) 当前状态（像素空间）
            s_cond: (B, L, D_semantic) 语义条件（来自 DiT）
            lr_spatial: (B, D_pix, H, W) LR 空间特征（可选）
        Returns:
            v_pred: (B, C, H, W) 预测的速度场
        """
        # 像素嵌入
        x = self.pixel_embed(x_t)  # (B, D_pix, H, W)
        if lr_spatial is not None:
            x = x + lr_spatial

        # PiT blocks
        for block in self.blocks:
            x = block(x, s_cond)

        # 输出投影
        v_pred = self.output_proj(x)  # (B, C, H, W)

        return v_pred
