"""DiT（Diffusion Transformer）— Patch‑level 语义通路。

由 N 个 DiTBlock 堆叠而成，输出 Scond（语义条件 tokens）。
"""

import torch
import torch.nn as nn

from .dit_block import DiTBlock


class DiT(nn.Module):
    """Patch‑level DiT。

    将输入图像 x_t 做 patchify → N 层 DiTBlock → 输出 Scond。
    """

    def __init__(self,
                 img_size: int = 256,
                 patch_size: int = 16,
                 in_channels: int = 1,
                 hidden_dim: int = 768,
                 num_heads: int = 12,
                 n_layers: int = 26,
                 time_embed_dim: int = 256,
                 lr_embed_dim: int = 128,
                 wfno_modes: int = 12,
                 wfno_alpha: float = 0.7,
                 gate_hidden_mult: int = 1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * patch_size * in_channels, hidden_dim)

        # 可学习的 position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        # 时间嵌入 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # LR 编码器（从 LR(64) 提取条件）
        self.lr_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),    # 64 → 32
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),              # 32 → 16
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),             # 16 → 8
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(128, lr_embed_dim),
            nn.SiLU(),
        )

        # 条件融合：time_embed + lr_embed → c
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim + lr_embed_dim, hidden_dim),
            nn.SiLU(),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                wfno_modes=wfno_modes,
                wfno_alpha=wfno_alpha,
                gate_hidden_mult=gate_hidden_mult,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 跳过 AdaLN 层（将在 block 初始化中单独处理）
                if m in {b.adaLN_proj1[-1] for b in self.blocks} | \
                       {b.adaLN_proj2[-1] for b in self.blocks}:
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_t: torch.Tensor, t_emb: torch.Tensor,
                lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, C, H, W) 当前状态
            t_emb: (B, time_embed_dim) 时间嵌入
            lr: (B, C, 64, 64) 原始 LR 图像
        Returns:
            Scond: (B, L, D) 语义条件 tokens
        """
        B, C, H, W = x_t.shape
        p = self.patch_size
        h, w = H // p, W // p

        # Patchify
        x_patches = x_t.reshape(B, C, h, p, w, p)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5)
        x_patches = x_patches.reshape(B, h * w, p * p * C)

        # Embed
        x = self.patch_embed(x_patches)  # (B, L, D)
        x = x + self.pos_embed

        # 时间条件
        t_feat = self.time_mlp(t_emb)    # (B, D)

        # LR 条件
        lr_feat = self.lr_encoder(lr)    # (B, lr_embed_dim)

        # 全局条件向量 c
        c = self.cond_proj(torch.cat([t_feat, lr_feat], dim=-1))  # (B, D)

        # DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # 最终 Scond + 时间嵌入
        x = self.final_norm(x)

        return x  # (B, L, D)
