"""PixelDiT-SR：顶层模型。

结合 DiT（Patch‑level 语义通路）和 PiT（Pixel‑level 纹理通路），
在 Rectified Flow 框架下预测速度场 v_θ。

Forward path:  x_t = (1 - t) * HR + t * LR_up
Target:       v = LR_up - HR = -R（残差的负值）
Model output: v_θ(x_t, t, LR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import DiT
from .pit import PiT
from utils.ode_solver import timestep_embedding


class ConvResBlock(nn.Module):
    """残差卷积块：Conv→GELU→Conv + skip connection。"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.act(self.conv1(x))) + self.skip(x))


class PixelDiTSR(nn.Module):
    """PixelDiT for Super-Resolution。

    输入 x_t（当前插值状态）、t（时间）、LR（原始低分图像），
    输出预测速度 v_pred。
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]

        # ---- 通用参数 ----
        img_size = cfg["img_size"]
        patch_size = cfg["patch_size"]
        in_channels = cfg["in_channels"]
        time_embed_dim = cfg["time_embed_dim"]

        # ---- DiT ----
        self.dit = DiT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=cfg["dit_hidden_dim"],
            num_heads=cfg["dit_num_heads"],
            n_layers=cfg["dit_n_layers"],
            time_embed_dim=time_embed_dim,
            lr_embed_dim=cfg["lr_encoder_embed_dim"],
            wfno_modes=cfg["wfno_modes"],
            wfno_alpha=cfg["wfno_alpha"],
            gate_hidden_mult=cfg["gate_hidden_mult"],
        )

        # ---- PiT ----
        self.pit = PiT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_pix=cfg["pit_hidden_dim"],
            d_cmp=cfg["pit_cmp_dim"],
            d_semantic=cfg["dit_hidden_dim"],
            n_layers=cfg["pit_n_layers"],
            wfno_modes=cfg["wfno_modes"],
            wfno_alpha=cfg["wfno_alpha"],
            attno_hidden_mult=cfg["attno_hidden_mult"],
            gate_hidden_mult=cfg["gate_hidden_mult"],
        )

        # ---- LR 空间编码器：多尺度残差编码，保留高频细节 ----
        pit_hidden = cfg["pit_hidden_dim"]
        # Stage 1: 64×64 残差特征提取
        self.lr_enc_stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, 1, 1),
            nn.GELU(),
            ConvResBlock(48, 48),
            ConvResBlock(48, 48),
        )
        # Stage 2: 128×128 残差细化（保持 48 通道不压缩）
        self.lr_enc_stage2 = nn.Sequential(
            ConvResBlock(48, 48),
            ConvResBlock(48, 48),
        )
        # Stage 3: 256×256 残差细化 + 通道投影
        self.lr_enc_stage3 = nn.Sequential(
            ConvResBlock(48, 48),
            nn.Conv2d(48, pit_hidden, 3, 1, 1),
        )

        self.time_embed_dim = time_embed_dim
        self.img_size = img_size

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, C, H, W) 当前状态（像素空间）
            t: (B,) 归一化时间（范围 [0, 1]）
            lr: (B, C, 64, 64) 原始 LR 图像
        Returns:
            v_pred: (B, C, H, W) 预测的速度场
        """
        # 时间嵌入
        t_emb = timestep_embedding(t, self.time_embed_dim)  # (B, D_t)

        # 1. DiT 生成 Scond（纯条件生成，不丢弃）
        s_cond = self.dit(x_t, t_emb, lr)  # (B, L, D)

        # 2. LR 空间特征（多尺度渐进上采样 → PiT，全程 48 通道）
        _, _, H, W = x_t.shape
        lr_feat = self.lr_enc_stage1(lr)                                # (B, 48, 64, 64)
        lr_feat = F.interpolate(lr_feat, scale_factor=2, mode="bicubic")# (B, 48, 128, 128)
        lr_feat = self.lr_enc_stage2(lr_feat)                           # (B, 48, 128, 128)
        lr_feat = F.interpolate(lr_feat, size=(H, W), mode="bicubic")   # (B, 48, 256, 256)
        lr_spatial = self.lr_enc_stage3(lr_feat)                        # (B, d_pix, H, W)

        # 3. PiT 预测速度
        v_pred = self.pit(x_t, s_cond, lr_spatial=lr_spatial)  # (B, C, H, W)

        return v_pred

    @torch.no_grad()
    def predict_single_step(self, lr_up: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """单步推理：从 t=1 直接用预测速度还原 HR。

        Rectified Flow: x_1=LR_up, v=d(x_t)/dt=LR_up-HR
        → HR = LR_up - v(x_1, t=1, LR)
        """
        was_training = self.training
        self.eval()
        B = lr.shape[0]
        t_one = torch.ones(B, device=lr.device)
        v_pred = self(lr_up, t_one, lr)
        sr = torch.clamp(lr_up - v_pred, 0.0, 1.0)
        if was_training:
            self.train()
        return sr
