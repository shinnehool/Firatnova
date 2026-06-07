"""Weighted Fourier Neural Operator（加权傅里叶神经算子）。

在标准 FNO 的基础上增加 Mode Rebalancing：
  w(ξ) = 1 + γ · ‖ξ‖^α
以强调高频模态，克服 FNO 的 mode truncation 导致的细节丢失。

输入/输出形状：2D 空间特征图 (B, H, W, D)。
"""

import torch
import torch.nn as nn
import torch.fft


class WFNO(nn.Module):
    """加权傅里叶神经算子。

    对 2D 空间特征图进行频域卷积 + Mode Rebalancing。
    """

    def __init__(self, hidden_dim: int, modes: int = 12, alpha: float = 0.7):
        """
        Args:
            hidden_dim: 特征维度 D
            modes: 保留的傅里叶模态数（实部维度截断）
            alpha: Mode Rebalancing 指数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modes = modes
        self.alpha = alpha

        # 可学习频域滤波器（复数）
        scale = 1.0 / (hidden_dim ** 0.5)
        self.W_freq_real = nn.Parameter(torch.randn(modes, modes, hidden_dim) * scale)
        self.W_freq_imag = nn.Parameter(torch.randn(modes, modes, hidden_dim) * scale)

        # Mode Rebalancing 的 γ
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def _get_mode_weights(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """计算每个频率位置的 rebalancing 权重。

        RFFT2(dim=(1,2)) 的频率布局:
          dim 1 (H): 完整 FFT → 频率索引 [0..H/2] 为正频, (H/2..H-1] 对应负频 (-(H-u))
          dim 2 (W): 半 RFFT → 频率索引 [0..W/2] 全为非负频

        Returns:
            (h, w, 1) 权重张量
        """
        u_raw = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)  # (h, 1)
        v = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0)       # (1, w)
        # 正确的物理频率幅值: min(index, H-index) 处理负频折叠
        u_freq = torch.minimum(u_raw, h - u_raw)  # (h, 1)
        freq_mag = torch.sqrt(u_freq ** 2 + v ** 2)  # (h, w)
        weight = 1.0 + self.gamma * (freq_mag ** self.alpha)
        return weight.unsqueeze(-1)  # (h, w, 1)

    @staticmethod
    def _complex_mult(a_real: torch.Tensor, a_imag: torch.Tensor,
                       b_real: torch.Tensor, b_imag: torch.Tensor):
        """复数乘法：(a_real + i*a_imag) * (b_real + i*b_imag)。"""
        real = a_real * b_real - a_imag * b_imag
        imag = a_real * b_imag + a_imag * b_real
        return real, imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, D) 2D 空间特征图
        Returns:
            (B, H, W, D) 经过频域卷积的输出
        """
        B, H, W, D = x.shape

        # 1. 2D RFFT
        x_freq = torch.fft.rfft2(x, dim=(1, 2))  # (B, H, W_f, D)
        W_f = x_freq.shape[2]  # = W // 2 + 1
        x_freq_real = x_freq.real  # (B, H, W_f, D)
        x_freq_imag = x_freq.imag

        # 2. Mode Rebalancing
        mode_weight = self._get_mode_weights(H, W_f, x.device)  # (H, W_f, 1)
        x_freq_real = x_freq_real * mode_weight
        x_freq_imag = x_freq_imag * mode_weight

        # 3. 频率截断（只保留低 modes）
        h_start = min(self.modes, H)
        w_start = min(self.modes, W_f)

        x_freq_real_trunc = x_freq_real[:, :h_start, :w_start, :]
        x_freq_imag_trunc = x_freq_imag[:, :h_start, :w_start, :]

        W_r = self.W_freq_real[:h_start, :w_start, :]  # (modes, modes, D)
        W_i = self.W_freq_imag[:h_start, :w_start, :]

        # 4. 复数乘法（逐元素 × 频域滤波器）
        out_real, out_imag = self._complex_mult(
            x_freq_real_trunc, x_freq_imag_trunc,
            W_r.unsqueeze(0), W_i.unsqueeze(0),
        )

        # 5. 填回全频谱（高频保留但不乘以 filter）
        out_full_real = x_freq_real.clone()
        out_full_imag = x_freq_imag.clone()
        out_full_real[:, :h_start, :w_start, :] = out_real
        out_full_imag[:, :h_start, :w_start, :] = out_imag

        # 6. IRFFT
        out_complex = torch.complex(out_full_real, out_full_imag)
        out = torch.fft.irfft2(out_complex, dim=(1, 2), s=(H, W))

        return out  # (B, H, W, D)
