"""ODE 求解器：均匀步长 + RK3 积分。

Rectified Flow 的 v = LR_up - HR 为常数，最优步长即均匀分布。
"""

import torch
import math
from typing import Callable


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """正弦时间嵌入。

    Args:
        t: (B,) 或标量，范围 [0, 1]
        dim: 嵌入维度
        max_period: 最大周期
    Returns:
        (B, dim)
    """
    if t.dim() == 0:
        t = t.unsqueeze(0)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def get_uniform_time_steps(num_steps: int, device: torch.device = None) -> torch.Tensor:
    """生成均匀分布的 ODE 时间步。

    Rectified Flow 的 v = LR_up - HR 为常数，最优步长即均匀分布。

    Args:
        num_steps: ODE 步数 N
        device: 目标设备
    Returns:
        (N+1,) 时间步 t_0=1, t_1, ..., t_N=0（从高到低，反向积分用）
    """
    return torch.linspace(1.0, 0.0, num_steps + 1, device=device)


class RK3ODESolver:
    """三阶 Runge-Kutta ODE 求解器。

    用于 Rectified Flow 的反向积分（从 t=1 → t=0）。
    dx/dt = v_θ(x, t)
    """

    @staticmethod
    def step(
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
    ) -> torch.Tensor:
        """单步 RK3 积分。

        Args:
            velocity_fn: (x, t_tensor) → v_pred
            x: 当前状态 (B, C, H, W)
            t_curr: 当前时间
            t_next: 下一时间
        Returns:
            更新后的 x
        """
        B = x.shape[0]
        device = x.device
        h = t_next - t_curr  # 负值（反向积分）

        t_curr_t = torch.full((B,), t_curr, device=device)
        t_mid_t = torch.full((B,), (t_curr + t_next) / 2, device=device)
        t_next_t = torch.full((B,), t_next, device=device)

        k1 = velocity_fn(x, t_curr_t)
        k2 = velocity_fn(x + h / 2 * k1, t_mid_t)
        # RK3 经典系数
        k3 = velocity_fn(x - h * k1 + 2 * h * k2, t_next_t)

        return x + h / 6 * (k1 + 4 * k2 + k3)
