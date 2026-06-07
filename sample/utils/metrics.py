"""PSNR / SSIM 指标计算。"""

import torch
import torch.nn.functional as F
from math import log10
from typing import Tuple


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """计算单张图像的 PSNR（输入形状 C×H×W 或 B×C×H×W）。"""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse < 1e-12:
        return 100.0
    return float(10.0 * log10(max_val * max_val / mse))


def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
         window_size: int = 11, sigma: float = 1.5) -> float:
    """计算单张灰度图的 SSIM（输入 C×H×W 或 B×C×H×W，C 应为 1）。"""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # 先展平 batch
    pred = pred.contiguous()
    target = target.contiguous()

    C = pred.shape[1]
    if C != 1:
        # 逐通道平均
        vals = []
        for c in range(C):
            vals.append(_ssim_single(pred[:, c:c+1, :, :], target[:, c:c+1, :, :], max_val, window_size, sigma))
        return float(sum(vals) / len(vals))

    return _ssim_single(pred, target, max_val, window_size, sigma)


def _ssim_single(img1: torch.Tensor, img2: torch.Tensor, max_val: float,
                 window_size: int, sigma: float) -> float:
    """单通道 SSIM 计算。"""
    from math import exp

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # 高斯窗口
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = gauss[:, None] * gauss[None, :]  # (W, W)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)
    window = window.expand(img1.size(1), 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def batch_psnr_ssim(pred: torch.Tensor, target: torch.Tensor,
                    max_val: float = 1.0) -> Tuple[float, float]:
    """批量计算 PSNR 和 SSIM。"""
    psnr_val = psnr(pred, target, max_val)
    ssim_val = ssim(pred, target, max_val)
    return psnr_val, ssim_val
