"""可视化工具：三联图、散点图、误差图、频谱图。"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将 tensor 转为 numpy，范围 [0, 1]，形状 H×W。"""
    img = tensor.detach().cpu().squeeze().numpy()
    if img.ndim == 3:
        img = img[0]  # 取第一通道
    img = np.clip(img, 0.0, 1.0)
    return img


def save_triple(lr: torch.Tensor, hr: torch.Tensor, sr: torch.Tensor,
                save_path: str, prefix: str = "sample", sr_second: Optional[torch.Tensor] = None):
    """保存三联图（第 1 行 LR，第 2 行 HR，第 3 行 SR）。
    若提供 sr_second，则第 4 行显示对比 SR。
    """
    lr_img = _to_numpy(lr)
    hr_img = _to_numpy(hr)
    sr_img = _to_numpy(sr)

    nrows = 4 if sr_second is not None else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(6, 6 * nrows))

    idx = 0
    axes[idx].imshow(lr_img, cmap="gray", vmin=0, vmax=1)
    axes[idx].set_title("LR (Input)")
    axes[idx].axis("off")
    idx += 1

    axes[idx].imshow(hr_img, cmap="gray", vmin=0, vmax=1)
    axes[idx].set_title("HR (Ground Truth)")
    axes[idx].axis("off")
    idx += 1

    axes[idx].imshow(sr_img, cmap="gray", vmin=0, vmax=1)
    axes[idx].set_title("SR (Prediction)")
    axes[idx].axis("off")
    idx += 1

    if sr_second is not None:
        sr2_img = _to_numpy(sr_second)
        axes[idx].imshow(sr2_img, cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title("SR (Compare)")
        axes[idx].axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_error_map(sr: torch.Tensor, hr: torch.Tensor, save_path: str):
    """保存误差图（|SR - HR|）。"""
    sr_img = _to_numpy(sr)
    hr_img = _to_numpy(hr)
    error = np.abs(sr_img - hr_img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(sr_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("SR")
    axes[0].axis("off")

    axes[1].imshow(hr_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("HR")
    axes[1].axis("off")

    im = axes[2].imshow(error, cmap="hot", vmin=0, vmax=error.max() if error.max() > 0 else 1)
    axes[2].set_title("|SR - HR|")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_spectrum_map(sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor, save_path: str):
    """保存频谱图（对数幅度谱）。"""
    sr_img = _to_numpy(sr)
    hr_img = _to_numpy(hr)
    lr_img = _to_numpy(lr)

    def log_mag(img):
        f = np.fft.fftshift(np.fft.fft2(img))
        return np.log(np.abs(f) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(
        axes,
        [log_mag(lr_img), log_mag(hr_img), log_mag(sr_img)],
        ["LR Spectrum", "HR Spectrum", "SR Spectrum"]
    ):
        im = ax.imshow(img, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_plot(losses: List[float], save_path: str, label: str = "Loss"):
    """保存 loss 曲线。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(label)
    ax.set_title(f"{label} Curve")
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(steps: List[int], values: List[float], save_path: str, metric_name: str):
    """保存指标趋势图。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, values, marker="o", markersize=3, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Trend")
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_scatter(x_values: List[float], y_values: List[float],
                 save_path: str, xlabel: str = "X", ylabel: str = "Y"):
    """保存散点图。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_values, y_values, s=10, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel}")
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
