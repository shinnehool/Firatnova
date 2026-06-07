"""Checkpoint 保存与恢复，支持断点续训。"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_psnr: float,
    loss_history: list,
    psnr_history: list,
    ssim_history: list,
    step_history: list,
    save_path: str,
    ema_model: Optional[nn.Module] = None,
):
    """保存 checkpoint。调用方负责指定目标路径。"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss_history": loss_history,
        "psnr_history": psnr_history,
        "ssim_history": ssim_history,
        "step_history": step_history,
    }

    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint 已保存: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema_model: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """加载 checkpoint 并恢复训练状态。

    Returns:
        checkpoint dict 包含 epoch / global_step / best_psnr / loss_history 等
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 恢复模型
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"模型权重已从 {checkpoint_path} 恢复")

    # 恢复 EMA
    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
        logger.info("EMA 权重已恢复")

    # 恢复优化器
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("优化器状态已恢复")

    # 恢复调度器
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("调度器状态已恢复")

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """查找最新的 checkpoint_latest.pth。"""
    latest = Path(checkpoint_dir) / "checkpoint_latest.pth"
    if latest.exists():
        return str(latest)
    return None
