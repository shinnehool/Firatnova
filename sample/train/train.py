"""训练入口 — Rectified Flow + v-prediction + 均匀步长 ODE 医学图像 4× 超分。

用法:
    python -m train.train                          # 使用默认 config
    python -m train.train --resume                 # 从最新 checkpoint 续训
    python -m train.train --config config/config.yaml
"""

import sys
import os
import copy
import argparse
import logging
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from utils.config_loader import load_config
from utils.dataset import create_dataloaders
from utils.metrics import batch_psnr_ssim
from utils.visualization import save_triple, save_error_map, save_spectrum_map, save_loss_plot, save_metric_plot
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.ode_solver import RK3ODESolver, get_uniform_time_steps
from models import PixelDiTSR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
class EMA:
    """指数移动平均。"""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


# ---------------------------------------------------------------------------
# 训练主函数
# ---------------------------------------------------------------------------
def train(config: dict, resume: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    train_cfg = config["training"]
    model_cfg = config["model"]

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    logger.info("加载数据集...")
    train_loader, val_loader = create_dataloaders(
        train_dir=train_cfg["train_dir"],
        val_dir=train_cfg["val_dir"],
        hr_size=train_cfg["hr_size"],
        lr_size=train_cfg["lr_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        val_shuffle=train_cfg.get("val_shuffle", False),
    )
    logger.info(f"训练集样本数: {len(train_loader.dataset)}")
    logger.info(f"验证集样本数: {len(val_loader.dataset)}")

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------
    logger.info("构建模型...")
    model = PixelDiTSR(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数: {n_params:,}")

    # EMA
    ema = EMA(model, decay=train_cfg["ema_decay"])

    # ------------------------------------------------------------------
    # 优化器 & 调度器
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )

    lr_type = train_cfg["lr_scheduler"]
    if lr_type == "step":
        scheduler = MultiStepLR(
            optimizer,
            milestones=train_cfg["lr_step_epochs"],
            gamma=train_cfg["lr_decay_factor"],
        )
    elif lr_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    elif lr_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    else:
        scheduler = None

    # Warmup: 从 0 线性升到 base_lr
    warmup_steps = train_cfg.get("warmup_steps", 0)
    base_lr = train_cfg["lr"]
    if warmup_steps > 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = 1e-7  # warmup 起始 LR

    # ------------------------------------------------------------------
    # 断点续训
    # ------------------------------------------------------------------
    start_epoch = 1
    global_step = 0
    best_psnr = 0.0
    loss_history = []
    psnr_history = []
    ssim_history = []
    step_history = []

    ckpt_dir = train_cfg["checkpoint_dir"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    if resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            logger.info(f"恢复训练: {latest}")
            # 先用空 EMA 占位（load_checkpoint 不处理 EMA）
            tmp_ema_model = copy.deepcopy(model)
            ckpt = load_checkpoint(latest, model, optimizer, scheduler,
                                   ema_model=tmp_ema_model, device=device)
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_psnr = ckpt.get("best_psnr", 0.0)
            loss_history = ckpt.get("loss_history", [])
            psnr_history = ckpt.get("psnr_history", [])
            ssim_history = ckpt.get("ssim_history", [])
            step_history = ckpt.get("step_history", [])
            # 恢复 EMA shadow
            if "ema_state_dict" in ckpt:
                for name in ema.shadow:
                    if name in ckpt["ema_state_dict"]:
                        ema.shadow[name] = ckpt["ema_state_dict"][name].to(device)
            logger.info(f"从 epoch {start_epoch} / step {global_step} 继续")
        else:
            logger.info("未找到 checkpoint，从头训练。")

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    total_epochs = train_cfg["epochs"]
    log_interval = train_cfg["log_interval"]
    val_interval = train_cfg["val_interval"]
    grad_clip = train_cfg["grad_clip"]

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=120)

        for batch_idx, (lr, lr_up, hr) in enumerate(pbar):
            lr = lr.to(device, non_blocking=True)
            lr_up = lr_up.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            B = hr.shape[0]

            # ---- Rectified Flow: 均匀 t 采样 [0, 1]，训练-推理分布一致 ----
            t = torch.rand(B, device=device)
            x_t = (1.0 - t[:, None, None, None]) * hr + t[:, None, None, None] * lr_up

            # 目标速度: v = d/dt x_t = LR_up - HR
            v_target = lr_up - hr

            # ---- 前向 ----
            v_pred = model(x_t, t, lr)

            # ---- Loss: MSE + 频域 L1 ----
            loss_mse = F.mse_loss(v_pred, v_target)
            loss_freq = freq_loss(v_pred, v_target)
            loss = loss_mse + train_cfg["freq_loss_weight"] * loss_freq

            # ---- 反向 ----
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Warmup: 线性增加 LR
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / max(warmup_steps, 1)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr * lr_scale

            # EMA
            ema.update()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            loss_history.append(loss.item())

            # 进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            # ---- 日志 ----
            if global_step % log_interval == 0:
                logger.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss: {loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

            # ---- 验证 ----
            if global_step % val_interval == 0:
                val_psnr, val_ssim = validate(model, val_loader, device, config, ema=ema)
                psnr_history.append(val_psnr)
                ssim_history.append(val_ssim)
                step_history.append(global_step)

                logger.info(
                    f"验证 | Step {global_step} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}"
                )

                # 保存最佳模型
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, global_step,
                        best_psnr, loss_history, psnr_history, ssim_history, step_history,
                        str(Path(ckpt_dir) / "checkpoint_best.pth"),
                        ema_model=ema_model_from_shadow(model, ema),
                    )
                    logger.info(f"新最佳 PSNR: {best_psnr:.2f} dB")

                # 保存最新
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    best_psnr, loss_history, psnr_history, ssim_history, step_history,
                    str(Path(ckpt_dir) / "checkpoint_latest.pth"),
                    ema_model=ema_model_from_shadow(model, ema),
                )

                # 图表
                save_loss_plot(loss_history, "loss_curve.png", "Training Loss")
                if len(psnr_history) > 1:
                    save_metric_plot(step_history, psnr_history, "psnr_curve.png", "PSNR (dB)")
                    save_metric_plot(step_history, ssim_history, "ssim_curve.png", "SSIM")

                model.train()

        # ---- Epoch 结束 ----
        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch} 完成 | 平均 Loss: {avg_loss:.6f}")

        # 调度器
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(best_psnr)
            else:
                scheduler.step()

        # 定期保存
        if epoch % train_cfg["save_interval"] == 0 and epoch > 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                best_psnr, loss_history, psnr_history, ssim_history, step_history,
                str(Path(ckpt_dir) / f"checkpoint_epoch_{epoch:04d}.pth"),
                ema_model=ema_model_from_shadow(model, ema),
            )

        # 每个 epoch 保存最新
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            best_psnr, loss_history, psnr_history, ssim_history, step_history,
            str(Path(ckpt_dir) / "checkpoint_latest.pth"),
            ema_model=ema_model_from_shadow(model, ema),
        )

    logger.info("训练完成！")
    logger.info(f"最佳 PSNR: {best_psnr:.2f} dB")


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model: PixelDiTSR, val_loader, device: torch.device,
             config: dict, ema: EMA = None, max_samples: int = 16):
    """在验证集上计算 PSNR/SSIM，并保存三联图。"""

    p_psnr = []
    p_ssim = []

    # 使用 EMA 权重进行推理评估
    eval_model = ema_model_from_shadow(model, ema) if ema is not None else model

    for i, (lr, lr_up, hr) in enumerate(val_loader):
        if i >= max_samples:
            break

        lr = lr.to(device)
        lr_up = lr_up.to(device)
        hr = hr.to(device)

        sr = ode_inference(eval_model, lr, lr_up, config, device)

        # 单步直出（诊断用）
        eval_model.eval()
        t_one = torch.ones(1, device=device)
        v_t1 = eval_model(lr_up, t_one, lr)
        sr_single = torch.clamp(lr_up - v_t1, 0.0, 1.0)

        psnr_val, ssim_val = batch_psnr_ssim(sr, hr)
        psnr_single, _ = batch_psnr_ssim(sr_single, hr)
        p_psnr.append(psnr_val)
        p_ssim.append(ssim_val)

        # 保存第一张的三联图
        if i == 0:
            save_triple(lr_up[0], hr[0], sr[0], "result/val_triple.png")
            save_error_map(sr[0], hr[0], "result/val_error.png")
            save_spectrum_map(sr[0], hr[0], lr_up[0], "result/val_spectrum.png")
            logger.info(f"  ODE {config['inference']['num_ode_steps']}步: PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} | 单步直出: PSNR={psnr_single:.2f}")

    avg_psnr = float(sum(p_psnr) / max(len(p_psnr), 1))
    avg_ssim = float(sum(p_ssim) / max(len(p_ssim), 1))
    return avg_psnr, avg_ssim


# ---------------------------------------------------------------------------
# 频域损失
# ---------------------------------------------------------------------------
def freq_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """频域 L1 损失：在傅里叶空间计算 L1，强调高频一致性。"""
    v_pred_f = torch.fft.rfft2(v_pred, dim=(-2, -1))
    v_target_f = torch.fft.rfft2(v_target, dim=(-2, -1))
    return F.l1_loss(v_pred_f.real, v_target_f.real) + F.l1_loss(v_pred_f.imag, v_target_f.imag)


# ---------------------------------------------------------------------------
# ODE 推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def ode_inference(model: PixelDiTSR, lr: torch.Tensor, lr_up: torch.Tensor,
                   config: dict, device: torch.device) -> torch.Tensor:
    """多步 RK3 ODE 推理：均匀步长，从 t=1 积分到 t=0。只在最后一步 clamp。"""
    model.eval()
    num_steps = config["inference"]["num_ode_steps"]
    t_steps = get_uniform_time_steps(num_steps, device)  # (N+1,)  t_0=1 → t_N=0
    x = lr_up
    solver = RK3ODESolver()

    for i in range(num_steps):
        t_curr = t_steps[i].item()
        t_next = t_steps[i + 1].item()

        def velocity_fn(x_t, t_tensor):
            return model(x_t, t_tensor, lr)

        x = solver.step(velocity_fn, x, t_curr, t_next)

    # 只在最后一步 clamp，避免累积截断误差
    return torch.clamp(x, 0.0, 1.0)


def ema_model_from_shadow(model: torch.nn.Module, ema: EMA) -> torch.nn.Module:
    """从 EMA shadow 中导出模型副本。"""
    import copy
    ema_model = copy.deepcopy(model)
    for name, param in ema_model.named_parameters():
        if name in ema.shadow:
            param.data.copy_(ema.shadow[name])
    return ema_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PixelDiT-SR 训练")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--resume", action="store_true", help="从最新 checkpoint 续训")
    parser.add_argument("--smoke", action="store_true", help="使用 smoke 配置")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
