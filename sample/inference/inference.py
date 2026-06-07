"""推理入口 — 对验证集/任意图像执行 4× 超分推理。

用法:
    python -m inference.inference
    python -m inference.inference --input test/
    python -m inference.inference --input image.png --checkpoint checkpoints/checkpoint_best.pth
"""

import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils.config_loader import load_config
from utils.metrics import batch_psnr_ssim
from utils.visualization import save_triple, save_error_map, save_spectrum_map
from utils.checkpoint import load_checkpoint
from utils.ode_solver import RK3ODESolver, get_uniform_time_steps
from models import PixelDiTSR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(config: dict, checkpoint_path: str, input_path: str,
                  output_dir: str = "result"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------
    logger.info("加载模型...")
    model = PixelDiTSR(config).to(device)
    model.eval()

    ckpt = load_checkpoint(checkpoint_path, model, device=device)
    logger.info(f"已加载 checkpoint (epoch {ckpt.get('epoch', '?')})")

    # ------------------------------------------------------------------
    # 查找输入文件
    # ------------------------------------------------------------------
    root = Path(__file__).resolve().parent.parent
    input_path = root / input_path if not Path(input_path).is_absolute() else Path(input_path)

    if input_path.is_dir():
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.npy"]:
            image_files.extend(sorted(input_path.glob(ext)))
        image_files = sorted(image_files, key=lambda p: p.stem)
        logger.info(f"找到 {len(image_files)} 张图像在 {input_path}")
    else:
        image_files = [input_path]
        logger.info(f"单张图像: {input_path}")

    if not image_files:
        raise RuntimeError("未找到图像文件")

    # ------------------------------------------------------------------
    # 推理循环
    # ------------------------------------------------------------------
    out_root = root / output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    num_steps = config["inference"]["num_ode_steps"]
    lr_size = config["training"].get("lr_size", 64)
    hr_size = config["training"].get("hr_size", 256)

    all_psnr = []
    all_ssim = []

    for img_path in tqdm(image_files, desc="推理"):
        try:
            in_tensor = _load_image(img_path)
        except Exception as e:
            logger.error(f"加载失败: {img_path} — {e}")
            continue

        in_tensor = in_tensor.to(device)
        if in_tensor.dim() == 3:
            in_tensor = in_tensor.unsqueeze(0)

        B, C, H_img, W_img = in_tensor.shape
        name = img_path.stem

        # 判断输入是 LR 还是 HR（短边 < 100 视为 LR）
        is_lr_input = min(H_img, W_img) < 100

        if is_lr_input:
            # 输入为 LR：直接上采样后推理，无 ground truth
            lr = in_tensor if H_img == lr_size and W_img == lr_size else \
                 F.interpolate(in_tensor, (lr_size, lr_size), mode="bicubic")
            lr_up = F.interpolate(lr, (hr_size, hr_size), mode="bicubic")
            sr = ode_infer(model, lr, lr_up, num_steps, device)

            _save_single(sr[0], out_root / f"{name}_SR.png")
            _save_single(lr_up[0], out_root / f"{name}_LRup.png")

            # 三联图（无 HR：第 2 行为 LR_up bicubic 参考）
            save_triple(
                lr_up[0], lr_up[0], sr[0],
                str(out_root / f"{name}_triple.png"),
                sr_second=lr_up[0],
            )
            logger.info(f"   {name}: LR→SR (无 metrics)")

        else:
            # 输入为 HR（有 ground truth，可算指标）
            hr_tensor = in_tensor
            lr = F.interpolate(hr_tensor, (lr_size, lr_size), mode="bicubic")
            lr_up = F.interpolate(lr, (hr_size, hr_size), mode="bicubic")
            sr = ode_infer(model, lr, lr_up, num_steps, device)

            # 恢复原图尺寸
            sr_eval = F.interpolate(sr, (H_img, W_img), mode="bicubic") \
                      if (H_img != hr_size or W_img != hr_size) else sr

            psnr_val, ssim_val = batch_psnr_ssim(sr_eval, hr_tensor)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)

            _save_single(sr_eval[0], out_root / f"{name}_SR.png")
            _save_single(lr_up[0], out_root / f"{name}_LRup.png")
            save_triple(lr_up[0], hr_tensor[0], sr_eval[0],
                         str(out_root / f"{name}_triple.png"))
            save_error_map(sr_eval[0], hr_tensor[0],
                           str(out_root / f"{name}_error.png"))
            save_spectrum_map(sr_eval[0], hr_tensor[0], lr_up[0],
                              str(out_root / f"{name}_spectrum.png"))
            logger.info(f"   {name}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")

    # ---- 汇总 ----
    if all_psnr:
        logger.info(f"平均 PSNR: {np.mean(all_psnr):.2f} dB | 平均 SSIM: {np.mean(all_ssim):.4f}")
        logger.info(f"结果保存在: {out_root}")
    else:
        logger.warning("无有效推理结果。")

    return all_psnr, all_ssim


@torch.no_grad()
def ode_infer(model: PixelDiTSR, lr: torch.Tensor, lr_up: torch.Tensor,
               num_steps: int, device: torch.device) -> torch.Tensor:
    """多步 RK3 ODE 推理：均匀步长，从 t=1 积分到 t=0。只在最后一步 clamp。"""
    model.eval()
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


def _load_image(path: Path) -> torch.Tensor:
    """加载图像为 (C, H, W) tensor，范围 [0, 1]。"""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = arr.transpose(2, 0, 1)
        img = torch.from_numpy(arr.astype(np.float32))
    else:
        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")
        img = TF.to_tensor(img)
    return torch.clamp(img, 0.0, 1.0)


def _save_single(tensor: torch.Tensor, path: Path):
    """保存单张图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor.squeeze().cpu().numpy()
    img = np.clip(img, 0.0, 1.0) * 255
    img = img.astype(np.uint8)
    Image.fromarray(img, mode="L").save(str(path))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PixelDiT-SR 推理")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pth")
    parser.add_argument("--input", type=str, default="test")
    parser.add_argument("--output", type=str, default="result")
    args = parser.parse_args()

    config = load_config(args.config)
    run_inference(config, args.checkpoint, args.input, args.output)


if __name__ == "__main__":
    main()
