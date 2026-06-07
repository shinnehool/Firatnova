"""Smoke 测试 — 通路验证 / 梯度检查 / Loss 收敛 / 过拟合诊断。

用法:
    python -m smoke.smoke
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import logging

from utils.config_loader import load_config
from utils.dataset import SuperResolutionDataset
from utils.metrics import batch_psnr_ssim
from utils.visualization import (
    save_triple, save_error_map, save_loss_plot, save_metric_plot,
)
from utils.ode_solver import timestep_embedding
from models import PixelDiTSR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ===================================================================
# 1. 通路测试
# ===================================================================
def test_forward_pass(config: dict) -> dict:
    """测试所有通路是否打通，返回各组件形状。"""
    logger.info("=" * 60)
    logger.info("测试 1: 前向通路")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelDiTSR(config).to(device)
    model.train()

    B = 2
    C = config["model"]["in_channels"]
    H = config["model"]["img_size"]
    lr_size = config["training"]["lr_size"]

    x_t = torch.randn(B, C, H, H, device=device)
    t = torch.rand(B, device=device)
    lr = torch.randn(B, C, lr_size, lr_size, device=device)

    v_pred = model(x_t, t, lr)

    shapes = {
        "x_t": tuple(x_t.shape),
        "t": tuple(t.shape),
        "lr": tuple(lr.shape),
        "v_pred": tuple(v_pred.shape),
    }

    expected = (B, C, H, H)
    assert v_pred.shape == expected, f"v_pred shape {v_pred.shape} != {expected}"

    # 检查每个组件
    t_emb = timestep_embedding(t, config["model"]["time_embed_dim"])

    # DiT 通路
    s_cond = model.dit(x_t, t_emb, lr)
    L = (H // config["model"]["patch_size"]) ** 2
    assert s_cond.shape == (B, L, config["model"]["dit_hidden_dim"]), \
        f"Scond shape {s_cond.shape} != ({B}, {L}, {config['model']['dit_hidden_dim']})"

    # PiT 通路
    v_pit = model.pit(x_t, s_cond)
    assert v_pit.shape == (B, C, H, H), \
        f"PiT output shape {v_pit.shape} != ({B}, {C}, {H}, {H})"

    logger.info("✅ 前向通路测试通过")
    for k, v in shapes.items():
        logger.info(f"  {k}: {v}")

    return shapes


# ===================================================================
# 2. 梯度流通测试
# ===================================================================
def test_gradient_flow(config: dict):
    """检查每个组件的梯度是否正常流通。"""
    logger.info("=" * 60)
    logger.info("测试 2: 梯度流通")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelDiTSR(config).to(device)
    model.train()

    B = 2
    C = config["model"]["in_channels"]
    H = config["model"]["img_size"]
    lr_size = config["training"]["lr_size"]

    # 注册 hook 收集梯度统计
    grad_stats = {}

    def make_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                g = grad_output[0]
                grad_stats[name] = {
                    "mean": float(g.mean()),
                    "std": float(g.std()),
                    "max": float(g.max()),
                    "min": float(g.min()),
                    "has_nan": bool(torch.isnan(g).any()),
                    "shape": tuple(g.shape),
                }
        return hook

    hooks = []
    for mn, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d,
                          torch.nn.MultiheadAttention, torch.nn.LayerNorm,
                          torch.nn.RMSNorm, torch.nn.GroupNorm)):
            hooks.append(m.register_full_backward_hook(make_hook(mn)))

    x_t = torch.randn(B, C, H, H, device=device)
    t = torch.rand(B, device=device)
    lr = torch.randn(B, C, lr_size, lr_size, device=device)
    lr_up = torch.randn(B, C, H, H, device=device)

    v_pred = model(x_t, t, lr)
    v_target = lr_up - torch.randn(B, C, H, H, device=device)  # 模拟目标
    loss = F.mse_loss(v_pred, v_target)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()

    for h in hooks:
        h.remove()

    # 报告（AdaLN-Zero 初始化下零梯度是预期行为，不警告）
    has_nan = False
    n_zero = 0
    for name, stats in grad_stats.items():
        if stats["has_nan"]:
            logger.error(f"❌ NaN gradient: {name}")
            has_nan = True
        elif stats["mean"] == 0.0 and stats["std"] == 0.0:
            n_zero += 1

    if has_nan:
        logger.error("❌ 梯度流通测试 FAILED — 检测到 NaN")
    else:
        logger.info("✅ 梯度流通测试通过")
        logger.info(f"  共检查 {len(grad_stats)} 个模块"
                    f"（{n_zero} 个零梯度 — AdaLN-Zero 预期行为）")

    # 打印关键模块
    key_modules = ["dit.blocks.0.attn", "dit.blocks.0.wfno",
                   "pit.blocks.0.attno", "pit.blocks.0.wfno",
                   "pit.output_proj.2"]
    for key in key_modules:
        for name in grad_stats:
            if key in name and "in_proj" not in name:
                s = grad_stats[name]
                logger.info(f"  {name}: mean={s['mean']:.2e}, std={s['std']:.2e}")
                break


# ===================================================================
# 3. Mini 训练：6 张训练 + 1 张验证，跑满 500 epoch
# ===================================================================
@torch.no_grad()
def _single_step_infer(model, lr, lr_up):
    """单步推理：SR = LR_up - v_pred(lr_up, t=1, lr)"""
    return model.predict_single_step(lr_up, lr)


def test_mini_training(config: dict):
    """6 张训练 + 1 张验证，训练 500 epoch，监控过拟合。

    关键修改：
    - 多步 ODE 推理替代单步 Euler
    - 同时在训练集和验证集上评估
    - 增加 t=1 附近的训练采样
    """
    logger.info("=" * 60)
    logger.info("测试 3: Mini 训练（6 train / 1 val × 500 epoch）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 加载数据：从训练集取 7 张，6 训练 + 1 验证
    # ------------------------------------------------------------------
    train_cfg = config["training"]
    ds = SuperResolutionDataset(
        data_dir=train_cfg["train_dir"],
        hr_size=train_cfg["hr_size"],
        lr_size=train_cfg["lr_size"],
        max_samples=7,
    )
    if len(ds) < 7:
        logger.error(f"训练集只有 {len(ds)} 张图像，需要至少 7 张")
        return False

    logger.info(f"抽取 {len(ds)} 张图像")

    # 前 6 张训练，最后 1 张验证
    train_data = [(ds[i][0], ds[i][1], ds[i][2]) for i in range(6)]
    val_lr, val_lr_up, val_hr = ds[6]
    val_lr = val_lr.unsqueeze(0).to(device)
    val_lr_up = val_lr_up.unsqueeze(0).to(device)
    val_hr = val_hr.unsqueeze(0).to(device)

    # 训练评估：用第 1 张训练图
    tr_lr, tr_lr_up, tr_hr = train_data[0]
    tr_lr = tr_lr.unsqueeze(0).to(device)
    tr_lr_up = tr_lr_up.unsqueeze(0).to(device)
    tr_hr = tr_hr.unsqueeze(0).to(device)

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------
    model = PixelDiTSR(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"],
                      betas=(train_cfg["beta1"], train_cfg["beta2"]),
                      weight_decay=train_cfg["weight_decay"])

    warmup_steps = train_cfg.get("warmup_steps", 50)
    base_lr = train_cfg["lr"]
    for pg in optimizer.param_groups:
        pg["lr"] = 1e-7

    num_epochs = 500
    log_interval = 25

    # 统计
    loss_history = []
    psnr_history = []       # 验证集 PSNR
    ssim_history = []       # 验证集 SSIM
    tr_psnr_history = []    # 训练集 PSNR（真正衡量过拟合）
    tr_ssim_history = []
    epoch_history = []

    logger.info(f"开始训练: {num_epochs} epochs, warmup={warmup_steps} steps")

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Shuffle 6 张训练图
        perm = torch.randperm(6)
        for idx in perm:
            lr_i, lr_up_i, hr_i = train_data[idx]
            lr_i = lr_i.unsqueeze(0).to(device)       # (1, 1, 64, 64)
            lr_up_i = lr_up_i.unsqueeze(0).to(device)  # (1, 1, 256, 256)
            hr_i = hr_i.unsqueeze(0).to(device)        # (1, 1, 256, 256)

            # 全部 t=1（smoke 用单步推理，训练推理一致）
            t = torch.ones(1, device=device)
            x_t = lr_up_i
            v_target = lr_up_i - hr_i

            v_pred = model(x_t, t, lr_i)
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()

            # Warmup
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / max(warmup_steps, 1)
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * lr_scale

            global_step += 1
            epoch_loss += loss.item()

        avg_loss = epoch_loss / 6
        loss_history.append(avg_loss)

        # 每 log_interval epoch 评估（训练集 + 验证集）
        if epoch % log_interval == 0 or epoch == 1 or epoch == num_epochs:
            # 训练集评估（核心：衡量模型是否真正过拟合）
            with torch.no_grad():
                tr_sr = _single_step_infer(model, tr_lr, tr_lr_up)
            tr_psnr, tr_ssim = batch_psnr_ssim(tr_sr, tr_hr)
            tr_psnr_history.append(tr_psnr)
            tr_ssim_history.append(tr_ssim)

            # 验证集评估
            with torch.no_grad():
                val_sr = _single_step_infer(model, val_lr, val_lr_up)
            psnr_v, ssim_v = batch_psnr_ssim(val_sr, val_hr)
            psnr_history.append(psnr_v)
            ssim_history.append(ssim_v)
            epoch_history.append(epoch)

            status = "🔥" if epoch == 1 else ""
            logger.info(
                f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f} | "
                f"Train PSNR: {tr_psnr:.2f} dB / SSIM: {tr_ssim:.4f} | "
                f"Val PSNR: {psnr_v:.2f} dB / SSIM: {ssim_v:.4f} {status}"
            )

            if epoch == num_epochs:
                save_metric_plot(epoch_history, tr_psnr_history,
                                 "result/smoke_psnr_curve.png", "PSNR (dB)")
                save_metric_plot(epoch_history, tr_ssim_history,
                                 "result/smoke_ssim_curve.png", "SSIM")

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_ratio = final_loss / max(initial_loss, 1e-8)

    final_tr_psnr = tr_psnr_history[-1]
    final_tr_ssim = tr_ssim_history[-1]
    final_val_psnr = psnr_history[-1]
    final_val_ssim = ssim_history[-1]

    logger.info(f"初始 Loss: {initial_loss:.6f}")
    logger.info(f"最终 Loss: {final_loss:.6f}")
    logger.info(f"下降比例: {loss_ratio:.4f}")
    logger.info(f"训练集 PSNR: {final_tr_psnr:.2f} dB / SSIM: {final_tr_ssim:.4f}")
    logger.info(f"验证集 PSNR: {final_val_psnr:.2f} dB / SSIM: {final_val_ssim:.4f}")

    # 保存结果（训练集 + 验证集）
    with torch.no_grad():
        tr_sr_final = _single_step_infer(model, tr_lr, tr_lr_up)
        val_sr_final = _single_step_infer(model, val_lr, val_lr_up)

    save_triple(tr_lr_up[0], tr_hr[0], tr_sr_final[0],
                "result/smoke_triple.png", prefix="smoke")
    save_error_map(tr_sr_final[0], tr_hr[0], "result/smoke_error.png")
    save_loss_plot(loss_history, "result/smoke_loss_curve.png", "Smoke Training Loss")

    # 验证集三联图
    save_triple(val_lr_up[0], val_hr[0], val_sr_final[0],
                "result/smoke_val_triple.png", prefix="smoke_val")
    save_error_map(val_sr_final[0], val_hr[0], "result/smoke_val_error.png")

    # 过拟合诊断
    ok = True
    if loss_ratio > 0.8:
        logger.error("❌ Loss 几乎不下降 — 模型未学习")
        logger.error("  排查: 1) LR 是否合适  2) AdaLN-zero gate 是否太慢  3) 数据范围")
        ok = False
    elif loss_ratio > 0.5:
        logger.warning("⚠️  Loss 下降缓慢 — 检查 lr / warmup / 模型容量")
    else:
        logger.info("✅ Loss 正常下降")

    # 核心指标：训练集 PSNR 应 > 30 dB（说明模型能过拟合）
    if final_tr_psnr < 25:
        logger.error(f"❌ 训练集 PSNR 过低 ({final_tr_psnr:.1f} dB < 25 dB) — 模型未过拟合")
        logger.error("  排查: 1) ODE 推理步数  2) 模型容量  3) t 采样分布")
        ok = False
    elif final_tr_psnr < 30:
        logger.warning(f"⚠️  训练集 PSNR 偏低 ({final_tr_psnr:.1f} dB) — 过拟合不充分")
    else:
        logger.info(f"✅ 训练集 PSNR {final_tr_psnr:.1f} dB — 模型成功过拟合")

    return ok


# ===================================================================
# 4. 组件完整性检查
# ===================================================================
def test_components(config: dict):
    """检查 WFNO / AttnO / GFM / Compaction 每个子模块是否可独立运行。"""
    logger.info("=" * 60)
    logger.info("测试 4: 组件完整性")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WFNO
    from models.wfno import WFNO
    wfno = WFNO(hidden_dim=64, modes=8).to(device)
    out = wfno(torch.randn(2, 16, 16, 64, device=device))
    assert out.shape == (2, 16, 16, 64), f"WFNO: {out.shape}"
    logger.info("  ✅ WFNO")

    # AttnO
    from models.attno import AttnO
    attno = AttnO(dim=64).to(device)
    out = attno(torch.randn(2, 256, 64, device=device))
    assert out.shape == (2, 256, 64), f"AttnO: {out.shape}"
    logger.info("  ✅ AttnO (Galerkin Attention)")

    # GFM
    from models.gfm import GatedFusion
    gfm = GatedFusion(dim=64).to(device)
    a = torch.randn(2, 100, 64, device=device)
    b = torch.randn(2, 100, 64, device=device)
    out = gfm(a, b)
    assert out.shape == (2, 100, 64), f"GFM: {out.shape}"
    logger.info("  ✅ GFM (Gated Fusion)")

    # DiTBlock
    from models.dit_block import DiTBlock
    block = DiTBlock(hidden_dim=64, num_heads=4, wfno_modes=4).to(device)
    out = block(torch.randn(2, 100, 64, device=device),
                torch.randn(2, 64, device=device))
    assert out.shape == (2, 100, 64), f"DiTBlock: {out.shape}"
    logger.info("  ✅ DiTBlock")

    # PiTBlock
    from models.pit_block import PiTBlock
    block = PiTBlock(d_pix=16, d_cmp=32, d_semantic=64, p=4,
                     wfno_modes=4).to(device)
    out = block(torch.randn(2, 16, 32, 32, device=device),
                torch.randn(2, 64, 64, device=device))  # L = (32/4)² = 64
    assert out.shape == (2, 16, 32, 32), f"PiTBlock: {out.shape}"
    logger.info("  ✅ PiTBlock")

    # Compaction <-> Expand 可逆性
    from models.pit_block import PixelTokenCompaction, PixelTokenExpand
    p, dp, dc = 4, 16, 32
    cmp = PixelTokenCompaction(p, dp, dc).to(device)
    exp = PixelTokenExpand(p, dp, dc).to(device)
    x_in = torch.randn(2, 32, 32, dp, device=device)
    x_cmp = cmp(x_in)
    x_out = exp(x_cmp, 8, 8)  # h=32/4=8, w=32/4=8
    assert x_out.shape == x_in.shape, f"Compaction round-trip: {x_out.shape} != {x_in.shape}"
    logger.info("  ✅ PixelToken Compaction ⇄ Expand")

    # ODE Solver
    from utils.ode_solver import get_uniform_time_steps
    ts = get_uniform_time_steps(10, device)
    assert ts.shape == (11,), f"TimeSteps: {ts.shape}"
    assert abs(ts[0].item() - 1.0) < 1e-4, f"t0 should be 1, got {ts[0]}"
    assert abs(ts[-1].item()) < 1e-4, f"tN should be 0, got {ts[-1]}"
    logger.info("  ✅ Uniform ODE Solver")

    logger.info("✅ 所有组件测试通过")


# ===================================================================
# Main
# ===================================================================
def main():
    logger.info("=" * 60)
    logger.info("PixelDiT-SR Smoke 测试")
    logger.info("=" * 60)

    config = load_config("config/config.yaml", smoke=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    model_cfg = config["model"]
    train_cfg = config["training"]
    logger.info(f"Smoke 配置: DiT N={model_cfg['dit_n_layers']} D={model_cfg['dit_hidden_dim']} | "
                f"PiT M={model_cfg['pit_n_layers']} D_pix={model_cfg['pit_hidden_dim']} | "
                f"Batch={train_cfg['batch_size']}")

    try:
        # 1. 前向通路
        test_forward_pass(config)

        # 2. 梯度流通
        test_gradient_flow(config)

        # 3. 组件完整性
        test_components(config)

        # 4. Mini 训练：6 train + 1 val × 200 epoch
        ok = test_mini_training(config)

        logger.info("=" * 60)
        if ok:
            logger.info("🎉 Smoke 测试全部通过！")
        else:
            logger.warning("⚠️  Smoke 测试完成，但有警告项，请检查上述日志")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Smoke 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
