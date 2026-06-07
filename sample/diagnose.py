"""诊断脚本：逐一检查训练-推理链路的每个环节。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from utils.config_loader import load_config
from utils.dataset import create_dataloaders
from utils.metrics import batch_psnr_ssim
from utils.checkpoint import load_checkpoint
from utils.ode_solver import RK3ODESolver, timestep_embedding, get_uniform_time_steps
from models import PixelDiTSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 加载模型
config = load_config("config/config_fast.yaml")
model = PixelDiTSR(config).to(device)
ckpt_path = "checkpoints/checkpoint_latest.pth"
try:
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    print(f"已加载 checkpoint (epoch {ckpt.get('epoch', '?')})")
except:
    print("未找到 checkpoint，使用随机初始化模型")

model.eval()

# 加载数据
_, val_loader = create_dataloaders(
    train_dir="train_fast", val_dir="train_fast",
    hr_size=256, lr_size=64, batch_size=2, num_workers=0, val_shuffle=True,
)
lr, lr_up, hr = next(iter(val_loader))
lr, lr_up, hr = lr.to(device), lr_up.to(device), hr.to(device)

print(f"\n{'='*60}")
print("1. 数据范围检查")
print(f"   LR   : min={lr.min():.4f} max={lr.max():.4f}")
print(f"   LR_up: min={lr_up.min():.4f} max={lr_up.max():.4f}")
print(f"   HR   : min={hr.min():.4f} max={hr.max():.4f}")
print(f"   v_target = lr_up - hr: min={(lr_up-hr).min():.4f} max={(lr_up-hr).max():.4f}")

print(f"\n{'='*60}")
print("2. 速度预测 — 不同 t 值的表现")

B = lr.shape[0]
solver = RK3ODESolver()
num_steps = config["inference"]["num_ode_steps"]
t_steps = get_uniform_time_steps(num_steps, device)

for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    t = torch.full((B,), t_val, device=device)
    x_t = (1 - t[:, None, None, None]) * hr + t[:, None, None, None] * lr_up
    with torch.no_grad():
        v_pred = model(x_t, t, lr)
    v_target = lr_up - hr
    mse = F.mse_loss(v_pred, v_target).item()
    cos_sim = F.cosine_similarity(
        v_pred.reshape(B, -1), v_target.reshape(B, -1), dim=-1
    ).mean().item()
    sr_direct = torch.clamp(lr_up - v_pred, 0.0, 1.0)
    psnr_direct, _ = batch_psnr_ssim(sr_direct, hr)
    print(f"   t={t_val:.2f}: MSE={mse:.6f} cos_sim={cos_sim:.4f} 预测SR→PSNR={psnr_direct:.2f}")

print(f"\n{'='*60}")
print("2b. 单步推理 (predict_single_step)")
with torch.no_grad():
    sr_single = model.predict_single_step(lr_up, lr)
psnr_single, ssim_single = batch_psnr_ssim(sr_single, hr)
print(f"   单步 PSNR={psnr_single:.2f} SSIM={ssim_single:.4f}")

print(f"\n{'='*60}")
print("3. ODE 积分 — 不同步数的最终 PSNR")

for n_steps in [1, 5, 10, 15, 30]:
    x = lr_up[:1]
    for i in range(n_steps, 0, -1):
        idx = int(round(i * num_steps / n_steps))
        idx = min(idx, num_steps)
        idx_prev = int(round((i - 1) * num_steps / n_steps))
        idx_prev = max(idx_prev, 0)
        t_curr = t_steps[idx].item()
        t_next = t_steps[idx_prev].item()

        with torch.no_grad():
            def vf(x_t, t_t):
                return model(x_t, t_t, lr[:1])
            x = torch.clamp(solver.step(vf, x, t_curr, t_next), 0.0, 1.0)
    psnr_n, _ = batch_psnr_ssim(x, hr[:1])
    print(f"   ODE {n_steps:2d}步: PSNR={psnr_n:.2f}")

print(f"\n{'='*60}")
print("4. 均匀时间步检查")
t_vals = t_steps.cpu().tolist()
print(f"   步数={num_steps}")
print(f"   前5步: {[f'{x:.4f}' for x in t_vals[:5]]}")
print(f"   后5步: {[f'{x:.4f}' for x in t_vals[-5:]]}")
diffs = [abs(t_vals[i+1] - t_vals[i]) for i in range(len(t_vals)-1)]
print(f"   步长: min={min(diffs):.4f} max={max(diffs):.4f} mean={sum(diffs)/len(diffs):.4f}")

print(f"\n{'='*60}")
print("5. 模型各通路贡献分析")
# Check: 关闭 DiT (Scond → zeros) 的影响
with torch.no_grad():
    t_one = torch.ones(B, device=device)
    x_t_1 = lr_up

    # 正常通路
    t_emb = timestep_embedding(t_one, 256)
    s_cond = model.dit(x_t_1, t_emb, lr)

    # 准备 lr_spatial
    _, _, H, W = x_t_1.shape
    lr_feat = model.lr_enc_stage1(lr)
    lr_feat = F.interpolate(lr_feat, scale_factor=2, mode="bicubic")
    lr_feat = model.lr_enc_stage2(lr_feat)
    lr_feat = F.interpolate(lr_feat, size=(H, W), mode="bicubic")
    lr_spatial = model.lr_enc_stage3(lr_feat)

    v_full = model.pit(x_t_1, s_cond, lr_spatial=lr_spatial)
    v_noscond = model.pit(x_t_1, torch.zeros_like(s_cond), lr_spatial=lr_spatial)
    diff_ratio = (v_full - v_noscond).abs().mean() / (v_full.abs().mean() + 1e-8)
    print(f"   DiT Scond 通路贡献: Δv/v = {diff_ratio:.4f} (越大越重要)")

    # 单步推理 PSNR（有/无 DiT 条件）
    sr_cond = model.predict_single_step(lr_up, lr)
    sr_uncond = torch.clamp(lr_up - v_noscond, 0.0, 1.0)
    psnr_cond, _ = batch_psnr_ssim(sr_cond, hr)
    psnr_uncond, _ = batch_psnr_ssim(sr_uncond, hr)
    print(f"   单步 PSNR (有 DiT 条件): {psnr_cond:.2f} dB")
    print(f"   单步 PSNR (无 DiT 条件): {psnr_uncond:.2f} dB")

print(f"\n{'='*60}")
print("6. Bicubic 基线")
bicubic_psnr, _ = batch_psnr_ssim(lr_up[:1], hr[:1])
print(f"   Bicubic PSNR: {bicubic_psnr:.2f} dB")

print(f"\n{'='*60}")
print("7. AdaLN Gate 状态抽样")
# 检查 DiT 第一个 block 的 gate
block0 = model.dit.blocks[0]
c = model.dit.cond_proj(torch.cat([
    model.dit.time_mlp(timestep_embedding(t_one, 256)),
    model.dit.lr_encoder(lr),
], dim=-1))
params1 = block0.adaLN_proj1(c)
_, _, gate_alpha1 = params1.chunk(3, dim=-1)
print(f"   DiT Block0 gate: mean={gate_alpha1.mean():.4f} std={gate_alpha1.std():.4f}")

# PiT 第一个 block
pit_block0 = model.pit.blocks[0]
x_pit = model.pit.pixel_embed(x_t_1)
x_nchw = x_pit
x_nhwc = x_nchw.permute(0, 2, 3, 1)
params1_pit = pit_block0.adaln1(model.dit(x_t_1, timestep_embedding(t_one, 256), lr), 16, 16)
gate1_pit = params1_pit[..., 2, :]
print(f"   PiT Block0 gate: mean={gate1_pit.mean():.4f} std={gate1_pit.std():.4f}")

print(f"\n{'='*60}")
print("8. ODE 积分轨迹追踪")
# 追踪前几步 v_pred 的幅值
x_trace = lr_up[:1]
for i in range(num_steps, num_steps - 5, -1):
    t_curr = t_steps[i].item()
    t_next = t_steps[i - 1].item()

    with torch.no_grad():
        def vf2(x_t, t_t):
            return model(x_t, t_t, lr[:1])
        k1_val = vf2(x_trace, torch.full((1,), t_curr, device=device))
        x_trace = torch.clamp(solver.step(vf2, x_trace, t_curr, t_next), 0.0, 1.0)
    print(f"   步{i:2d} t={t_curr:.4f}: |k1|={k1_val.abs().mean():.4f} x_range=[{x_trace.min():.3f},{x_trace.max():.3f}]")
