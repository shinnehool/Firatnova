"""项目入口：使用 tf 环境 Python 运行所有脚本。

用法（从项目根目录）:
    python run.py train            → 正式训练
    python run.py train_fast       → 快速验证训练
    python run.py inference        → 推理
    python run.py smoke            → smoke 测试
    python run.py diagnose         → 诊断脚本

所有命令自动使用 D:\Anaconda\envs\tf\python.exe
（PyTorch 2.11.0+cu128，支持 RTX 5060 Ti sm_120）。
"""

import subprocess
import sys
from pathlib import Path

PYTHON = r"D:\Anaconda\envs\tf\python.exe"
PROJECT = Path(__file__).resolve().parent

SCRIPTS = {
    "train":       "train/train.py",
    "train_fast":  "train/train_fast.py",
    "inference":   "inference/inference.py",
    "smoke":       "smoke/smoke.py",
    "diagnose":    "diagnose.py",
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        print("用法: python run.py <command> [extra args...]")
        print("可用命令:")
        for name, path in SCRIPTS.items():
            print(f"  {name:<12} → {path}")
        sys.exit(1)

    cmd = sys.argv[1]
    extra = sys.argv[2:]
    script = str(PROJECT / SCRIPTS[cmd])

    args = [PYTHON, script] + extra
    print(f"Run: {' '.join(args)}", file=sys.stderr)
    sys.exit(subprocess.call(args))
