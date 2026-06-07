"""从 config.yaml 加载配置，支持 smoke 模式覆盖。"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "config/config.yaml", smoke: bool = False) -> Dict[str, Any]:
    """加载配置。若 smoke=True，用 smoke 字段覆盖主配置。"""
    root = Path(__file__).resolve().parent.parent
    full_path = root / config_path

    if not full_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if smoke:
        smoke_cfg = cfg.get("smoke", {})
        if "model" in smoke_cfg:
            cfg["model"].update(smoke_cfg["model"])
        if "training" in smoke_cfg:
            cfg["training"].update(smoke_cfg["training"])

    return cfg


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个配置字典。"""
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
