"""数据集与 DataLoader：从文件夹加载图像，创建 LR-HR 配对。

支持两种结构：
  1. 单目录（HR 图像，自动下采样生成 LR）
  2. 配对目录（HR/ 和 LR/ 子文件夹，按同名文件配对）
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class SuperResolutionDataset(Dataset):
    """医学图像超分数据集。

    自动检测 data_dir 下的 HR/ 和 LR/ 子文件夹：
    - 若存在：按同名文件配对加载
    - 若不存在：从 HR 图像 bicubic 下采样创建 LR
    """

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".bmp"}

    def __init__(
        self,
        data_dir: str,
        hr_size: int = 256,
        lr_size: int = 64,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.split = split
        self.paired = False

        root = Path(__file__).resolve().parent.parent
        data_path = root / data_dir
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件夹不存在: {data_path}")

        hr_sub = data_path / "HR"
        lr_sub = data_path / "LR"

        if hr_sub.is_dir() and lr_sub.is_dir():
            # ---- 配对模式 ----
            self.paired = True
            lr_files: dict = {}
            for ext in self.VALID_EXTENSIONS:
                for p in lr_sub.glob(f"*{ext}"):
                    lr_files[p.stem] = p

            self.hr_paths: List[Path] = []
            self.lr_paths: List[Path] = []
            for ext in self.VALID_EXTENSIONS:
                for hr_p in sorted(hr_sub.glob(f"*{ext}")):
                    stem = hr_p.stem
                    if stem in lr_files:
                        self.hr_paths.append(hr_p)
                        self.lr_paths.append(lr_files[stem])

            if not self.hr_paths:
                raise RuntimeError(f"在 {hr_sub} / {lr_sub} 中未找到配对图像")
        else:
            # ---- 单目录模式（从 HR 自动生成 LR）----
            seen = set()
            self.hr_paths = []
            for ext in self.VALID_EXTENSIONS:
                for p in sorted(data_path.glob(f"*{ext}")):
                    if p.stem not in seen:
                        seen.add(p.stem)
                        self.hr_paths.append(p)
            self.lr_paths = []  # 不使用

            if not self.hr_paths:
                raise RuntimeError(
                    f"在 {data_path} 中未找到任何图像文件。"
                    f"支持的格式: {self.VALID_EXTENSIONS}"
                )

        if max_samples is not None and max_samples > 0:
            self.hr_paths = self.hr_paths[:max_samples]
            if self.paired:
                self.lr_paths = self.lr_paths[:max_samples]

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _load_image(self, path: Path) -> torch.Tensor:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hr = self._load_image(self.hr_paths[idx])

        # 调整到统一 HR 尺寸
        _, h, w = hr.shape
        if h != self.hr_size or w != self.hr_size:
            hr = torch.clamp(TF.resize(hr, [self.hr_size, self.hr_size],
                             interpolation=TF.InterpolationMode.BICUBIC), 0.0, 1.0)

        if self.paired:
            lr = self._load_image(self.lr_paths[idx])
            _, lh, lw = lr.shape
            if lh != self.lr_size or lw != self.lr_size:
                lr = torch.clamp(TF.resize(lr, [self.lr_size, self.lr_size],
                                 interpolation=TF.InterpolationMode.BICUBIC), 0.0, 1.0)
        else:
            lr = torch.clamp(TF.resize(hr, [self.lr_size, self.lr_size],
                             interpolation=TF.InterpolationMode.BICUBIC), 0.0, 1.0)

        # ---- 数据增强（训练时；对 HR 和 LR 同步）----
        if self.split == "train":
            # 随机水平翻转
            if torch.rand(1).item() < 0.5:
                hr = TF.hflip(hr)
                lr = TF.hflip(lr)
            # 随机垂直翻转
            if torch.rand(1).item() < 0.5:
                hr = TF.vflip(hr)
                lr = TF.vflip(lr)
            # 随机 90° 旋转
            if torch.rand(1).item() < 0.5:
                k = torch.randint(0, 4, (1,)).item()
                hr = torch.rot90(hr, k, dims=(-2, -1))
                lr = torch.rot90(lr, k, dims=(-2, -1))

        lr_up = torch.clamp(TF.resize(lr, [self.hr_size, self.hr_size],
                            interpolation=TF.InterpolationMode.BICUBIC), 0.0, 1.0)

        return lr, lr_up, hr


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    hr_size: int = 256,
    lr_size: int = 64,
    batch_size: int = 8,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    val_shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader。"""
    train_dataset = SuperResolutionDataset(
        data_dir=train_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        split="train",
        max_samples=max_train_samples,
    )
    val_dataset = SuperResolutionDataset(
        data_dir=val_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        split="val",
        max_samples=max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证时使用 batch=1 便于计算指标
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
