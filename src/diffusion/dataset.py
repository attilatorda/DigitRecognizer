"""Datasets for the diffusion augmentation track.

Grayscale28Dataset — loads (image, class_id) pairs from either:
  - mode='folder': one subdirectory per class containing PNG files
  - mode='npy':    a float32 .npy image array + int64 .npy label array

prepare_mnist_phase1 — extracts MNIST images for selected class IDs and saves
them as PNG files into a folder-per-class layout for Phase 1 training.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Grayscale28Dataset(Dataset):
    """
    Returns (image_tensor, class_id_tensor) pairs.

    image_tensor: float32 (1, 28, 28) in [0, 1]
    class_id_tensor: int64 scalar
    """

    def __init__(self, mode: str, **kwargs):
        if mode == "folder":
            self._init_folder(**kwargs)
        elif mode == "npy":
            self._init_npy(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'folder' or 'npy'.")

    def _init_folder(self, root: str, class_ids: list[int] | None = None):
        root = Path(root)
        images, labels = [], []
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            try:
                cls_id = int(cls_dir.name)
            except ValueError:
                continue
            if class_ids is not None and cls_id not in class_ids:
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                img = Image.open(img_path).convert("L").resize((28, 28), Image.NEAREST)
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(cls_id)
        self._images = np.stack(images, axis=0)   # (N, 28, 28) float32
        self._labels = np.array(labels, dtype=np.int64)
        print(f"[dataset] loaded {len(self._images)} images into RAM")

    def _init_npy(self, images_path: str, labels_path: str):
        images = np.load(images_path)
        labels = np.load(labels_path)
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0 if images.max() > 1.0 else images.astype(np.float32)
        self._images = images
        self._labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        x = torch.tensor(self._images[idx]).unsqueeze(0)
        return x, torch.tensor(self._labels[idx], dtype=torch.int64)


def prepare_mnist_phase1(
    mnist_path: str,
    out_dir: str,
    class_ids: list[int],
    digit_map: dict[int, int],
    max_per_class: int = 5000,
) -> None:
    """
    Save MNIST training images for the given CultiVar class IDs as PNG files.

    class_ids: CultiVar-17 class IDs (e.g. [10, 12, 13, 15] for 3, 5, 6, 8)
    digit_map: CLASS17_TO_DIGIT10 mapping
    """
    import sys
    root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root))
    from src.common.data_io import load_mnist_idx

    images, labels = load_mnist_idx(mnist_path, kind="train")

    out_dir = Path(out_dir)
    for cls_id in class_ids:
        digit = digit_map[cls_id]
        cls_out = out_dir / str(cls_id)
        cls_out.mkdir(parents=True, exist_ok=True)
        indices = np.where(labels == digit)[0][:max_per_class]
        for i, idx in enumerate(indices):
            img = Image.fromarray(images[idx], mode="L")
            img.save(cls_out / f"{i:05d}.png")
        print(f"[phase1] class {cls_id} (digit {digit}): saved {len(indices)} images -> {cls_out}")
