"""
Visual + statistical inspection of the augmentation pipeline.

Generates a grid image showing each augmentation mode applied to a chosen
template, and prints pixel-distribution stats compared to MNIST ground truth.

Usage:
    python scripts/inspect_augmentations.py
    python scripts/inspect_augmentations.py --template-idx 3 --n-per-mode 8
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.data_io import load_mnist_idx
from src.variants17.augment import (
    apply_augmentation,
    elastic_distort,
    sample_augmentation_params,
    stroke_width_adjust,
)
from src.variants17.label_schema import LABELS_17


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255, 0, 255).astype(np.uint8)


def _stats(imgs: list[np.ndarray], label: str):
    arr = np.stack(imgs)  # (N, 28, 28), float32 [0,1]
    mean = arr.mean()
    std = arr.std()
    nonzero = arr[arr > 0.05]
    nz_frac = len(nonzero) / arr.size
    nz_mean = nonzero.mean() if len(nonzero) > 0 else 0.0
    print(f"  {label:<30s}  mean={mean:.3f}  std={std:.3f}  nonzero_frac={nz_frac:.3f}  nonzero_mean={nz_mean:.3f}")


def make_grid(columns: list[list[np.ndarray]], cell_size: int = 28, pad: int = 2) -> Image.Image:
    n_cols = len(columns)
    n_rows = max(len(c) for c in columns)
    w = n_cols * cell_size + (n_cols + 1) * pad
    h = n_rows * cell_size + (n_rows + 1) * pad
    grid = Image.new("L", (w, h), color=128)
    for col_idx, col in enumerate(columns):
        for row_idx, img01 in enumerate(col):
            x = pad + col_idx * (cell_size + pad)
            y = pad + row_idx * (cell_size + pad)
            cell = Image.fromarray(_to_uint8(img01))
            grid.paste(cell, (x, y))
    return grid


def main(args):
    templates_u8 = np.load(os.path.join(args.data_dir, "train_images.npy"))
    template = templates_u8[args.template_idx].astype(np.float32) / 255.0
    label = LABELS_17[args.template_idx]
    print(f"Template {args.template_idx}: {label}")

    mnist_images, _ = load_mnist_idx(args.mnist_path, "t10k")
    mnist_sample = mnist_images[:200].astype(np.float32) / 255.0

    rng = np.random.default_rng(args.seed)
    n = args.n_per_mode

    # --- build columns ---
    cols = {}

    # Original (repeated)
    cols["original"] = [template] * n

    # Rotation only
    def _rotation_only(img):
        p = sample_augmentation_params(rng, elastic_prob=0.0, stroke_prob=0.0)
        p["do_elastic"] = False
        p["stroke_mode"] = "none"
        return apply_augmentation(img, p, rng)
    cols["rotation_only"] = [_rotation_only(template) for _ in range(n)]

    # Elastic mild
    cols["elastic_mild"] = [
        elastic_distort(template, alpha=15.0, sigma=4.0, rng=rng) for _ in range(n)
    ]

    # Elastic strong
    cols["elastic_strong"] = [
        elastic_distort(template, alpha=34.0, sigma=4.0, rng=rng) for _ in range(n)
    ]

    # Stroke disk_dilate
    cols["disk_dilate_r1"] = [
        stroke_width_adjust(template, "disk_dilate", radius=1) for _ in range(n)
    ]

    # Stroke disk_erode
    cols["disk_erode_r1"] = [
        stroke_width_adjust(template, "disk_erode", radius=1) for _ in range(n)
    ]

    # Stroke blur_soft
    cols["blur_soft"] = [
        stroke_width_adjust(template, "blur_soft", blur_sigma=float(rng.uniform(0.6, 1.2))) for _ in range(n)
    ]

    # Full augmentation (all ops combined)
    def _full(img):
        p = sample_augmentation_params(rng, elastic_prob=args.elastic_prob, stroke_prob=args.stroke_prob)
        return apply_augmentation(img, p, rng)
    cols["full_aug"] = [_full(template) for _ in range(n)]

    # MNIST reference samples
    idx = rng.integers(0, len(mnist_sample), size=n)
    cols["mnist_ref"] = [mnist_sample[i] for i in idx]

    # --- print stats ---
    print("\nPixel distribution stats (float [0,1]):")
    for name, imgs in cols.items():
        _stats(imgs, name)

    # --- save grid ---
    os.makedirs(args.out_dir, exist_ok=True)
    grid = make_grid(list(cols.values()), cell_size=28, pad=2)
    # Scale up for readability
    scale = 4
    grid = grid.resize((grid.width * scale, grid.height * scale), Image.Resampling.NEAREST)
    out_path = os.path.join(args.out_dir, f"augmentation_preview_tmpl{args.template_idx:02d}.png")
    grid.save(out_path)

    col_names = " | ".join(cols.keys())
    print(f"\nColumns: {col_names}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/mnist17_variants")
    parser.add_argument("--mnist-path", default="mnist_data")
    parser.add_argument("--out-dir", default="experiments/reports")
    parser.add_argument("--template-idx", type=int, default=0)
    parser.add_argument("--n-per-mode", type=int, default=8)
    parser.add_argument("--elastic-prob", type=float, default=0.70)
    parser.add_argument("--stroke-prob", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
