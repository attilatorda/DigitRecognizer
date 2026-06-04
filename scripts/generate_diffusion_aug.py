"""
Generate augmented training images for all 17 CultiVar classes using the
trained class-conditional DDPM (DDIM sampling).

Usage:
    python scripts/generate_diffusion_aug.py \\
        --checkpoint experiments/checkpoints/diffusion/phase2_final.pt \\
        --n-per-class 256 --batch-size 64
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from denoising_diffusion_pytorch import Unet
from src.common.utils import ensure_dir
from src.diffusion.conditional import ConditionalGaussianDiffusion
from src.variants17.label_schema import LABELS_17


NUM_CLASSES = 17


def build_model(dim: int = 32, timesteps: int = 1000) -> ConditionalGaussianDiffusion:
    unet = Unet(dim=dim, channels=1, dim_mults=(1, 2, 4))
    sampling_timesteps = min(100, timesteps)
    return ConditionalGaussianDiffusion(
        unet, num_classes=NUM_CLASSES, image_size=28,
        timesteps=timesteps, sampling_timesteps=sampling_timesteps,
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] device={device}  n_per_class={args.n_per_class}")

    ckpt_path = os.path.join(ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    model = build_model(dim=args.dim, timesteps=args.timesteps).to(device)
    model.load(ckpt_path, device)
    model.eval()
    print(f"[gen] loaded {ckpt_path}")

    out_dir = os.path.join(ROOT, args.out_dir)
    ensure_dir(out_dir)

    all_images, all_labels = [], []

    for cls_id in range(NUM_CLASSES):
        remaining = args.n_per_class
        cls_images = []
        print(f"[gen] class {cls_id:2d} ({LABELS_17[cls_id]:<14}) — generating {remaining} samples ...")

        while remaining > 0:
            batch = min(remaining, args.batch_size)
            classes = torch.full((batch,), cls_id, dtype=torch.long, device=device)
            samples = model.sample(classes)          # (batch, 1, 28, 28) float in [0,1]
            samples = samples.squeeze(1).cpu().numpy().astype(np.float32)
            cls_images.append(samples)
            remaining -= batch

        cls_arr = np.concatenate(cls_images, axis=0)[:args.n_per_class]
        all_images.append(cls_arr)
        all_labels.append(np.full(len(cls_arr), cls_id, dtype=np.int64))
        print(f"[gen]   -> {cls_arr.shape}  min={cls_arr.min():.3f}  max={cls_arr.max():.3f}")

    images_out = np.concatenate(all_images, axis=0)
    labels_out = np.concatenate(all_labels, axis=0)

    img_path = os.path.join(out_dir, "generated_images.npy")
    lbl_path = os.path.join(out_dir, "generated_labels.npy")
    np.save(img_path, images_out)
    np.save(lbl_path, labels_out)
    print(f"\n[gen] saved {images_out.shape} images -> {img_path}")
    print(f"[gen] saved {labels_out.shape} labels -> {lbl_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="experiments/checkpoints/diffusion/phase2_final.pt")
    p.add_argument("--n-per-class", type=int,   default=256)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--dim",         type=int,   default=32)
    p.add_argument("--timesteps",   type=int,   default=1000)
    p.add_argument("--out-dir",     default="experiments/checkpoints/diffusion_aug")
    main(p.parse_args())
