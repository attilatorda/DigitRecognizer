"""
Train a class-conditional DDPM for CultiVar-17 augmented image generation.

Two training phases:

  Phase 1 — MNIST images for single-variant digits (3, 5, 6, 8):
      python scripts/train_diffusion.py --phase 1 --epochs 200

  Phase 2 — Fine-tune on CultiVar-17 augmented set (all 17 classes):
      python scripts/train_diffusion.py --phase 2 \\
          --resume experiments/checkpoints/diffusion/phase1_final.pt \\
          --epochs 100
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from denoising_diffusion_pytorch import Unet
from src.common.utils import ensure_dir, set_seed
from src.diffusion.conditional import ConditionalGaussianDiffusion
from src.diffusion.dataset import Grayscale28Dataset, prepare_mnist_phase1
from src.variants17.label_schema import CLASS17_TO_DIGIT10

# CultiVar-17 single-variant class IDs (digits 3, 5, 6, 8 → no style ambiguity)
PHASE1_CLASS_IDS = [10, 12, 13, 15]  # 3, 5, 6, 8


def build_model(num_classes: int, dim: int = 32) -> ConditionalGaussianDiffusion:
    unet = Unet(dim=dim, channels=1, dim_mults=(1, 2, 4))
    return ConditionalGaussianDiffusion(
        unet,
        num_classes=num_classes,
        image_size=28,
        timesteps=1000,
        sampling_timesteps=100,
    )


def train_one_epoch(model, loader, opt, device, scaler=None):
    model.train()
    total = 0.0
    for imgs, classes in loader:
        imgs, classes = imgs.to(device), classes.to(device)
        opt.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = model(imgs, classes)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = model(imgs, classes)
            loss.backward()
            opt.step()
        total += loss.item()
    return total / max(len(loader), 1)


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diffusion] device={device}  phase={args.phase}  epochs={args.epochs}")

    ckpt_dir = os.path.join(ROOT, "experiments", "checkpoints", "diffusion")
    ensure_dir(ckpt_dir)

    # ------------------------------------------------------------------ data
    if args.phase == 1:
        phase1_data = os.path.join(ROOT, "data", "diffusion", "mnist_phase1")
        if not os.path.isdir(phase1_data):
            print("[diffusion] Preparing MNIST Phase 1 data ...")
            prepare_mnist_phase1(
                mnist_path=os.path.join(ROOT, "mnist_data"),
                out_dir=phase1_data,
                class_ids=PHASE1_CLASS_IDS,
                digit_map=CLASS17_TO_DIGIT10,
                max_per_class=args.max_per_class,
            )
        dataset = Grayscale28Dataset(mode="folder", root=phase1_data, class_ids=PHASE1_CLASS_IDS)
        num_classes = 17  # build full 17-class model even in phase 1 so phase 2 can resume
        print(f"[diffusion] Phase 1 dataset: {len(dataset)} images")

    else:  # phase 2
        aug_images = os.path.join(ROOT, args.aug_images)
        aug_labels = os.path.join(ROOT, args.aug_labels)
        dataset = Grayscale28Dataset(mode="npy", images_path=aug_images, labels_path=aug_labels)
        num_classes = 17
        print(f"[diffusion] Phase 2 dataset: {len(dataset)} images")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ----------------------------------------------------------------- model
    model = build_model(num_classes=num_classes, dim=args.dim).to(device)

    if args.resume:
        resume_path = os.path.join(ROOT, args.resume) if not os.path.isabs(args.resume) else args.resume
        model.load(resume_path, device)
        print(f"[diffusion] Resumed from {resume_path}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    # ----------------------------------------------------------------- train
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, opt, device, scaler)
        print(f"[diffusion] epoch={epoch}/{args.epochs}  loss={loss:.4f}")

        if epoch % args.save_every == 0:
            ckpt = os.path.join(ckpt_dir, f"phase{args.phase}_epoch{epoch:04d}.pt")
            model.save(ckpt)
            print(f"[diffusion] saved {ckpt}")

    final = os.path.join(ckpt_dir, f"phase{args.phase}_final.pt")
    model.save(final)
    print(f"[diffusion] final checkpoint: {final}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase",         type=int,   default=1,    choices=[1, 2])
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--dim",           type=int,   default=32,   help="U-Net base dim")
    p.add_argument("--save-every",    type=int,   default=50)
    p.add_argument("--resume",        type=str,   default="",   help="Checkpoint path to resume from")
    p.add_argument("--max-per-class", type=int,   default=5000, help="Phase 1: max MNIST images per class")
    p.add_argument("--aug-images",    type=str,
                   default="experiments/checkpoints/variants17/augmented_train/images.npy")
    p.add_argument("--aug-labels",    type=str,
                   default="experiments/checkpoints/variants17/augmented_train/labels17.npy")
    main(p.parse_args())
