"""Per-budget diffusion augmentation for the Track 9 grand ensemble.

A light, 10-class class-conditional DDPM is trained on the labeled budget subset, then
used to generate a synthetic image bank. The bank is concatenated with the real budget
images to train the ensemble's "diffusion" CNN member (a CNNMember).

This reuses the Track 5 model (src/diffusion/conditional.ConditionalGaussianDiffusion)
with num_classes=10 and leaves the existing 17-class CultiVar scripts untouched.

A DDPM cannot learn a usable generator from a handful of images, so the Track 9 harness
only invokes this above a budget threshold (default n >= 500). Even there the generator
is light (small U-Net, few timesteps/epochs) — the bank quality is a documented caveat.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from denoising_diffusion_pytorch import Unet
from src.common.utils import set_seed
from src.diffusion.conditional import ConditionalGaussianDiffusion

NUM_CLASSES = 10


def _build(dim: int, timesteps: int, sampling_steps: int) -> ConditionalGaussianDiffusion:
    unet = Unet(dim=dim, channels=1, dim_mults=(1, 2, 4))
    return ConditionalGaussianDiffusion(
        unet,
        num_classes=NUM_CLASSES,
        image_size=28,
        timesteps=timesteps,
        sampling_timesteps=min(sampling_steps, timesteps),
    )


def train_and_generate(
    images_u8: np.ndarray,
    labels: np.ndarray,
    device,
    n_per_class: int = 200,
    dim: int = 16,
    timesteps: int = 250,
    sampling_steps: int = 50,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 2e-4,
    seed: int = 0,
    verbose: bool = False,
):
    """Train a light 10-class DDPM on (images_u8, labels) and return a generated bank.

    Returns (gen_images_u8 (M,28,28) uint8, gen_labels (M,) int64), M = n_per_class*10.
    fp32 only (AMP is NaN-prone on Turing).
    """
    set_seed(seed)
    model = _build(dim, timesteps, sampling_steps).to(device)

    X = torch.tensor(images_u8.astype(np.float32) / 255.0).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for imgs, classes in loader:
            imgs, classes = imgs.to(device), classes.to(device)
            opt.zero_grad()
            loss = model(imgs, classes)
            loss.backward()
            opt.step()
            tot += loss.item()
        if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
            print(f"    [ddpm] epoch {ep}/{epochs} loss={tot/max(len(loader),1):.4f}", flush=True)

    # ---- generate the bank ----
    model.eval()
    gen_imgs, gen_lbls = [], []
    for cls in range(NUM_CLASSES):
        remaining = n_per_class
        while remaining > 0:
            b = min(remaining, batch_size)
            classes = torch.full((b,), cls, dtype=torch.long, device=device)
            samples = model.sample(classes).squeeze(1).clamp(0, 1).cpu().numpy()
            gen_imgs.append((samples * 255).astype(np.uint8))
            gen_lbls.append(np.full(b, cls, dtype=np.int64))
            remaining -= b
    return np.concatenate(gen_imgs, axis=0), np.concatenate(gen_lbls, axis=0)
