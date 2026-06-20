"""
Track 8 (#1): introspect the best digit net -- read its *technique* by visualization.

The reproducible MNIST record holder (An et al., github ansh941/MnistSimpleCNN) is a majority
vote of three deep, pooling-free CNNs with kernels 3/5/7. We activation-maximize each class for
each sub-model (M3/M5/M7) and for the shallow SimpleCNN baseline, and read off what the record
holder's design buys. Honest finding (depth, not kernel size): the deep record sub-models form
smooth, prototype-like canonical class images (mean total variation ~0.08), whereas the shallow
SimpleCNN produces high-frequency texture that barely resembles a digit (TV ~0.5). The three
sub-models' templates differ from one another, consistent with the decorrelation that makes
their ensemble work.

Usage:  python scripts/run_track8_introspect_best.py [--smoke]
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.data_io import load_mnist_idx
from src.common.utils import set_seed
from src.ensemble.members import CNNMember
from src.track9.record_models import RecordModel, train_record_model
from scripts.run_introspection_reconstruct import activation_maximize


def _tv(img):
    return float(np.abs(np.diff(img, axis=0)).mean() + np.abs(np.diff(img, axis=1)).mean())


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[best] device={device}", flush=True)
    raw_tr, y_tr = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    if args.smoke:
        raw_tr, y_tr = raw_tr[:6000], y_tr[:6000]
    ep = 2 if args.smoke else args.epochs
    set_seed(0)

    print("[best] training SimpleCNN + record-holder M3/M5/M7 ...", flush=True)
    models = {"SimpleCNN": CNNMember(device, epochs=min(8, ep)).fit(raw_tr, y_tr).model.eval()}
    for k in (3, 5, 7):
        m = RecordModel(k)
        train_record_model(m, raw_tr, y_tr, device, epochs=ep, seed=k)
        models[f"M{k}"] = m.eval()

    digits = [0, 1, 7] if args.smoke else list(range(10))
    recon = {name: [] for name in models}
    tvs = {name: [] for name in models}
    for name, model in models.items():
        for c in digits:
            img = activation_maximize(model, device, (lambda x, c=c: model(x)[0, c]),
                                      steps=120 if args.smoke else 300)
            recon[name].append(img); tvs[name].append(_tv(img))
        print(f"  {name:10s} mean canonical-image smoothness (TV) = {np.mean(tvs[name]):.4f}", flush=True)

    # grid: rows = models, cols = digits
    nrow, ncol = len(models), len(digits)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.0, nrow * 1.05))
    for r, name in enumerate(models):
        for cidx, c in enumerate(digits):
            ax = axes[r, cidx]
            ax.imshow(recon[name][cidx], cmap="gray"); ax.axis("off")
            if r == 0:
                ax.set_title(str(c), fontsize=10)
        axes[r, 0].set_ylabel(name, fontsize=10, rotation=90)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    fig.suptitle("Track 8 #1: canonical digits per model -- depth yields smooth, prototype-like "
                 "templates (record M3/M5/M7) vs high-frequency texture (shallow SimpleCNN)",
                 fontsize=10)
    plt.tight_layout()
    fp = os.path.join(ROOT, "experiments", "reports", "figures", "fig_track8_receptive_fields.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    plt.savefig(fp, dpi=170, facecolor="white"); plt.close()
    print(f"[best] saved {fp}")
    print("\n=== smoothness (TV) of canonical images, by model (lower = coarser/smoother) ===")
    for name in models:
        print(f"  {name:10s} {np.mean(tvs[name]):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    main(p.parse_args())
