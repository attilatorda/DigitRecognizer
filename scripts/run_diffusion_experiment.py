"""
One-shot digit recognition experiment using DDPM-generated training images.

Runs the same SimpleCNN + EmbeddingCNN proto pipeline as run_oneshot_experiment.py
but with DDPM-generated images as the training set instead of morphological augmentation.
Prints a comparison against the Variants17 baseline from oneshot_results.json.

Usage:
    python scripts/run_diffusion_experiment.py
    python scripts/run_diffusion_experiment.py --seeds 1 --epochs 1   # smoke-test
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir, set_seed
from src.variants17.label_schema import CLASS17_TO_DIGIT10

# Reuse training/eval helpers from the variants17 track
from run_oneshot_experiment import (
    nearest_template_accuracy,
    run_config,
    train_cnn_one_seed,
    train_proto_one_seed,
    write_markdown,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seeds))
    proto_seeds = list(range(args.proto_seeds))
    print(f"[diffusion_exp] device={device}  seeds={seeds}  proto_seeds={proto_seeds}")

    # Load generated training images
    img_path = os.path.join(ROOT, args.gen_images)
    lbl_path = os.path.join(ROOT, args.gen_labels)
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Generated images not found: {img_path}\n"
            "Run generate_diffusion_aug.py first."
        )

    gen_images = np.load(img_path)           # (N, 28, 28) float32 [0,1]
    gen_labels = np.load(lbl_path)           # (N,) int64

    # Convert float [0,1] → uint8 for compatibility with train_cnn_one_seed
    gen_images_u8 = (gen_images * 255).clip(0, 255).astype(np.uint8)
    print(f"[diffusion_exp] generated training set: {gen_images_u8.shape}")

    # Load MNIST test set
    test_images, test_labels = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    train_images_mnist, train_labels_mnist = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")

    # Load CultiVar-17 templates for nearest-template baseline + proto prototypes
    templates_u8 = np.load(os.path.join(ROOT, "data", "processed", "mnist17_variants", "train_images.npy"))

    out_dir = os.path.join(ROOT, "experiments", "checkpoints", "diffusion_experiment")
    report_dir = os.path.join(ROOT, "experiments", "reports")
    ensure_dir(out_dir)

    results = []

    # Nearest-template baseline (uses raw CultiVar templates, not generated images)
    import time
    t0 = time.perf_counter()
    nt_test  = nearest_template_accuracy(templates_u8, test_images,         test_labels)
    nt_train = nearest_template_accuracy(templates_u8, train_images_mnist,  train_labels_mnist)
    nearest  = {"test_acc": nt_test, "train_acc": nt_train, "time_s": time.perf_counter() - t0}
    print(f"[diffusion_exp] nearest_template  test={nt_test*100:.2f}%  train={nt_train*100:.2f}%")

    shared = dict(
        test_images=test_images, test_labels=test_labels,
        train_images_mnist=train_images_mnist, train_labels_mnist=train_labels_mnist,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device,
    )

    def _aug_dir(name):
        return os.path.join(out_dir, name, "augmented_train")

    # No-aug CNN (using DDPM images without further augmentation)
    results.append(run_config(
        "ddpm_no_aug_cnn",
        lambda seed, **kw: train_cnn_one_seed(
            seed=seed, templates_u8=gen_images_u8, labels17=gen_labels,
            elastic_prob=0.0, stroke_prob=0.0,
            out_dir=os.path.join(out_dir, "ddpm_no_aug_cnn"),
            save_augmented_dir=_aug_dir("ddpm_no_aug_cnn"), **kw,
        ),
        seeds, **shared,
    ))

    # Full-aug CNN (DDPM images + morphological augmentation on top)
    results.append(run_config(
        "ddpm_full_aug_cnn",
        lambda seed, **kw: train_cnn_one_seed(
            seed=seed, templates_u8=gen_images_u8, labels17=gen_labels,
            elastic_prob=0.70, stroke_prob=0.80,
            out_dir=os.path.join(out_dir, "ddpm_full_aug_cnn"),
            save_augmented_dir=_aug_dir("ddpm_full_aug_cnn"), **kw,
        ),
        seeds, **shared,
    ))

    # Proto embedding (DDPM images + morphological augmentation)
    results.append(run_config(
        "ddpm_proto",
        lambda seed, **kw: train_proto_one_seed(
            seed=seed, templates_u8=gen_images_u8, labels17=gen_labels,
            elastic_prob=0.70, stroke_prob=0.80,
            emb_dim=64,
            out_dir=os.path.join(out_dir, "ddpm_proto"),
            save_augmented_dir=_aug_dir("ddpm_proto"), **kw,
        ),
        proto_seeds, **shared,
    ))

    # Save results
    json_path = os.path.join(report_dir, "diffusion_experiment_results.json")
    payload = {"nearest_template": nearest, "configs": results, "local_cnn_acc": None}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[diffusion_exp] saved {json_path}")

    # Summary
    print("\n=== DIFFUSION EXPERIMENT SUMMARY ===")
    print(f"{'Config':<24} {'Test':>10} {'Train':>10}")
    print("-" * 48)
    print(f"{'nearest_template':<24} {nt_test*100:>9.2f}%  {nt_train*100:>9.2f}%")
    for r in results:
        print(f"{r['name']:<24} {r['mean_test_acc']*100:>9.2f}%  {r['mean_train_acc']*100:>9.2f}%")

    # Compare against Variants17 baseline
    baseline_path = os.path.join(report_dir, "oneshot_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_map = {c["name"]: c for c in baseline.get("configs", [])}
        name_pairs = [
            ("ddpm_no_aug_cnn",   "no_aug_cnn"),
            ("ddpm_full_aug_cnn", "full_aug_cnn"),
            ("ddpm_proto",        "proto"),
        ]
        print("\n=== VS VARIANTS17 BASELINE ===")
        print(f"{'Config':<24} {'Baseline':>12} {'DDPM':>12} {'Delta':>8}")
        print("-" * 60)
        for ddpm_name, base_name in name_pairs:
            ddpm_r = next((r for r in results if r["name"] == ddpm_name), None)
            base_r = baseline_map.get(base_name)
            if ddpm_r and base_r:
                b = base_r["mean_test_acc"] * 100
                d = ddpm_r["mean_test_acc"] * 100
                sign = "+" if d >= b else ""
                print(f"{ddpm_name:<24} {b:>11.2f}%  {d:>11.2f}%  {sign}{d-b:>6.2f}pp")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gen-images",  default="experiments/checkpoints/diffusion_aug/generated_images.npy")
    p.add_argument("--gen-labels",  default="experiments/checkpoints/diffusion_aug/generated_labels.npy")
    p.add_argument("--epochs",      type=int,   default=8)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seeds",       type=int,   default=3)
    p.add_argument("--proto-seeds", type=int,   default=5)
    main(p.parse_args())
