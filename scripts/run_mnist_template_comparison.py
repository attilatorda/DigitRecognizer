"""
Random-MNIST template comparison experiment.

Validates CultiVar-17 template quality by replacing the hand-drawn templates with
17 randomly selected MNIST training images (same class structure, 5 seeds).

Runs three configurations per seed:
  mnist_nearest   — L2 nearest-template, no training
  mnist_no_aug    — SimpleCNN, no elastic/stroke augmentation
  mnist_proto     — EmbeddingCNN + prototype embedding (full augmentation)

Compares results to the pre-computed CultiVar-17 baselines from oneshot_results.json.

Usage:
    python scripts/run_mnist_template_comparison.py
    python scripts/run_mnist_template_comparison.py --seeds 5 --epochs 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir, set_seed
from src.variants17.label_schema import CLASS17_TO_DIGIT10, LABELS_17
from run_oneshot_experiment import (
    nearest_template_accuracy,
    train_cnn_one_seed,
    train_proto_one_seed,
)


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------

def select_mnist_templates(train_images, train_labels, seed):
    """
    Draw 17 MNIST training images following CultiVar-17 class structure.
    CLASS17_TO_DIGIT10 maps each of the 17 slots to a digit (0-9).
    Digits with multiple slots (0,1,2,4,7,9) each receive a different image.
    """
    rng = np.random.default_rng(seed)
    digit_pool = {d: list(np.where(train_labels == d)[0]) for d in range(10)}
    for pool in digit_pool.values():
        rng.shuffle(pool)
    cursor = {d: 0 for d in range(10)}
    templates = []
    for class_id in range(17):
        digit = CLASS17_TO_DIGIT10[class_id]
        templates.append(train_images[digit_pool[digit][cursor[digit]]])
        cursor[digit] += 1
    return np.stack(templates)   # (17, 28, 28) uint8


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seeds))
    labels17 = np.arange(17, dtype=np.int64)

    print(f"[mnist_cmp] device={device}  seeds={seeds}  epochs={args.epochs}")

    # Load MNIST once
    test_images, test_labels = load_mnist_idx(args.mnist_path, "t10k")
    train_images, train_labels = load_mnist_idx(args.mnist_path, "train")
    print(f"[mnist_cmp] MNIST train={len(train_images)}  test={len(test_images)}")

    # Load CultiVar-17 baselines from existing results
    cultivar_nearest = None
    cultivar_proto_mean = cultivar_proto_std = None
    cultivar_no_aug_mean = cultivar_no_aug_std = None
    results_path = os.path.join(args.report_dir, "oneshot_results.json")
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            existing = json.load(f)
        cultivar_nearest = existing["nearest_template"]["test_acc"]
        for cfg in existing["configs"]:
            if cfg["name"] == "proto":
                cultivar_proto_mean = cfg["mean_test_acc"]
                cultivar_proto_std = cfg["std_test_acc"]
            if cfg["name"] == "no_aug_cnn":
                cultivar_no_aug_mean = cfg["mean_test_acc"]
                cultivar_no_aug_std = cfg["std_test_acc"]
        print(f"[mnist_cmp] CultiVar-17 baselines loaded: "
              f"nearest={cultivar_nearest*100:.2f}%  "
              f"no_aug={cultivar_no_aug_mean*100:.2f}%  "
              f"proto={cultivar_proto_mean*100:.2f}%")

    shared = dict(
        test_images=test_images, test_labels=test_labels,
        train_images_mnist=train_images, train_labels_mnist=train_labels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device,
    )

    per_seed_results = []
    nearest_accs, no_aug_accs, proto_accs = [], [], []

    for seed in seeds:
        print(f"\n[mnist_cmp] === seed={seed} ===")
        templates = select_mnist_templates(train_images, train_labels, seed)
        set_seed(seed)

        # 1. Nearest-template (no training)
        t0 = time.perf_counter()
        nt_acc = nearest_template_accuracy(templates, test_images, test_labels)
        print(f"  nearest   {nt_acc*100:.2f}%  ({time.perf_counter()-t0:.1f}s)")
        nearest_accs.append(nt_acc)

        # 2. No-aug CNN
        no_aug_acc, _, _ = train_cnn_one_seed(
            seed=seed, templates_u8=templates, labels17=labels17,
            elastic_prob=0.0, stroke_prob=0.0,
            out_dir=os.path.join(args.out_dir, f"mnist_no_aug_seed{seed}"),
            **shared,
        )
        print(f"  no_aug    {no_aug_acc*100:.2f}%")
        no_aug_accs.append(no_aug_acc)

        # 3. Proto (full augmentation)
        proto_acc, _, _ = train_proto_one_seed(
            seed=seed, templates_u8=templates, labels17=labels17,
            elastic_prob=args.elastic_prob, stroke_prob=args.stroke_prob,
            emb_dim=args.emb_dim,
            out_dir=os.path.join(args.out_dir, f"mnist_proto_seed{seed}"),
            **shared,
        )
        print(f"  proto     {proto_acc*100:.2f}%")
        proto_accs.append(proto_acc)

        per_seed_results.append({
            "seed": seed,
            "nearest": float(nt_acc),
            "no_aug_cnn": float(no_aug_acc),
            "proto": float(proto_acc),
        })

    # Summary
    summary = {
        "mnist_nearest_mean": float(np.mean(nearest_accs)),
        "mnist_nearest_std":  float(np.std(nearest_accs)),
        "mnist_no_aug_mean":  float(np.mean(no_aug_accs)),
        "mnist_no_aug_std":   float(np.std(no_aug_accs)),
        "mnist_proto_mean":   float(np.mean(proto_accs)),
        "mnist_proto_std":    float(np.std(proto_accs)),
    }

    payload = {
        "seeds": seeds,
        "cultivar17_nearest":  {"test_acc": cultivar_nearest},
        "cultivar17_no_aug":   {"mean": cultivar_no_aug_mean, "std": cultivar_no_aug_std},
        "cultivar17_proto":    {"mean": cultivar_proto_mean,  "std": cultivar_proto_std},
        "per_seed": per_seed_results,
        "summary": summary,
    }

    ensure_dir(args.report_dir)
    out_path = os.path.join(args.report_dir, "mnist_template_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[mnist_cmp] saved {out_path}")

    # Comparison table
    print("\n=== TEMPLATE QUALITY COMPARISON ===")
    print(f"{'Method':<28} {'CultiVar-17':>14} {'Random MNIST':>14}")
    print("-" * 58)
    if cultivar_nearest is not None:
        print(f"{'Nearest-template (L2)':<28} "
              f"{cultivar_nearest*100:>13.2f}% "
              f"{summary['mnist_nearest_mean']*100:>13.2f}%"
              f"  +-{summary['mnist_nearest_std']*100:.2f}")
    if cultivar_no_aug_mean is not None:
        print(f"{'No-aug CNN':<28} "
              f"{cultivar_no_aug_mean*100:>13.2f}% "
              f"{summary['mnist_no_aug_mean']*100:>13.2f}%"
              f"  +-{summary['mnist_no_aug_std']*100:.2f}")
    if cultivar_proto_mean is not None:
        print(f"{'Proto embedding':<28} "
              f"{cultivar_proto_mean*100:>13.2f}% "
              f"{summary['mnist_proto_mean']*100:>13.2f}%"
              f"  +-{summary['mnist_proto_std']*100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path",   default="mnist_data")
    parser.add_argument("--out-dir",      default="experiments/checkpoints/mnist_template_comparison")
    parser.add_argument("--report-dir",   default="experiments/reports")
    parser.add_argument("--seeds",        type=int, default=5)
    parser.add_argument("--epochs",       type=int, default=8)
    parser.add_argument("--batch-size",   type=int, default=128)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--emb-dim",      type=int, default=64)
    parser.add_argument("--elastic-prob", type=float, default=0.70)
    parser.add_argument("--stroke-prob",  type=float, default=0.80)
    main(parser.parse_args())
