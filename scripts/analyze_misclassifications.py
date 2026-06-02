"""
Misclassification analysis for the best proto one-shot model.

Loads the best proto checkpoint (seed 0, ~82% MNIST accuracy), runs full inference
on the MNIST test set, and produces:
  - Per-digit accuracy breakdown
  - 10×10 confusion matrix
  - Top confused digit pairs with counts and percentages
  - experiments/reports/misclassification_analysis.json
  - experiments/reports/figures/fig4_confusion.png

Usage:
    python scripts/analyze_misclassifications.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.data_io import load_mnist_idx
from src.variants17.label_schema import CLASS17_TO_DIGIT10, LABELS_17
from src.variants17.train_variants17_proto import (
    EmbeddingCNN,
    compute_prototypes,
    get_mnist_predictions,
)


# ---------------------------------------------------------------------------
# Confusion matrix figure
# ---------------------------------------------------------------------------

def make_confusion_figure(cm, per_digit_acc, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel("Predicted digit", fontsize=11)
    ax.set_ylabel("True digit", fontsize=11)
    ax.set_title("Confusion matrix — proto (best seed)", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")

    # Right: per-digit accuracy bar chart, sorted worst -> best
    ax2 = axes[1]
    digits_sorted = sorted(per_digit_acc, key=per_digit_acc.get)
    vals = [per_digit_acc[d] * 100 for d in digits_sorted]
    colors = ["#c0392b" if v < 70 else "#e67e22" if v < 85 else "#27ae60"
              for v in vals]
    bars = ax2.barh(range(10), vals, color=colors)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f"Digit {d}" for d in digits_sorted])
    ax2.set_xlabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Per-digit accuracy", fontsize=12)
    ax2.set_xlim(0, 105)
    ax2.axvline(np.mean(vals), color="navy", linewidth=1.2, linestyle="--", alpha=0.6)
    for bar, val in zip(bars, vals):
        ax2.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=8)
    ax2.yaxis.grid(False)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[analysis] saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[analysis] device={device}")

    # Paths
    ckpt_path      = os.path.join(ROOT, "experiments", "checkpoints",
                                  "oneshot_comparison", "proto", "best_seed0.pt")
    templates_path = os.path.join(ROOT, "data", "processed", "mnist17_variants",
                                  "train_images.npy")
    labels17_path  = os.path.join(ROOT, "data", "processed", "mnist17_variants",
                                  "train_labels17.npy")
    mnist_path     = os.path.join(ROOT, "mnist_data")
    out_json       = os.path.join(ROOT, "experiments", "reports",
                                  "misclassification_analysis.json")
    out_fig        = os.path.join(ROOT, "experiments", "reports", "figures",
                                  "fig4_confusion.png")

    # Load model
    model = EmbeddingCNN(emb_dim=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # run_oneshot_experiment saves the state dict directly (not wrapped in {"model": ...})
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"[analysis] loaded checkpoint {ckpt_path}")

    # Load templates -> prototypes
    templates_u8 = np.load(templates_path)
    labels17     = np.load(labels17_path)
    support_x01  = templates_u8.astype(np.float32) / 255.0
    prototypes   = compute_prototypes(model, support_x01, labels17, device)

    # Load MNIST test set
    test_images, test_labels = load_mnist_idx(mnist_path, "t10k")
    print(f"[analysis] MNIST test images: {len(test_images)}")

    # Run inference
    pred10, pred17 = get_mnist_predictions(model, test_images, prototypes, device)
    overall_acc = float((pred10 == test_labels).mean())
    print(f"[analysis] overall accuracy: {overall_acc*100:.2f}%")

    # 10×10 confusion matrix
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(test_labels, pred10):
        cm[int(t), int(p)] += 1

    # Per-digit accuracy
    per_digit_acc = {}
    for d in range(10):
        row_sum = cm[d].sum()
        per_digit_acc[d] = int(cm[d, d]) / int(row_sum) if row_sum > 0 else 0.0

    print("\n[analysis] Per-digit accuracy:")
    for d in range(10):
        print(f"  digit {d}: {per_digit_acc[d]*100:.1f}%  "
              f"(correct={cm[d,d]} / total={cm[d].sum()})")

    # Top confused pairs (off-diagonal)
    confused_pairs = []
    for t in range(10):
        for p in range(10):
            if t != p and cm[t, p] > 0:
                confused_pairs.append({
                    "true": int(t),
                    "pred": int(p),
                    "count": int(cm[t, p]),
                    "pct": round(float(cm[t, p]) / len(test_labels) * 100, 3),
                })
    confused_pairs.sort(key=lambda x: -x["count"])
    top10 = confused_pairs[:10]

    print("\n[analysis] Top confused pairs (true -> predicted):")
    for pair in top10:
        print(f"  {pair['true']} -> {pair['pred']:>2}  "
              f"count={pair['count']:>4}  ({pair['pct']:.2f}% of test set)")

    # Save JSON
    payload = {
        "checkpoint": "proto/best_seed0.pt",
        "overall_accuracy": overall_acc,
        "per_digit_accuracy": {str(d): round(per_digit_acc[d], 4) for d in range(10)},
        "confusion_matrix": cm.tolist(),
        "top_confused_pairs": top10,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[analysis] saved {out_json}")

    # Figure
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    make_confusion_figure(cm, per_digit_acc, out_fig)

    # Suggest improvements
    print("\n[analysis] Improvement hints (worst digits):")
    worst = sorted(per_digit_acc, key=per_digit_acc.get)[:3]
    for d in worst:
        misses = [(p, cm[d, p]) for p in range(10) if p != d and cm[d, p] > 0]
        misses.sort(key=lambda x: -x[1])
        top_miss = [(int(p), int(n)) for p, n in misses[:2]]
        variants = [LABELS_17[c] for c, digit in CLASS17_TO_DIGIT10.items() if digit == d]
        miss_str = ", ".join(f"{p} ({n}x)" for p, n in top_miss)
        print(f"  Digit {d} ({per_digit_acc[d]*100:.1f}%): "
              f"confused with {miss_str} -- "
              f"CultiVar variants: {variants}")


if __name__ == "__main__":
    main()
