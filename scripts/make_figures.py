"""
Generate all figures for the CultiVar-17 paper.

Outputs (in experiments/reports/figures/):
  fig1_templates.png      — 17-template grid with class labels
  fig2_augmentation.png   — augmentation gallery (modes × examples)
  fig3_results.png        — accuracy bar chart

Usage:
    python scripts/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.variants17.augment import apply_augmentation, elastic_distort, sample_augmentation_params, stroke_width_adjust
from src.variants17.label_schema import LABELS_17

OUT_DIR = os.path.join(ROOT, "experiments", "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

TEMPLATES_PATH = os.path.join(ROOT, "data", "processed", "mnist17_variants", "train_images.npy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_pil(arr01: np.ndarray, scale: int = 1) -> Image.Image:
    """Convert float32 [0,1] 28×28 array to PIL L image, optionally upscaled."""
    img = Image.fromarray((arr01 * 255).clip(0, 255).astype(np.uint8), mode="L")
    if scale > 1:
        img = img.resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)
    return img


def paste_with_label(canvas, img_pil, x, y, label, font, label_color=180):
    canvas.paste(img_pil, (x, y))
    draw = ImageDraw.Draw(canvas)
    label_y = y + img_pil.height + 2
    draw.text((x, label_y), label, fill=label_color, font=font)


# ---------------------------------------------------------------------------
# Figure 1 — Template grid
# ---------------------------------------------------------------------------

def make_fig1(templates_u8: np.ndarray, scale: int = 8, out_path: str = ""):
    n = len(templates_u8)           # 17
    cols = 6
    rows = (n + cols - 1) // cols  # 3

    cell_w = 28 * scale
    label_h = 14                   # pixels below each image for the label
    pad = 6
    gap = 10

    canvas_w = cols * cell_w + (cols - 1) * gap + 2 * pad
    canvas_h = rows * (cell_w + label_h + gap) + 2 * pad
    canvas = Image.new("L", (canvas_w, canvas_h), color=40)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for i, tmpl in enumerate(templates_u8):
        row = i // cols
        col = i % cols
        x = pad + col * (cell_w + gap)
        y = pad + row * (cell_w + label_h + gap)
        img_pil = to_pil(tmpl.astype(np.float32) / 255.0, scale=scale)
        label = LABELS_17[i].replace("_", " ")
        paste_with_label(canvas, img_pil, x, y, label, font)

    canvas.save(out_path)
    print(f"[figures] saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Augmentation gallery
# ---------------------------------------------------------------------------

def make_fig2(templates_u8: np.ndarray, template_idx: int = 0,
              n_per_col: int = 5, scale: int = 6, out_path: str = ""):
    rng = np.random.default_rng(7)
    base = templates_u8[template_idx].astype(np.float32) / 255.0

    # Full-aug: force mild stroke mode so no sample gets erased
    def _full_aug_sample(img, rng):
        p = sample_augmentation_params(rng, elastic_prob=0.85, stroke_prob=0.80)
        p["stroke_mode"] = rng.choice(["disk_dilate", "blur_soft", "none"])
        p["stroke_radius"] = 1
        return apply_augmentation(img, p, rng)

    columns = {
        "Original": [base] * n_per_col,
        "Elastic\n(mild)": [elastic_distort(base, 15.0, 4.0, rng) for _ in range(n_per_col)],
        "Elastic\n(strong)": [elastic_distort(base, 32.0, 3.5, rng) for _ in range(n_per_col)],
        "Stroke\ndilate": [stroke_width_adjust(base, "disk_dilate", radius=1) for _ in range(n_per_col)],
        "Stroke\nblur": [stroke_width_adjust(base, "blur_soft", blur_sigma=float(rng.uniform(0.8, 1.1))) for _ in range(n_per_col)],
        "Full\naug": [_full_aug_sample(base, rng) for _ in range(n_per_col)],
    }

    cell_w = 28 * scale
    header_h = 30
    pad = 6
    gap = 8

    n_cols = len(columns)
    canvas_w = n_cols * cell_w + (n_cols - 1) * gap + 2 * pad
    canvas_h = header_h + n_per_col * (cell_w + gap) + 2 * pad
    canvas = Image.new("L", (canvas_w, canvas_h), color=40)

    try:
        font_h = ImageFont.truetype("arial.ttf", 11)
        font_s = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font_h = ImageFont.load_default()
        font_s = font_h

    draw = ImageDraw.Draw(canvas)
    for col_idx, (col_label, imgs) in enumerate(columns.items()):
        x0 = pad + col_idx * (cell_w + gap)
        draw.text((x0 + 2, pad), col_label, fill=200, font=font_h)
        for row_idx, img01 in enumerate(imgs):
            y = header_h + pad + row_idx * (cell_w + gap)
            canvas.paste(to_pil(img01, scale=scale), (x0, y))

    canvas.save(out_path)
    print(f"[figures] saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Results bar chart
# ---------------------------------------------------------------------------

def make_fig3(results_path: str, out_path: str = ""):
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    nearest = data["nearest_template"]["test_acc"] * 100
    configs = {r["name"]: r for r in data["configs"]}
    supervised = (data.get("local_cnn_acc") or 0.9911) * 100

    has_ablation = "elastic_only_cnn" in configs and "stroke_only_cnn" in configs

    if has_ablation:
        labels = [
            "Nearest\ntemplate\n(L2)",
            "No-aug\nCNN",
            "Elastic\nonly",
            "Stroke\nonly",
            "Full-aug\nCNN",
            "Proto\nembedding",
        ]
        means = [
            nearest,
            configs["no_aug_cnn"]["mean_test_acc"] * 100,
            configs["elastic_only_cnn"]["mean_test_acc"] * 100,
            configs["stroke_only_cnn"]["mean_test_acc"] * 100,
            configs["full_aug_cnn"]["mean_test_acc"] * 100,
            configs["proto"]["mean_test_acc"] * 100,
        ]
        stds = [
            0,
            configs["no_aug_cnn"]["std_test_acc"] * 100,
            configs["elastic_only_cnn"]["std_test_acc"] * 100,
            configs["stroke_only_cnn"]["std_test_acc"] * 100,
            configs["full_aug_cnn"]["std_test_acc"] * 100,
            configs["proto"]["std_test_acc"] * 100,
        ]
        colors = ["#8ab4c9", "#b0cfe0", "#7aafcc", "#5a9fbc", "#4a90c4", "#1a5f9e"]
    else:
        labels = ["Nearest\ntemplate\n(L2)", "No-aug\nCNN", "Full-aug\nCNN", "Proto\nembedding"]
        means  = [nearest,
                  configs["no_aug_cnn"]["mean_test_acc"] * 100,
                  configs["full_aug_cnn"]["mean_test_acc"] * 100,
                  configs["proto"]["mean_test_acc"] * 100]
        stds   = [0,
                  configs["no_aug_cnn"]["std_test_acc"] * 100,
                  configs["full_aug_cnn"]["std_test_acc"] * 100,
                  configs["proto"]["std_test_acc"] * 100]
        colors = ["#8ab4c9", "#8ab4c9", "#4a90c4", "#1a5f9e"]

    fig, ax = plt.subplots(figsize=(8.5 if has_ablation else 6.5, 3.8))
    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5,
                  error_kw={"elinewidth": 1.2, "ecolor": "#333"})

    # Supervised reference line
    ax.axhline(supervised, color="#c0392b", linewidth=1.4, linestyle="--", zorder=3)
    ax.text(len(labels) - 0.45, supervised + 0.5, f"Supervised CNN\n{supervised:.1f}%",
            color="#c0392b", fontsize=8.5, va="bottom", ha="right")

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        label = f"{mean:.1f}%" if std == 0 else f"{mean:.1f}%\n±{std:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (std or 0) + 0.8,
                label, ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("MNIST test accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[figures] saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    templates_u8 = np.load(TEMPLATES_PATH)

    make_fig1(
        templates_u8,
        scale=8,
        out_path=os.path.join(OUT_DIR, "fig1_templates.png"),
    )
    make_fig2(
        templates_u8,
        template_idx=0,
        n_per_col=5,
        scale=6,
        out_path=os.path.join(OUT_DIR, "fig2_augmentation.png"),
    )
    make_fig3(
        results_path=os.path.join(ROOT, "experiments", "reports", "oneshot_results.json"),
        out_path=os.path.join(OUT_DIR, "fig3_results.png"),
    )


if __name__ == "__main__":
    main()
