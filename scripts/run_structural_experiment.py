"""
Track 6 — Structural bag-of-features one-shot digit recognition.

Pipeline:
  1. Skeletonize CultiVar-17 templates; extract structural features (17 vectors)
  2. Skeletonize MNIST test images; extract features
  3. 1-NN cosine classification against templates
  4. Map 17-class -> 10-digit; report MNIST accuracy vs Variants17 baseline (77.46%)

Usage:
    python scripts/run_structural_experiment.py --smoke      # 100 test images
    python scripts/run_structural_experiment.py              # full 10K  (slow)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir
from src.skeleton.skeletonize import zhang_suen_skeletonize_uint8
from src.structural.feature_extractor import extract_features
from src.structural.classifier import StructuralClassifier
from src.variants17.label_schema import CLASS17_TO_DIGIT10


def _skeletonize_u8(img_u8: np.ndarray) -> np.ndarray:
    """Zhang-Suen skeletonize a single uint8 image -> uint8 {0,255}."""
    return zhang_suen_skeletonize_uint8(img_u8)


def main(args):
    t0 = time.perf_counter()

    # --- templates ---
    templates_u8 = np.load(os.path.join(ROOT, "data", "processed", "mnist17_variants", "train_images.npy"))
    labels17 = np.load(os.path.join(ROOT, "data", "processed", "mnist17_variants", "train_labels17.npy"))
    template_features = [extract_features(_skeletonize_u8(t)) for t in templates_u8]

    clf = StructuralClassifier()
    clf.fit(template_features, labels17)
    print(f"[structural] fitted {len(templates_u8)} templates")

    # --- test set ---
    test_images, test_labels = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    if args.smoke:
        test_images, test_labels = test_images[:args.smoke_n], test_labels[:args.smoke_n]
    print(f"[structural] classifying {len(test_images)} test images ...")

    correct = 0
    for i, img in enumerate(test_images):
        feats = extract_features(_skeletonize_u8(img))
        pred17 = clf.predict_one(feats)
        pred10 = CLASS17_TO_DIGIT10[pred17]
        if pred10 == test_labels[i]:
            correct += 1
        if args.smoke and (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(test_images)}  running acc={correct/(i+1)*100:.1f}%")

    acc = correct / len(test_images)
    elapsed = time.perf_counter() - t0
    print(f"\n[structural] MNIST accuracy: {acc*100:.2f}%  ({len(test_images)} images, {elapsed:.1f}s)")

    # --- baseline comparison ---
    baseline_path = os.path.join(ROOT, "experiments", "reports", "oneshot_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            base = json.load(f)
        proto = next((c for c in base.get("configs", []) if c["name"] == "proto"), None)
        if proto:
            b = proto["mean_test_acc"] * 100
            print(f"[structural] vs Variants17 proto baseline: {b:.2f}%  (delta {acc*100-b:+.2f}pp)")

    if not args.smoke:
        ensure_dir(os.path.join(ROOT, "experiments", "reports"))
        out = os.path.join(ROOT, "experiments", "reports", "structural_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"mnist_accuracy": acc, "n_test": len(test_images),
                       "elapsed_s": elapsed}, f, indent=2)
        print(f"[structural] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Run on a small subset only")
    p.add_argument("--smoke-n", type=int, default=100)
    main(p.parse_args())
