"""
Track 6 v2 — structural recognition with rich features + a trained classifier.

Two improvements over v1 (1-NN vs 17 templates, 35.81%):
  1. RICH FEATURES: 88-dim vector (60-dim bag + orientation histogram, curvature/
     inflection stats, geometry, endpoint/loop position).
  2. REFERENCE BANK: extract features from thousands of augmented/DDPM training
     images instead of 17 clean templates, then train LogisticRegression / kNN.

Usage:
    python scripts/run_structural_v2_experiment.py
    python scripts/run_structural_v2_experiment.py --bank both --smoke
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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir
from src.skeleton.skeletonize import zhang_suen_skeletonize_uint8
from src.structural.rich_features import extract_rich_vector, RICH_DIM
from src.variants17.label_schema import CLASS17_TO_DIGIT10


def _to_u8(img):
    if img.dtype != np.uint8:
        return (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def _featurize(images, tag=""):
    t0 = time.perf_counter()
    X = np.zeros((len(images), RICH_DIM), dtype=np.float32)
    for i, img in enumerate(images):
        X[i] = extract_rich_vector(zhang_suen_skeletonize_uint8(_to_u8(img)))
        if tag and (i + 1) % 2000 == 0:
            print(f"  [{tag}] featurized {i+1}/{len(images)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"  [{tag}] done {len(images)} in {time.perf_counter()-t0:.0f}s", flush=True)
    return X


def _load_bank(which):
    """Return (images, digit10_labels) for the requested reference bank."""
    parts = []
    if which in ("morphological", "both"):
        p = os.path.join(ROOT, "experiments/checkpoints/variants17/augmented_train")
        imgs = np.load(os.path.join(p, "images.npy"))
        l17 = np.load(os.path.join(p, "labels17.npy"))
        parts.append((imgs, np.array([CLASS17_TO_DIGIT10[int(l)] for l in l17])))
        print(f"[bank] morphological: {len(imgs)} images")
    if which in ("ddpm", "both"):
        p = os.path.join(ROOT, "experiments/checkpoints/diffusion_aug")
        imgs = np.load(os.path.join(p, "generated_images.npy"))
        l17 = np.load(os.path.join(p, "generated_labels.npy"))
        parts.append((imgs, np.array([CLASS17_TO_DIGIT10[int(l)] for l in l17])))
        print(f"[bank] ddpm: {len(imgs)} images")
    images = np.concatenate([p[0] for p in parts], axis=0)
    labels = np.concatenate([p[1] for p in parts], axis=0)
    return images, labels


def main(args):
    t_start = time.perf_counter()

    # --- reference bank ---
    bank_images, bank_labels = _load_bank(args.bank)
    print(f"[bank] total {len(bank_images)} reference images, {RICH_DIM}-dim features")
    X_train = _featurize(bank_images, tag="bank")
    y_train = bank_labels

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    # --- classifiers ---
    logreg = LogisticRegression(max_iter=2000, C=1.0)
    logreg.fit(X_train_s, y_train)
    knn = KNeighborsClassifier(n_neighbors=args.k, weights="distance")
    knn.fit(X_train_s, y_train)

    # --- test set ---
    test_images, test_labels = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    if args.smoke:
        test_images, test_labels = test_images[:args.smoke_n], test_labels[:args.smoke_n]
    X_test = _featurize(test_images, tag="test")
    X_test_s = scaler.transform(X_test)

    acc_logreg = float((logreg.predict(X_test_s) == test_labels).mean())
    acc_knn = float((knn.predict(X_test_s) == test_labels).mean())

    elapsed = time.perf_counter() - t_start
    print("\n=== TRACK 6 v2 RESULTS ===")
    print(f"reference bank      : {args.bank}  ({len(bank_images)} images)")
    print(f"feature dim         : {RICH_DIM}")
    print(f"LogisticRegression  : {acc_logreg*100:.2f}%")
    print(f"kNN (k={args.k}, dist)    : {acc_knn*100:.2f}%")
    print(f"total time          : {elapsed:.0f}s")
    print("\n--- vs prior ---")
    print(f"v1 structural (1-NN, 17 templates) : 35.81%")
    print(f"Variants17 proto baseline          : 77.46%")
    best = max(acc_logreg, acc_knn)
    print(f"v2 best                            : {best*100:.2f}%  "
          f"(+{best*100-35.81:.2f}pp vs v1, {best*100-77.46:+.2f}pp vs proto)")

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "structural_v2_results.json")
        ensure_dir(os.path.dirname(out))
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "bank": args.bank, "n_reference": len(bank_images),
                "feature_dim": RICH_DIM,
                "logreg_acc": acc_logreg, "knn_acc": acc_knn,
                "knn_k": args.k, "elapsed_s": elapsed,
                "v1_acc": 0.3581, "proto_baseline": 0.7746,
            }, f, indent=2)
        print(f"\n[v2] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="morphological",
                   choices=["morphological", "ddpm", "both"],
                   help="Reference bank source")
    p.add_argument("--k", type=int, default=5, help="kNN neighbours")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=500)
    main(p.parse_args())
