import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir


def variant_split_label(digit, img):
    if digit == 0:
        return 1 if img[14, :].mean() > 50 else 0  # crude crossed detector split
    if digit == 1:
        return 2 if img[:8, :].mean() > 35 else 3
    if digit == 4:
        return 7 if img[10:18, 10:18].mean() > 55 else 6
    if digit == 7:
        return 10 if img[14, :].mean() > 50 else 11
    mapping = {2: 4, 3: 5, 5: 8, 6: 9, 8: 12, 9: 13}
    return mapping[int(digit)]


def main(args):
    ensure_dir(args.out_dir)
    for split in ["train", "t10k"]:
        images, labels = load_mnist_idx(args.mnist_path, split)
        labels14 = np.array([variant_split_label(int(d), img) for d, img in zip(labels, images)], dtype=np.int64)
        np.save(os.path.join(args.out_dir, f"{split}_images.npy"), images)
        np.save(os.path.join(args.out_dir, f"{split}_labels14.npy"), labels14)
        print(f"saved {split}: {images.shape}, labels={labels14.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path", default="mnist_data")
    parser.add_argument("--out-dir", default="data/processed/mnist14_variants")
    main(parser.parse_args())
