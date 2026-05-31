import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.utils import ensure_dir
from src.variants17.label_schema import LABELS_17


def _binarize(img_gray: np.ndarray) -> np.ndarray:
    # foreground=True where dark stroke likely exists
    return img_gray < 245


def _find_true_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d)))
    return runs


def _center_to_canvas(crop_gray: np.ndarray, out_size: int = 28) -> np.ndarray:
    fg = _binarize(crop_gray)
    ys, xs = np.where(fg)
    if len(xs) == 0 or len(ys) == 0:
        return np.full((out_size, out_size), 255, dtype=np.uint8)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    tight = crop_gray[y0:y1, x0:x1]

    h, w = tight.shape
    max_side = max(h, w)
    scale = min(20.0 / max_side, 1.0)  # keep inside 20x20 center box
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    resized = np.array(Image.fromarray(tight).resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.uint8)
    canvas = np.full((out_size, out_size), 0, dtype=np.uint8)  # black background = MNIST convention

    oy = (out_size - nh) // 2
    ox = (out_size - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = 255 - resized  # invert: dark ink → bright stroke
    return canvas


def extract_17_digits(image_path: str) -> tuple[np.ndarray, list[dict]]:
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    # from inspection: center band contains the digit strip
    band = arr[230:451, :]
    fg = _binarize(band)

    # 1) x columns containing any dark pixel
    x_has_fg = fg.any(axis=0)
    x_regions = _find_true_runs(x_has_fg)

    if len(x_regions) != 17:
        raise ValueError(f"Expected 17 x-regions, found {len(x_regions)}")

    mnist_crops = []
    raw_crops = []
    meta = []
    for i, (x0, x1) in enumerate(x_regions):
        x_slice = band[:, x0:x1]
        x_fg = fg[:, x0:x1]

        # 2) y rows containing any dark pixel in this x region
        y_has_fg = x_fg.any(axis=1)
        y_regions = _find_true_runs(y_has_fg)
        if not y_regions:
            y0, y1 = 0, x_slice.shape[0]
        else:
            # if there are small disconnected specks, keep full span for the digit
            y0 = y_regions[0][0]
            y1 = y_regions[-1][1]

        raw = x_slice[y0:y1, :]
        centered = _center_to_canvas(raw, out_size=28)

        mnist_crops.append(centered)
        raw_crops.append(raw)
        meta.append(
            {
                "index": i,
                "label": LABELS_17[i],
                "x0": int(x0),
                "x1": int(x1),
                "y0": int(y0),
                "y1": int(y1),
                "w": int(x1 - x0),
                "h": int(y1 - y0),
            }
        )

    return np.stack(mnist_crops, axis=0), raw_crops, meta


def main(args):
    if len(LABELS_17) != 17:
        raise ValueError(f"Expected 17 labels, got {len(LABELS_17)}")

    ensure_dir(args.out_dir)
    templates, raw_templates, meta = extract_17_digits(args.image_path)
    labels = np.arange(17, dtype=np.int64)

    np.save(os.path.join(args.out_dir, "train_images.npy"), templates)
    np.save(os.path.join(args.out_dir, "train_labels17.npy"), labels)

    vis_dir = os.path.join(args.out_dir, "templates")
    ensure_dir(vis_dir)
    for i, t in enumerate(templates):
        Image.fromarray(t).save(os.path.join(vis_dir, f"{i:02d}_{LABELS_17[i]}.png"))

    raw_vis_dir = os.path.join(args.out_dir, "templates_raw")
    ensure_dir(raw_vis_dir)
    for i, t in enumerate(raw_templates):
        Image.fromarray(t).save(os.path.join(raw_vis_dir, f"{i:02d}_{LABELS_17[i]}.png"))

    with open(os.path.join(args.out_dir, "split_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved templates: {templates.shape} -> {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", default="data/processed/17digits_fixed_equal_height_thickness.png")
    parser.add_argument("--out-dir", default="data/processed/mnist17_variants")
    main(parser.parse_args())
