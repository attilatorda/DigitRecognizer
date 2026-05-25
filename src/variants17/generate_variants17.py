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
    canvas = np.full((out_size, out_size), 255, dtype=np.uint8)

    oy = (out_size - nh) // 2
    ox = (out_size - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def extract_17_digits(image_path: str) -> tuple[np.ndarray, list[dict]]:
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    # from inspection: center band contains the digit strip
    band = arr[230:451, :]
    h, w = band.shape
    slot_w = w // 17

    crops = []
    meta = []
    for i in range(17):
        x0 = i * slot_w
        x1 = (i + 1) * slot_w if i < 16 else w
        raw = band[:, x0:x1]
        centered = _center_to_canvas(raw, out_size=28)
        crops.append(centered)
        meta.append({"index": i, "label": LABELS_17[i], "x0": int(x0), "x1": int(x1), "h": int(h)})

    return np.stack(crops, axis=0), meta


def main(args):
    if len(LABELS_17) != 17:
        raise ValueError(f"Expected 17 labels, got {len(LABELS_17)}")

    ensure_dir(args.out_dir)
    templates, meta = extract_17_digits(args.image_path)
    labels = np.arange(17, dtype=np.int64)

    np.save(os.path.join(args.out_dir, "train_images.npy"), templates)
    np.save(os.path.join(args.out_dir, "train_labels17.npy"), labels)

    vis_dir = os.path.join(args.out_dir, "templates")
    ensure_dir(vis_dir)
    for i, t in enumerate(templates):
        Image.fromarray(t).save(os.path.join(vis_dir, f"{i:02d}_{LABELS_17[i]}.png"))

    with open(os.path.join(args.out_dir, "split_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved templates: {templates.shape} -> {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", default="data/processed/17digits_fixed_equal_height_thickness.png")
    parser.add_argument("--out-dir", default="data/processed/mnist17_variants")
    main(parser.parse_args())
