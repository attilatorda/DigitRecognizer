"""Corruption suite for the Track 9c robustness study.

apply_corruption(images_u8, kind, severity, rng) maps a batch of (N,28,28) uint8 MNIST
images to a corrupted (N,28,28) uint8 batch. Kinds:

  clean          - identity
  gaussian_noise - additive Gaussian pixel noise
  blur           - Gaussian blur (defocus)
  stroke_dilate  - morphological thickening of strokes (skeletonization is invariant to
                   this by construction -> the structural member's expected strong suit)
  occlusion      - a random square of the image is erased to background

One moderate `severity` per kind is the default; severity is exposed so a robustness-vs-
severity curve can be swept later. `denoise_for_skeleton` is a light median filter applied
before skeletonization so additive noise does not spawn spurious skeleton branches.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import dilation, disk

CORRUPTIONS = ["clean", "gaussian_noise", "blur", "stroke_dilate", "occlusion"]

# moderate default severities (chosen to clearly degrade a clean-trained CNN without
# destroying the digit)
_DEFAULT_SEVERITY = {
    "gaussian_noise": 0.5,   # std in [0,1] image units
    "blur": 1.1,             # Gaussian sigma (px)
    "stroke_dilate": 1,      # disk radius
    "occlusion": 10,         # erased square side (px)
}


def _to01(images_u8):
    return images_u8.astype(np.float32) / 255.0


def _to_u8(img01):
    return (np.clip(img01, 0.0, 1.0) * 255).astype(np.uint8)


def apply_corruption(images_u8: np.ndarray, kind: str, severity=None,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Corrupt a (N,28,28) uint8 batch. Returns (N,28,28) uint8."""
    if kind == "clean":
        return images_u8.copy()
    if kind not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {kind!r}; choose from {CORRUPTIONS}")
    rng = rng or np.random.default_rng(0)
    sev = _DEFAULT_SEVERITY[kind] if severity is None else severity
    x = _to01(images_u8)

    if kind == "gaussian_noise":
        x = x + rng.normal(0.0, sev, size=x.shape).astype(np.float32)
        return _to_u8(x)

    if kind == "blur":
        out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = gaussian_filter(x[i], sigma=sev)
        return _to_u8(out)

    if kind == "stroke_dilate":
        fp = disk(int(sev))
        out = np.empty_like(x)
        for i in range(len(x)):
            fg = dilation(x[i] > 0.3, fp)
            out[i] = np.where(fg, np.maximum(x[i], 0.85), x[i])
        return _to_u8(out)

    if kind == "occlusion":
        side = int(sev)
        out = x.copy()
        for i in range(len(out)):
            r = rng.integers(0, 28 - side + 1)
            c = rng.integers(0, 28 - side + 1)
            out[i, r:r + side, c:c + side] = 0.0
        return _to_u8(out)


def denoise_for_skeleton(images_u8: np.ndarray) -> np.ndarray:
    """Light 2x2-ish median filter to suppress speckle before skeletonization."""
    out = np.empty_like(images_u8)
    for i in range(len(images_u8)):
        out[i] = median_filter(images_u8[i], size=2)
    return out
