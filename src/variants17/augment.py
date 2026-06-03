import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.morphology import disk, dilation, erosion
from skimage.transform import AffineTransform, rotate, warp


def elastic_distort(
    img01: np.ndarray,
    alpha: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simard 2003-style elastic grid warping on a (28,28) float32 image."""
    h, w = img01.shape
    dy = gaussian_filter(rng.uniform(-1.0, 1.0, size=(h, w)), sigma=sigma) * alpha
    dx = gaussian_filter(rng.uniform(-1.0, 1.0, size=(h, w)), sigma=sigma) * alpha
    row_grid, col_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [
        np.clip(row_grid + dy, 0, h - 1),
        np.clip(col_grid + dx, 0, w - 1),
    ]
    warped = map_coordinates(img01, coords, order=1, mode="nearest")
    return np.clip(warped, 0.0, 1.0).astype(np.float32)


def stroke_width_adjust(
    img01: np.ndarray,
    mode: str,
    radius: int = 1,
    blur_sigma: float = 0.9,
) -> np.ndarray:
    """Adjust stroke width. Mode: 'disk_dilate' | 'disk_erode' | 'blur_soft' | 'none'."""
    if mode == "blur_soft":
        from scipy.ndimage import gaussian_filter as gf
        blurred = gf(img01, sigma=blur_sigma)
        return np.clip(blurred, 0.0, 1.0).astype(np.float32)

    if mode in ("disk_dilate", "disk_erode"):
        fp = disk(radius)
        fg = img01 > 0.3
        if mode == "disk_dilate":
            fg_new = dilation(fg, fp)
        else:
            fg_new = erosion(fg, fp)
        # Guard: if erosion wiped out almost all strokes, skip
        if fg_new.sum() < 5:
            return img01
        out = img01.copy()
        out = np.where(fg_new, np.maximum(out, 0.65), out)
        out = np.where(~fg_new, np.minimum(out, 0.15), out)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    return img01  # 'none'


def sample_augmentation_params(
    rng: np.random.Generator,
    elastic_prob: float = 0.70,
    stroke_prob: float = 0.80,
) -> dict:
    do_elastic = rng.random() < elastic_prob
    do_stroke = rng.random() < stroke_prob
    stroke_mode = (
        str(rng.choice(["disk_dilate", "disk_erode", "blur_soft"], p=[0.30, 0.15, 0.55]))
        if do_stroke
        else "none"
    )
    return {
        "rotation_deg": float(rng.uniform(-15.0, 15.0)),
        "stretch_x": float(rng.uniform(0.90, 1.10)),
        "stretch_y": float(rng.uniform(0.90, 1.10)),
        "translate_x": float(rng.uniform(-2.0, 2.0)),
        "translate_y": float(rng.uniform(-2.0, 2.0)),
        "do_elastic": bool(do_elastic),
        "elastic_alpha": float(rng.uniform(10.0, 34.0)),
        "elastic_sigma": float(rng.uniform(3.0, 5.0)),
        "stroke_mode": stroke_mode,
        "stroke_radius": 1,
        "stroke_blur_sigma": float(rng.uniform(0.6, 2.0)),
        "noise_std": float(rng.uniform(0.01, 0.04)),
    }


def apply_augmentation(
    img01: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply full augmentation pipeline to a (28,28) float32 image in [0,1]."""
    out = img01.copy()

    # 1. Rotation
    out = rotate(
        out,
        angle=params["rotation_deg"],
        resize=False,
        mode="edge",
        preserve_range=True,
    ).astype(np.float32)

    # 2. Affine stretch + translation (single pass)
    tf = AffineTransform(
        scale=(params["stretch_x"], params["stretch_y"]),
        translation=(params["translate_x"], params["translate_y"]),
    )
    out = warp(out, tf.inverse, preserve_range=True, mode="edge", output_shape=out.shape).astype(np.float32)

    # 3. Elastic distortion
    if params["do_elastic"]:
        out = elastic_distort(out, params["elastic_alpha"], params["elastic_sigma"], rng)

    # 4. Stroke-width simulation
    out = stroke_width_adjust(
        out,
        mode=params["stroke_mode"],
        radius=params["stroke_radius"],
        blur_sigma=params["stroke_blur_sigma"],
    )

    # 5. Gaussian noise
    noise = rng.normal(0.0, params["noise_std"], size=out.shape).astype(np.float32)
    out = np.clip(out + noise, 0.0, 1.0)

    return out.astype(np.float32)


def augment_dataset(
    templates_u8: np.ndarray,
    labels: np.ndarray,
    repeats: int = 256,
    seed: int = 42,
    elastic_prob: float = 0.70,
    stroke_prob: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment templates to a training set. Replaces the old noise-only augment_templates()."""
    rng = np.random.default_rng(seed)
    base = templates_u8.astype(np.float32) / 255.0
    X, Y = [], []
    for img, y in zip(base, labels):
        for _ in range(repeats):
            params = sample_augmentation_params(rng, elastic_prob=elastic_prob, stroke_prob=stroke_prob)
            sample = apply_augmentation(img, params, rng)
            X.append(sample)
            Y.append(y)
    return np.stack(X, axis=0), np.array(Y, dtype=np.int64)
