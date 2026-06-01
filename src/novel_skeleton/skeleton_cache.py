import os
import time

import numpy as np

from src.novel_skeleton.skeletonize import skeletonize_batch


def load_or_build(
    images: np.ndarray,
    method: str,
    split: str,
    cache_dir: str,
    progress_every: int = 500,
) -> tuple[np.ndarray, float]:
    """Return (skeletonized_images, elapsed_seconds).

    Loads from cache if available (elapsed=0), otherwise skeletonizes and saves.
    split is a label like 'train' or 'test' used only for the filename and logs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{method}_{split}.npy")

    if os.path.exists(path):
        print(f"[skeleton_cache] loading {path}", flush=True)
        return np.load(path), 0.0

    print(f"[skeleton_cache] building {path} ({len(images)} images)", flush=True)
    t0 = time.perf_counter()
    result = skeletonize_batch(images, method=method, progress_every=progress_every, split_name=split)
    elapsed = time.perf_counter() - t0
    np.save(path, result)
    print(f"[skeleton_cache] saved {path} in {elapsed:.1f}s", flush=True)
    return result, elapsed
