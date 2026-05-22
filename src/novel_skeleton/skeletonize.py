import numpy as np
from skimage.morphology import skeletonize


def zhang_suen_skeletonize_uint8(img_2d: np.ndarray) -> np.ndarray:
    """Apply Zhang-Suen-style thinning via scikit-image skeletonize.
    Input: uint8 [0..255], foreground assumed as bright strokes.
    Output: uint8 [0..255].
    """
    binary = img_2d > 30
    skel = skeletonize(binary)
    return (skel.astype(np.uint8) * 255)
