import numpy as np
from skimage.morphology import medial_axis, skeletonize, thin


def zhang_suen_skeletonize_uint8(img_2d: np.ndarray) -> np.ndarray:
    """Apply Zhang-Suen-style thinning via scikit-image skeletonize.
    Input: uint8 [0..255], foreground assumed as bright strokes.
    Output: uint8 [0..255].
    """
    binary = img_2d > 30
    skel = skeletonize(binary)
    return (skel.astype(np.uint8) * 255)


def skeletonize_uint8(img_2d: np.ndarray, method: str = "zhang") -> np.ndarray:
    """Apply configurable skeletonization/thinning on uint8 [0..255] grayscale image.

    Supported methods:
      - "zhang": scikit-image skeletonize (Zhang-Suen style for 2D)
      - "lee": scikit-image skeletonize(method="lee")
      - "thin": scikit-image thin
      - "medial_axis": scikit-image medial_axis
    """
    binary = img_2d > 30
    method = method.lower()

    if method == "zhang":
        skel = skeletonize(binary)
    elif method == "lee":
        skel = skeletonize(binary, method="lee")
    elif method == "thin":
        skel = thin(binary)
    elif method == "medial_axis":
        skel = medial_axis(binary)
    else:
        raise ValueError(
            f"Unsupported skeletonization method: {method}. "
            "Choose from: zhang, lee, thin, medial_axis"
        )

    return (skel.astype(np.uint8) * 255)
