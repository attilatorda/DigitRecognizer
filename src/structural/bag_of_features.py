"""Build fixed-length feature-count vectors from structural feature lists.

VOCABULARY is the sorted list of all (level, feature_type, size_class) entries.
vectorize() maps a feature list to a numpy count vector over VOCABULARY.
"""

import numpy as np

# Level 1 feature types
_L1_TYPES = ['endpoint', 'junction', 'loop_node']

# Level 2 feature types
_L2_TYPES = ['straight', 'curved', 'bent', 'right_angle', 'parallel_pair']

# Level 3 feature types
_L3_TYPES = ['crossbar', 'open_polygon', 'closed_loop', 'arc_chord']

_SIZE_CLASSES = ['XS', 'S', 'M', 'L', 'XL']

VOCABULARY: list[tuple] = [
    (level, ftype, size)
    for level, ftypes in [(1, _L1_TYPES), (2, _L2_TYPES), (3, _L3_TYPES)]
    for ftype in ftypes
    for size in _SIZE_CLASSES
]

_VOCAB_INDEX: dict[tuple, int] = {v: i for i, v in enumerate(VOCABULARY)}

VOCAB_DIM = len(VOCABULARY)  # 60


def vectorize(features: list) -> np.ndarray:
    """
    Parameters
    ----------
    features : list of (level, feature_type, size_class) tuples

    Returns
    -------
    float32 count vector of length VOCAB_DIM (60)
    """
    vec = np.zeros(VOCAB_DIM, dtype=np.float32)
    for f in features:
        idx = _VOCAB_INDEX.get(f)
        if idx is not None:
            vec[idx] += 1.0
    return vec


def normalize(vec: np.ndarray) -> np.ndarray:
    """L1-normalize a feature vector (standard bag-of-features normalization)."""
    s = vec.sum()
    return vec / s if s > 0 else vec
