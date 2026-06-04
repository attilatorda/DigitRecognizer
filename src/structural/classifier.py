"""Nearest-neighbour classifier over structural feature vectors."""

import numpy as np
from .bag_of_features import vectorize, normalize


class StructuralClassifier:
    """1-NN classifier on L1-normalized structural feature vectors (cosine distance)."""

    def __init__(self):
        self._template_vecs: np.ndarray | None = None  # (n_templates, VOCAB_DIM)
        self._template_labels: np.ndarray | None = None

    def fit(self, template_features: list, labels: np.ndarray) -> None:
        """template_features: list of feature-lists (one per template)."""
        vecs = [normalize(vectorize(f)) for f in template_features]
        self._template_vecs = np.stack(vecs, axis=0)
        self._template_labels = np.asarray(labels)

    def _cosine_nearest(self, vec: np.ndarray) -> int:
        t = self._template_vecs
        num = t @ vec
        den = np.linalg.norm(t, axis=1) * (np.linalg.norm(vec) + 1e-9) + 1e-9
        sims = num / den
        return int(self._template_labels[int(sims.argmax())])

    def predict_one(self, features: list) -> int:
        return self._cosine_nearest(normalize(vectorize(features)))

    def predict(self, feature_lists: list) -> np.ndarray:
        return np.array([self.predict_one(f) for f in feature_lists], dtype=np.int64)
