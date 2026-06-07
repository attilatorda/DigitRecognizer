"""Stacking meta-classifier over base recognisers' probability outputs.

Each base recogniser supplies a per-image probability vector; the meta-classifier is
trained on the concatenation. Because the base recognisers are fixed (trained on
non-MNIST data), the meta-classifier is the only component that consumes MNIST labels.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def _stack(member_probs):
    """Concatenate a list of (N, C) probability matrices into (N, sum C)."""
    return np.concatenate(member_probs, axis=1)


class StackingEnsemble:
    """
    Parameters
    ----------
    meta : 'logreg' | 'histgb'
        Meta-classifier. logreg is stable and label-efficient at small budgets;
        histgb is stronger at large budgets.
    """

    def __init__(self, meta: str = "logreg", random_state: int = 0):
        if meta == "logreg":
            self.clf = LogisticRegression(max_iter=2000, C=1.0)
        elif meta == "histgb":
            self.clf = HistGradientBoostingClassifier(max_iter=300, random_state=random_state)
        else:
            raise ValueError(f"unknown meta-classifier: {meta!r}")
        self.meta = meta

    def fit(self, member_probs: list, y: np.ndarray) -> "StackingEnsemble":
        self.clf.fit(_stack(member_probs), y)
        return self

    def predict(self, member_probs: list) -> np.ndarray:
        return self.clf.predict(_stack(member_probs))

    def score(self, member_probs: list, y: np.ndarray) -> float:
        return float((self.predict(member_probs) == y).mean())
