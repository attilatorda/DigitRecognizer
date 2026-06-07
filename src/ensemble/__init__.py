"""Track 7 — semi-supervised stacking ensemble over the one-shot recognisers.

NOT one-shot: this track deliberately steps outside the one-shot regime. It treats the
three one-shot recognisers (morphological proto, DDPM proto, structural RF) as fixed
feature extractors and trains a small meta-classifier on labeled MNIST data, studying
how few labels are needed.
"""

from .stacking import StackingEnsemble

__all__ = ["StackingEnsemble"]
