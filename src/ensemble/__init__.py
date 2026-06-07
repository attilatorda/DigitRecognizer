"""Track 7 — semi-supervised stacking ensemble over the one-shot recognisers.

NOT one-shot: this track deliberately steps outside the one-shot regime. It treats the
three one-shot recognisers (morphological proto, DDPM proto, structural RF) as fixed
feature extractors and trains a small meta-classifier on labeled MNIST data, studying
how few labels are needed.
"""

from .stacking import StackingEnsemble
from .members import CNNMember, FusionCNNMember, StructuralRFMember, train_cnn
from .exemplar import prototypicality_scores, select_top_k_per_class

__all__ = [
    "StackingEnsemble",
    "CNNMember", "FusionCNNMember", "StructuralRFMember", "train_cnn",
    "prototypicality_scores", "select_top_k_per_class",
]
