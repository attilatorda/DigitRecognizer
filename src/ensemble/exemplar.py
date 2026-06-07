"""Exemplar selector — find the training examples that "fit a class strongest".

Prototypicality combines two signals from a trained CNN member:
  - confidence: the model's softmax probability on the TRUE class (high = unambiguous)
  - centrality: closeness to the class centroid in 128-dim embedding space (close = typical)

score = z(confidence) - z(centroid_distance)   (higher = more prototypical)

The top-K per class are the cleanest, most representative examples — a coreset for
label-efficient training and the ideal seed set for the (deferred) diffusion generator.
"""

import numpy as np


def _zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v - v.mean()


def prototypicality_scores(cnn_member, images_u8, labels):
    """
    Parameters
    ----------
    cnn_member : a fitted CNNMember (provides predict_proba + embed)
    images_u8  : (N,28,28) uint8
    labels     : (N,) int

    Returns
    -------
    (N,) float scores; higher = more prototypical for its class.
    """
    labels = np.asarray(labels)
    probs = cnn_member.predict_proba(images_u8)          # (N,10)
    conf = probs[np.arange(len(labels)), labels]         # true-class confidence
    emb = cnn_member.embed(images_u8)                    # (N,128)

    # distance to own-class centroid
    dist = np.zeros(len(labels), dtype=np.float32)
    for c in range(10):
        m = labels == c
        if m.sum() == 0:
            continue
        centroid = emb[m].mean(axis=0)
        dist[m] = np.linalg.norm(emb[m] - centroid, axis=1)

    # z-score within class so classes are comparable
    score = np.zeros(len(labels), dtype=np.float32)
    for c in range(10):
        m = labels == c
        if m.sum() == 0:
            continue
        score[m] = _zscore(conf[m]) - _zscore(dist[m])
    return score


def select_top_k_per_class(scores, labels, k):
    """Return indices of the top-k highest-scoring (most prototypical) per class."""
    labels = np.asarray(labels)
    sel = []
    for c in range(10):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(-scores[idx])]
        sel.extend(order[:k].tolist())
    return np.array(sorted(sel))
