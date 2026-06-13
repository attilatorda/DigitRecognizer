"""Held-out-free combiners for the Track 9c ensemble.

All operate on a list of member probability arrays (each (N,10)) and return combined
probabilities (N,10). Because none of them need a held-out split, every member can train
on the FULL labeled budget -- the fix for the Track 9b fairness artifact, where the baseline
was starved of 30% of its labels to feed a logreg stacker.

  soft_vote      - equal-weight mean of member probabilities (parameter-free).
  conf_vote      - per-sample confidence-weighted mean; each member's weight is its own
                   prediction confidence on that image (max-prob or 1 - normalized entropy).
                   Adapts at test time, so it down-weights a member on inputs it cannot
                   handle (e.g. the structural member under noise) -- the featured combiner.
  weighted_vote  - fixed per-member weights (e.g. from cross-val accuracy).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def soft_vote(probs_list):
    return np.mean(probs_list, axis=0)


def _confidence(p, mode):
    if mode == "maxprob":
        return p.max(axis=1)
    # negative normalized entropy -> in [0,1], 1 == fully confident
    ent = -(p * np.log(p + _EPS)).sum(axis=1)
    return 1.0 - ent / np.log(p.shape[1])


def conf_vote(probs_list, mode="maxprob"):
    """Per-sample confidence-weighted average. weight_m(x) = conf_m(x) / sum_m conf_m(x)."""
    confs = np.stack([_confidence(p, mode) for p in probs_list], axis=0)  # (M, N)
    confs = confs / (confs.sum(axis=0, keepdims=True) + _EPS)             # normalize over M
    out = np.zeros_like(probs_list[0])
    for m, p in enumerate(probs_list):
        out += confs[m][:, None] * p
    return out


def weighted_vote(probs_list, weights):
    w = np.asarray(weights, dtype=np.float32)
    w = w / (w.sum() + _EPS)
    return np.tensordot(w, np.stack(probs_list, axis=0), axes=(0, 0))


def preds(combined_probs):
    return combined_probs.argmax(axis=1)


def accuracy(combined_probs, labels):
    return float((combined_probs.argmax(axis=1) == labels).mean())
