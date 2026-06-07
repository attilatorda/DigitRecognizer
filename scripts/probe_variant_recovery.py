"""
Track 8 gate experiment: is the 7/1 style variant recoverable from a 10-class CNN?

A standard 10-class CNN never sees variant labels and could learn to be invariant to
within-class style. We test whether the variant survives in its 128-dim embedding.

Variant pseudo-labels (cheap, from the Track 6 skeleton graph):
  - digit 7: "crossed" if the skeleton has >=1 junction (a crossbar crosses the diagonal),
             else "uncrossed".
  - digit 1: "ornamented" if >2 endpoints or >=1 junction (serif/foot/flag), else "plain".

For each digit we train a LINEAR PROBE (logreg) to predict the variant from:
  (a) the CNN embedding   -> does the net preserve the variant?
  (b) raw pixels          -> reference (variant is visible in pixels)
and compare to the majority base rate. KMeans(2) ARI tests unsupervised recoverability.

Usage:  python scripts/probe_variant_recovery.py
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

from src.common.data_io import load_mnist_idx
from src.ensemble.members import CNNMember, thin_batch
from src.structural.skeleton_graph import build_graph


def _graph_counts(thin_u8):
    n_end = n_junc = 0
    for node in build_graph(thin_u8)["nodes"]:
        if node["type"] == "endpoint":
            n_end += 1
        elif node["type"] == "junction":
            n_junc += 1
    return n_end, n_junc


def variant_labels(images_u8, digit):
    thin = thin_batch(images_u8)
    out = np.zeros(len(images_u8), dtype=int)
    for i, s in enumerate(thin):
        e, j = _graph_counts(s)
        if digit == 7:
            out[i] = 1 if j >= 1 else 0          # crossed vs uncrossed
        else:  # digit 1
            out[i] = 1 if (e > 2 or j >= 1) else 0  # ornamented vs plain
    return out


def _probe(X, v, tag):
    Xtr, Xte, vtr, vte = train_test_split(X, v, test_size=0.3, random_state=0, stratify=v)
    acc = (LogisticRegression(max_iter=2000).fit(Xtr, vtr).predict(Xte) == vte).mean()
    print(f"    {tag:14s} probe acc = {acc*100:.1f}%")
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}")
    raw_tr, y_tr = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_te, y_te = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")

    print("[probe] training 10-class CNN ...")
    cnn = CNNMember(device, epochs=6).fit(raw_tr, y_tr)
    acc10 = (cnn.predict_proba(raw_te).argmax(1) == y_te).mean()
    print(f"[probe] CNN 10-class test acc = {acc10*100:.2f}%")

    for digit in (7, 1):
        m = y_te == digit
        imgs = raw_te[m]
        v = variant_labels(imgs, digit)
        base = max(v.mean(), 1 - v.mean())
        name = {7: "crossed/uncrossed", 1: "ornamented/plain"}[digit]
        print(f"\n[digit {digit}]  variant={name}  n={len(v)}  "
              f"split={int(v.sum())}/{int((1-v).sum())}  base-rate={base*100:.1f}%")
        emb = cnn.embed(imgs)                              # (n,128)
        pix = imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0
        a_emb = _probe(emb, v, "embedding")
        a_pix = _probe(pix, v, "raw-pixel")
        ari = adjusted_rand_score(v, KMeans(2, n_init=10, random_state=0).fit_predict(emb))
        print(f"    KMeans(2) on embedding vs variant: ARI = {ari:.3f}")
        verdict = "RECOVERABLE" if a_emb > base + 0.1 else "weak"
        print(f"    -> embedding recovers variant: {verdict} "
              f"(+{(a_emb-base)*100:.1f}pp over base rate)")


if __name__ == "__main__":
    main()
