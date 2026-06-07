"""
Track 8 idea #2b: does auxiliary sub-class supervision improve low-label training?

Mechanism (the user's idea, = CultiVar 17->10 applied to real MNIST): when a digit has
internal variants, split it into sub-classes, expand the CNN head (extra neurons), train
on the finer labels, then MAP BACK to 10 digits at evaluation.

Sub-classes are defined in the Track 6 STRUCTURAL feature space (not CNN-embedding KMeans,
which the gate showed fails to surface the variant): for each digit, KMeans(k=2) on the
93-dim rich features -> 2 structure-defined sub-classes (e.g. crossed/uncrossed 7).

Compared at each low-label budget (3 seeds):
  - baseline : plain 10-way CNN
  - subclass : 2k-way CNN trained on structure-defined sub-classes, probs summed back to 10

Also reports the digit-7 sanity check: ARI of structural-space KMeans vs the crossbar
pseudo-label (should beat the embedding-space ARI~0 from the gate).

Usage:  python scripts/run_subclass_expansion.py [--smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir
from src.local_cnn.model import SimpleCNN
from src.ensemble.members import train_cnn, thin_batch, rich_features
from src.structural.skeleton_graph import build_graph

BUDGETS = [100, 250, 500, 1000, 2500, 5000]


def _probs(model, raw_u8, device, bs=512):
    model.to(device).eval()
    X = torch.tensor(raw_u8.astype(np.float32) / 255.0).unsqueeze(1)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(X[i:i+bs].to(device)), 1).cpu().numpy())
    return np.concatenate(out, 0)


def assign_subclasses(feats, labels, k=2):
    """Per-digit KMeans(k) in structural space -> contiguous sub-class ids + id->digit map."""
    sub = np.zeros(len(labels), dtype=int)
    sub2digit = []
    nxt = 0
    for d in range(10):
        idx = np.where(labels == d)[0]
        kk = min(k, len(idx))
        if kk <= 1:
            sub[idx] = nxt; sub2digit.append(d); nxt += 1; continue
        cl = KMeans(kk, n_init=5, random_state=0).fit_predict(feats[idx])
        for c in range(kk):
            sub[idx[cl == c]] = nxt; sub2digit.append(d); nxt += 1
    return sub, np.array(sub2digit)


def _train_eval(raw_tr, lab_tr, n_classes, raw_te, y_te, device, epochs, sub2digit=None):
    model = SimpleCNN(num_classes=n_classes, in_channels=1)
    X = torch.tensor(raw_tr.astype(np.float32) / 255.0).unsqueeze(1)
    train_cnn(model, X, lab_tr, device, epochs)
    p = _probs(model, raw_te, device)
    if sub2digit is None:
        return float((p.argmax(1) == y_te).mean())
    # sum sub-class probs back to 10 digits
    p10 = np.zeros((len(y_te), 10), dtype=np.float32)
    for j, d in enumerate(sub2digit):
        p10[:, d] += p[:, j]
    return float((p10.argmax(1) == y_te).mean())


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[subclass] device={device}")
    rng0 = np.random.default_rng(0)
    raw_pool, y_pool = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_te, y_te = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    budgets = [250, 1000] if args.smoke else BUDGETS
    seeds = 1 if args.smoke else 3
    pool_n = 12000 if args.smoke else 30000

    keep = rng0.choice(len(y_pool), size=min(pool_n, len(y_pool)), replace=False)
    raw_pool, y_pool = raw_pool[keep], y_pool[keep]
    t0 = time.perf_counter()
    pool_feats = rich_features(thin_batch(raw_pool))
    print(f"[subclass] structural features for pool={len(y_pool)} in {time.perf_counter()-t0:.0f}s", flush=True)

    # sanity: structural-space KMeans recovers the crossed-7 variant?
    m7 = y_pool == 7
    thin7 = thin_batch(raw_pool[m7])
    cross = np.array([1 if sum(nd["type"] == "junction" for nd in build_graph(s)["nodes"]) >= 1 else 0
                      for s in thin7])
    ari7 = adjusted_rand_score(cross, KMeans(2, n_init=5, random_state=0).fit_predict(pool_feats[m7]))
    print(f"[subclass] digit-7 structural-KMeans vs crossbar ARI = {ari7:.3f}  "
          f"(embedding-space was ~0.02)")

    results = {"baseline": {}, "subclass": {}}
    for n in budgets:
        b_accs, s_accs = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(10 + seed)
            sel = np.concatenate([rng.choice(np.where(y_pool == d)[0],
                                             size=max(1, n // 10), replace=False) for d in range(10)])
            b_accs.append(_train_eval(raw_pool[sel], y_pool[sel], 10, raw_te, y_te, device, args.epochs))
            sub, s2d = assign_subclasses(pool_feats[sel], y_pool[sel], k=2)
            s_accs.append(_train_eval(raw_pool[sel], sub, len(s2d), raw_te, y_te, device, args.epochs, s2d))
        results["baseline"][n] = (float(np.mean(b_accs)), float(np.std(b_accs)))
        results["subclass"][n] = (float(np.mean(s_accs)), float(np.std(s_accs)))
        d = (np.mean(s_accs) - np.mean(b_accs)) * 100
        print(f"  n={n:5d}  baseline={np.mean(b_accs)*100:5.2f}%  subclass={np.mean(s_accs)*100:5.2f}%  "
              f"delta={d:+.2f}pp", flush=True)

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "subclass_expansion_results.json")
        ensure_dir(os.path.dirname(out))
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"budgets": budgets, "seeds": seeds, "digit7_ari": float(ari7),
                       "results": results}, f, indent=2)
        print(f"[subclass] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    main(p.parse_args())
