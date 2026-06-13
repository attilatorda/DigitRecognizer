"""
Track 9c — leave-one-corruption-out (LOCO): the realistic "unseen corruption" test.

The robust experiment showed a corruption-AUGMENTED record holder (record_aug) beats the
agnostic ensemble at n>=100 -- but that baseline trained on the exact corruption it was
tested on. In deployment you do not know the corruption in advance. LOCO removes that unfair
advantage: for each held-out corruption h, train record_aug on the OTHER corruptions only and
test on h, then compare against the corruption-AGNOSTIC ensemble (trained on clean data,
numbers loaded from track9_robust_results.json -- no recompute).

Question: does corruption-agnostic structural/generative diversity generalize to UNSEEN
corruptions better than targeted corruption augmentation?

Lines per (held-out corruption, budget):
  record_vote      - clean-trained baseline           (from robust JSON)
  record_aug_full  - trained ON h (upper bound, unfair) (from robust JSON, n<=500)
  conf_all         - agnostic ensemble                 (from robust JSON)
  record_aug_loco  - trained on the OTHER corruptions, tested on unseen h   (computed here)

Usage:
    python scripts/run_track9_loco.py --smoke
    python scripts/run_track9_loco.py
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

from src.common.data_io import load_mnist_idx
from src.common.utils import set_seed
from src.track9.record_models import RecordModel, train_record_model, _predict_probs
from src.track9.corruptions import apply_corruption, CORRUPTIONS

BUDGETS = [100, 200, 500, 1000, 2000]
HELD_OUT = [c for c in CORRUPTIONS if c != "clean"]   # noise, blur, stroke_dilate, occlusion
RECORD_KERNELS = (3, 5, 7)
ROBUST_JSON = os.path.join(ROOT, "experiments", "reports", "track9_robust_results.json")


def _select(labels, n, rng):
    sel = []
    for c in range(10):
        sel.extend(rng.choice(np.where(labels == c)[0], size=n // 10, replace=False).tolist())
    return np.array(sel)


def _corrupt_train(raw_u8, rng, allowed):
    """Corruption-aware training set using only the `allowed` corruption kinds."""
    out = raw_u8.copy()
    for i in range(len(out)):
        if rng.random() < 0.5:
            k = allowed[rng.integers(len(allowed))]
            out[i] = apply_corruption(out[i:i + 1], k, rng=rng)[0]
    return out


def _majority(probs_list):
    preds = np.stack([p.argmax(1) for p in probs_list], axis=0)
    summed = np.sum(probs_list, axis=0)
    out = np.empty(preds.shape[1], dtype=np.int64)
    for i in range(preds.shape[1]):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        if counts.max() >= 2:
            cands = vals[counts == counts.max()]
            out[i] = cands[summed[i, cands].argmax()] if len(cands) > 1 else cands[0]
        else:
            out[i] = summed[i].argmax()
    return out


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[loco] device={device} smoke={args.smoke}", flush=True)
    raw_pool, y_pool = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_test, y_test = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    if args.smoke:
        raw_test, y_test = raw_test[:1500], y_test[:1500]

    # agnostic comparison numbers (computed once, in the robust run)
    agn = json.load(open(ROBUST_JSON))["accuracy"] if os.path.exists(ROBUST_JSON) else {}

    budgets = [200] if args.smoke else BUDGETS
    held_out = ["gaussian_noise"] if args.smoke else HELD_OUT
    seeds = 1 if args.smoke else args.seeds

    # cache held-out corrupted test sets (CNNs predict on images directly -> no skeletons)
    crng = np.random.default_rng(12345)
    test_imgs = {h: apply_corruption(raw_test, h, rng=crng) for h in held_out}

    results = {}
    for h in held_out:
        allowed = [c for c in HELD_OUT if c != h]   # train on the OTHER corruptions
        results[h] = {}
        for n in budgets:
            accs = []
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                sel = _select(y_pool, n, rng)
                raw_c, y_c = raw_pool[sel], y_pool[sel]
                arng = np.random.default_rng(900 + s)
                raw_aug = _corrupt_train(raw_c, arng, allowed)
                set_seed(s)
                probs = []
                for k in RECORD_KERNELS:
                    m = RecordModel(k)
                    train_record_model(m, raw_aug, y_c, device, epochs=args.record_epochs, seed=s + k)
                    probs.append(_predict_probs(m, test_imgs[h], device))
                accs.append(float((_majority(probs) == y_test).mean()))
            results[h][n] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "n": len(accs)}

            def fromj(line):
                try:
                    v = agn[line][h][str(n)]
                    return f"{v['mean']*100:.1f}" if v else "--"
                except KeyError:
                    return "--"
            print(f"  [{h:14s} n={n:4d}] record_aug_LOCO={np.mean(accs)*100:5.1f}  "
                  f"| agnostic conf_all={fromj('conf_all')}  record_vote={fromj('record_vote')}  "
                  f"record_aug_full={fromj('record_aug')}", flush=True)

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "track9_loco_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"budgets": budgets, "held_out": held_out, "seeds": seeds,
                       "record_aug_loco": {h: {str(n): results[h][n] for n in budgets} for h in held_out}},
                      f, indent=2)
        print(f"[loco] saved {out}", flush=True)
    else:
        print("[loco] smoke OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--record-epochs", type=int, default=40)
    a = p.parse_args()
    if a.smoke:
        a.seeds = 1
        a.record_epochs = 3
    main(a)
