"""
Track 7 — semi-supervised stacking ensemble (NOT one-shot).

Treats the three one-shot recognisers (morphological proto, DDPM proto, structural RF)
as fixed feature extractors and asks: how label-efficient is a meta-classifier stacked
on their probability outputs, vs. learning from raw pixels?

Setup (leakage-free): the base recognisers never saw MNIST. We split the 10k MNIST test
into a fixed 5k EVAL set and a 5k LABEL POOL. For each label budget we sample n labeled
images from the pool, train (a) a stacker on the members' 30-dim probability features and
(b) a logistic-regression baseline on raw 784-dim pixels, and evaluate both on the EVAL
set. Multiple seeds average over the label sample.

Reference lines (label-free, one-shot): best single member, and the soft-average ensemble.

Requires experiments/reports/member_probs.npz (run build_member_predictions.py first).

Usage:  python scripts/run_ensemble_track.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.common.data_io import load_mnist_idx
from src.ensemble.stacking import StackingEnsemble

BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000]
SEEDS = 5


def _sample(pool, y, n, rng):
    """Stratified-ish sample of n indices from pool."""
    sel = []
    per = max(1, n // 10)
    for d in range(10):
        cand = pool[y[pool] == d]
        take = min(per, len(cand))
        sel.extend(rng.choice(cand, size=take, replace=False))
    sel = np.array(sel)
    if len(sel) > n:
        sel = rng.choice(sel, size=n, replace=False)
    return sel


def main():
    npz = np.load(os.path.join(ROOT, "experiments", "reports", "member_probs.npz"))
    morph, ddpm, struct, y = npz["morph"], npz["ddpm"], npz["struct"], npz["y"]
    members_all = [morph, ddpm, struct]

    # raw pixels for the baseline
    test_images, y2 = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    assert np.array_equal(y, y2), "member_probs labels must match MNIST t10k order"
    raw = test_images.reshape(len(y), -1).astype(np.float32) / 255.0

    idx = np.arange(len(y))
    pool, ev = train_test_split(idx, test_size=0.5, random_state=0, stratify=y)

    # label-free references (one-shot) on the eval set
    accs_single = [(m[ev].argmax(1) == y[ev]).mean() for m in members_all]
    best_single = max(accs_single)
    soft_avg = (((morph + ddpm + struct) / 3.0)[ev].argmax(1) == y[ev]).mean()

    curve = {"stack": {}, "rawpix": {}}
    for n in BUDGETS:
        s_accs, r_accs = [], []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            tr = _sample(pool, y, n, rng)
            # stacker on member probs
            st = StackingEnsemble(meta="logreg").fit([m[tr] for m in members_all], y[tr])
            s_accs.append(st.score([m[ev] for m in members_all], y[ev]))
            # raw-pixel logreg baseline
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(raw[tr], y[tr])
            r_accs.append((lr.predict(raw[ev]) == y[ev]).mean())
        curve["stack"][n] = (float(np.mean(s_accs)), float(np.std(s_accs)))
        curve["rawpix"][n] = (float(np.mean(r_accs)), float(np.std(r_accs)))
        print(f"n={n:5d}  stack={np.mean(s_accs)*100:5.2f}%  rawpix={np.mean(r_accs)*100:5.2f}%", flush=True)

    # peak: histgb stacker at full budget
    tr = pool
    peak = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    peak.fit(np.concatenate([m[tr] for m in members_all], 1), y[tr])
    peak_acc = (peak.predict(np.concatenate([m[ev] for m in members_all], 1)) == y[ev]).mean()

    print("\n=== TRACK 7: SEMI-SUPERVISED STACKING (label-efficiency) ===")
    print(f"label-free one-shot refs (eval): best single {best_single*100:.2f}%, soft-avg {soft_avg*100:.2f}%")
    print(f"stacker @ 100 labels : {curve['stack'][100][0]*100:.2f}%   (raw pixels: {curve['rawpix'][100][0]*100:.2f}%)")
    print(f"stacker @ 500 labels : {curve['stack'][500][0]*100:.2f}%   (raw pixels: {curve['rawpix'][500][0]*100:.2f}%)")
    print(f"stacker peak (histgb, 5k pool): {peak_acc*100:.2f}%")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = BUDGETS
    sm = [curve["stack"][n][0] * 100 for n in xs]
    ss = [curve["stack"][n][1] * 100 for n in xs]
    rm = [curve["rawpix"][n][0] * 100 for n in xs]
    rs = [curve["rawpix"][n][1] * 100 for n in xs]
    ax.errorbar(xs, sm, yerr=ss, marker="o", color="#1a5f9e", label="Stacked one-shot members (30-dim)")
    ax.errorbar(xs, rm, yerr=rs, marker="s", color="#c0392b", label="Raw pixels (784-dim)")
    ax.axhline(best_single * 100, ls="--", color="#888", lw=1)
    ax.text(xs[0], best_single * 100 + 0.6, f"best single member (label-free) {best_single*100:.1f}%",
            fontsize=7.5, color="#555")
    ax.set_xscale("log")
    ax.set_xlabel("Labeled MNIST images (meta-training budget)")
    ax.set_ylabel("MNIST eval accuracy (%)")
    ax.set_title("Track 7: label efficiency of stacking one-shot recognisers")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    fig_path = os.path.join(ROOT, "experiments", "reports", "figures", "fig5_label_efficiency.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180, facecolor="white")
    plt.close()
    print(f"[track7] saved {fig_path}")

    out = os.path.join(ROOT, "experiments", "reports", "ensemble_track_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "note": "NOT one-shot: meta-classifier trained on labeled MNIST. Base members "
                    "are the three one-shot recognisers used as fixed feature extractors.",
            "eval_set": "fixed 5000 held-out MNIST test images",
            "label_free_refs": {"best_single_member": float(best_single),
                                 "soft_average": float(soft_avg)},
            "budgets": BUDGETS,
            "stack_curve": {str(n): curve["stack"][n] for n in BUDGETS},
            "rawpix_curve": {str(n): curve["rawpix"][n] for n in BUDGETS},
            "peak_histgb_full_pool": float(peak_acc),
        }, f, indent=2)
    print(f"[track7] saved {out}")


if __name__ == "__main__":
    main()
