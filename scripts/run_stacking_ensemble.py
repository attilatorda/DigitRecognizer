"""
Ensembling the three one-shot strategies — and the boundary where one-shot ends.

Consumes the per-member probability matrices from build_member_predictions.py
(morphological proto, DDPM proto, structural RF — each (10000,10) on MNIST test).

KEY DISTINCTION (one-shot validity):
  - LABEL-FREE combiners (majority vote, soft-average) stay in the one-shot regime —
    they use NO MNIST labels. This is a legitimate one-shot ensemble.
  - A STACKING meta-classifier must be TRAINED on labeled MNIST images. That exits the
    one-shot premise entirely (it is a small supervised classifier over the members'
    probability vectors). We report it only as a *supervised upper reference*, NOT a
    one-shot result — it is not comparable to the 17-template members.

The diagnostic (analyze_ensemble_potential.py) found a large oracle headroom (~91% if a
perfect selector picked a correct member). This script shows that headroom can only be
realised WITH labels: the label-free ensemble captures little of it, while the supervised
stacker captures most — confirming that the gap is a supervision boundary, not a free lunch.

Stacking is evaluated leakage-free (base members never saw MNIST; the meta-classifier is
trained on half the MNIST test and evaluated on the disjoint half).

Usage:  python scripts/run_stacking_ensemble.py
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    npz = np.load(os.path.join(ROOT, "experiments", "reports", "member_probs.npz"))
    morph, ddpm, struct, y = npz["morph"], npz["ddpm"], npz["struct"], npz["y"]
    members = {"morph_proto": morph, "ddpm_proto": ddpm, "struct_rf": struct}

    # Meta-features: concatenate the three 10-dim probability vectors -> 30-dim.
    X = np.concatenate([morph, ddpm, struct], axis=1)        # (N, 30)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.5, random_state=0, stratify=y)

    print("=== per-member accuracy (held-out half) ===")
    for name, p in members.items():
        print(f"  {name:12s} {(p[te].argmax(1) == y[te]).mean()*100:.2f}%")
    best_single = max((members[n][te].argmax(1) == y[te]).mean() for n in members)

    # Naive soft-average
    avg = (morph + ddpm + struct) / 3.0
    avg_acc = (avg[te].argmax(1) == y[te]).mean()

    # Stacking meta-classifiers
    logreg = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
    histgb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[tr], y[tr])
    acc_lr = (logreg.predict(X[te]) == y[te]).mean()
    acc_gb = (histgb.predict(X[te]) == y[te]).mean()

    # Oracle ceiling on the held-out half
    oracle = ((morph[te].argmax(1) == y[te]) |
              (ddpm[te].argmax(1) == y[te]) |
              (struct[te].argmax(1) == y[te])).mean()

    best_stack = max(acc_lr, acc_gb)
    print("\n=== ONE-SHOT ENSEMBLE (label-free — valid one-shot) ===")
    print(f"best single member     : {best_single*100:.2f}%")
    print(f"soft-average (label-free): {avg_acc*100:.2f}%  ({(avg_acc-best_single)*100:+.2f}pp)  <- legitimate one-shot ensemble")
    print("\n=== SUPERVISED UPPER REFERENCE (NOT one-shot — trained on 5k MNIST labels) ===")
    print(f"stacking logreg        : {acc_lr*100:.2f}%")
    print(f"stacking histgb        : {acc_gb*100:.2f}%")
    print(f"argmax-oracle ceiling  : {oracle*100:.2f}%")
    print("\nVerdict: the ~10pp oracle headroom is real but only realisable WITH labels.")
    print(f"Label-free, the ensemble adds just {(avg_acc-best_single)*100:+.2f}pp ({best_single*100:.1f}->{avg_acc*100:.1f}%).")
    print(f"The supervised stacker ({best_stack*100:.1f}%) exits the one-shot regime and is not comparable.")

    out = os.path.join(ROOT, "experiments", "reports", "stacking_ensemble_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "eval": "held-out 50% of MNIST test (5000 images)",
            "per_member": {n: float((members[n][te].argmax(1) == y[te]).mean()) for n in members},
            "one_shot_valid": {
                "best_single_member": float(best_single),
                "label_free_soft_average": float(avg_acc),
            },
            "NOT_one_shot_supervised_reference": {
                "note": "meta-classifier trained on 5000 labeled MNIST images; exits one-shot",
                "stacking_logreg": float(acc_lr),
                "stacking_histgb": float(acc_gb),
            },
            "argmax_oracle_ceiling": float(oracle),
        }, f, indent=2)
    print(f"[ensemble] saved {out}")


if __name__ == "__main__":
    main()
