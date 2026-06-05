"""
Track 6 v3 — classifier zoo + ensemble on the rich structural features.

v2 (kNN, 8704-image bank, 88-dim) reached 65.43%. v3 tests whether the bottleneck
is the classifier (kNN is weak) rather than the features, by training a panel of
stronger nonlinear classifiers on the SAME 88-dim vectors plus a soft-voting ensemble.
The reference bank's DDPM half now uses the high-capacity dim=32 generated images.

Usage:
    python scripts/run_structural_v3_experiment.py            # bank=both, full 10K
    python scripts/run_structural_v3_experiment.py --smoke
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir
# Reuse v2 featurization helpers (importing does not run its main())
from run_structural_v2_experiment import _featurize, _load_bank


def main(args):
    t0 = time.perf_counter()

    bank_images, y_train = _load_bank(args.bank)
    print(f"[v3] reference bank: {args.bank}  ({len(bank_images)} images)")
    X_train = _featurize(bank_images, tag="bank")

    test_images, test_labels = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    if args.smoke:
        test_images, test_labels = test_images[:args.smoke_n], test_labels[:args.smoke_n]
    X_test = _featurize(test_images, tag="test")

    # Each classifier gets its own StandardScaler via a pipeline.
    def sc(clf):
        return make_pipeline(StandardScaler(), clf)

    models = {
        "logreg":  sc(LogisticRegression(max_iter=2000, C=1.0)),
        "knn":     sc(KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "rf":      sc(RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0)),
        "histgb":  sc(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, random_state=0)),
        "mlp":     sc(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=0)),
    }

    accs = {}
    fitted = {}
    for name, model in models.items():
        tt = time.perf_counter()
        model.fit(X_train, y_train)
        acc = float((model.predict(X_test) == test_labels).mean())
        accs[name] = acc
        fitted[name] = model
        print(f"[v3] {name:8s} {acc*100:.2f}%  ({time.perf_counter()-tt:.0f}s)", flush=True)

    # Soft-voting ensemble of the probabilistic learners
    ens = VotingClassifier(
        estimators=[(n, fitted[n]) for n in ("logreg", "rf", "histgb", "mlp")],
        voting="soft", n_jobs=-1,
    )
    ens.fit(X_train, y_train)
    accs["ensemble"] = float((ens.predict(X_test) == test_labels).mean())
    print(f"[v3] {'ensemble':8s} {accs['ensemble']*100:.2f}%", flush=True)

    best_name = max(accs, key=accs.get)
    best = accs[best_name]
    elapsed = time.perf_counter() - t0

    print("\n=== TRACK 6 v3 RESULTS ===")
    for name in ("logreg", "knn", "rf", "histgb", "mlp", "ensemble"):
        print(f"  {name:8s} {accs[name]*100:6.2f}%")
    print(f"\nbest: {best_name} {best*100:.2f}%")
    print(f"v2 (kNN, 88-dim)      : 65.43%   (+{best*100-65.43:.2f}pp)")
    print(f"proto CNN baseline    : 77.46%   ({best*100-77.46:+.2f}pp)")
    print(f"total time            : {elapsed:.0f}s")

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "structural_v3_results.json")
        ensure_dir(os.path.dirname(out))
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "bank": args.bank, "n_reference": len(bank_images),
                "accuracies": accs, "best": best_name, "best_acc": best,
                "v2_acc": 0.6543, "proto_baseline": 0.7746, "elapsed_s": elapsed,
            }, f, indent=2)
        print(f"[v3] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="both", choices=["morphological", "ddpm", "both"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=500)
    main(p.parse_args())
