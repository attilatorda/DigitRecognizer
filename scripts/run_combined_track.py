"""
Track 7 (redefined) — combined best-of-everything supervised system, data-efficiency study.

Combines the strongest pieces of the prior tracks on the MNIST training set:
  CNN (raw)  +  Fusion CNN (raw + Guo-Hall 'thin' skeleton)  +  structural RF (93-dim)
stacked by a meta-classifier (src/ensemble/stacking.py), and an EXEMPLAR SELECTOR that
finds the training examples that "fit a class strongest".

Demonstrated in the LOW-LABEL regime (where it matters; at 60k a CNN is already ~99%):
for each label budget n and selection rule (random vs prototypical), train the members on
a 70% slice, fit the stacker on the members' probs over the held-out 30% (leakage-free),
and evaluate members + stack on the 10k MNIST test.

Two stories: (1) combining > best single member at low n; (2) prototypical selection >
random at low n.

Usage:
    python scripts/run_combined_track.py            # full
    python scripts/run_combined_track.py --smoke    # quick
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir
from src.ensemble.members import (
    CNNMember, FusionCNNMember, StructuralRFMember, thin_batch, rich_features,
)
from src.ensemble.exemplar import prototypicality_scores, select_top_k_per_class
from src.ensemble.stacking import StackingEnsemble

BUDGETS = [100, 250, 500, 1000, 2500, 5000]


def _select(mode, labels, scores, n, rng):
    """Return n indices, n//10 per class.
    'random'       — uniform baseline
    'prototypical' — top-k by score (easy/central; for representing a class / seeding gen)
    'hard'         — bottom-k by score (atypical/boundary; the opposite end of the selector)
    """
    per = max(1, n // 10)
    sel = []
    for c in range(10):
        idx = np.where(labels == c)[0]
        if mode == "prototypical":
            idx = idx[np.argsort(-scores[idx])][:per]
        elif mode == "hard":
            idx = idx[np.argsort(scores[idx])][:per]
        else:
            idx = rng.choice(idx, size=min(per, len(idx)), replace=False)
        sel.extend(idx.tolist())
    return np.array(sel)


def _eval_config(sel_idx, pool, cache, test, device, epochs, seed):
    """Train members on a 70% slice of sel_idx, stack on 30%, eval on test."""
    raw, thin, feats, y = pool
    tr, meta = train_test_split(sel_idx, test_size=0.3, random_state=seed,
                                stratify=y[sel_idx])
    members = {
        "cnn_raw":  CNNMember(device, epochs).fit(raw[tr], y[tr]),
        "fusion":   FusionCNNMember(device, epochs).fit(raw[tr], y[tr], thin_u8=thin[tr]),
        "struct_rf": StructuralRFMember().fit(raw[tr], y[tr], feats=feats[tr]),
    }
    # member probs (cached test features)
    def probs(m, on_test):
        if on_test:
            if isinstance(m, FusionCNNMember): return m.predict_proba(test["raw"], thin_u8=test["thin"])
            if isinstance(m, StructuralRFMember): return m.predict_proba(test["raw"], feats=test["feats"])
            return m.predict_proba(test["raw"])
        if isinstance(m, FusionCNNMember): return m.predict_proba(raw[meta], thin_u8=thin[meta])
        if isinstance(m, StructuralRFMember): return m.predict_proba(raw[meta], feats=feats[meta])
        return m.predict_proba(raw[meta])

    test_probs = {k: probs(m, True) for k, m in members.items()}
    meta_probs = {k: probs(m, False) for k, m in members.items()}
    yt = test["y"]

    accs = {k: float((p.argmax(1) == yt).mean()) for k, p in test_probs.items()}
    # stack (leakage-free: meta trained on held-out 30%)
    order = ["cnn_raw", "fusion", "struct_rf"]
    st = StackingEnsemble(meta="logreg").fit([meta_probs[k] for k in order], y[meta])
    accs["stacked"] = st.score([test_probs[k] for k in order], yt)
    accs["best_single"] = max(accs[k] for k in order)
    return accs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[combined] device={device}")
    rng0 = np.random.default_rng(0)

    raw_pool, y_pool = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_test, y_test = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    budgets = [100, 500] if args.smoke else BUDGETS
    seeds = 1 if args.smoke else 3
    pool_n = 8000 if args.smoke else args.pool_n

    # cap the pool (coreset source)
    keep = rng0.choice(len(y_pool), size=min(pool_n, len(y_pool)), replace=False)
    raw_pool, y_pool = raw_pool[keep], y_pool[keep]

    # --- precompute skeletons + features once ---
    t0 = time.perf_counter()
    pool_thin = thin_batch(raw_pool);  test_thin = thin_batch(raw_test)
    pool_feats = rich_features(pool_thin);  test_feats = rich_features(test_thin)
    print(f"[combined] precomputed thin+features in {time.perf_counter()-t0:.0f}s "
          f"(pool={len(y_pool)}, test={len(y_test)})", flush=True)
    pool = (raw_pool, pool_thin, pool_feats, y_pool)
    test = {"raw": raw_test, "thin": test_thin, "feats": test_feats, "y": y_test}

    # --- exemplar scores (coreset framing: score the pool with a CNN on its labels) ---
    score_cnn = CNNMember(device, epochs=args.epochs).fit(raw_pool, y_pool)
    scores = prototypicality_scores(score_cnn, raw_pool, y_pool)
    print(f"[combined] scored pool prototypicality (range {scores.min():.2f}..{scores.max():.2f})", flush=True)

    modes = ("random", "prototypical") if args.smoke else ("random", "prototypical", "hard")
    results = {m: {} for m in modes}
    for mode in modes:
        for n in budgets:
            accs_runs = []
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                sel = _select(mode, y_pool, scores, n, rng)
                accs_runs.append(_eval_config(sel, pool, None, test, device, args.epochs, s))
            agg = {k: (float(np.mean([r[k] for r in accs_runs])),
                       float(np.std([r[k] for r in accs_runs]))) for k in accs_runs[0]}
            results[mode][n] = agg
            print(f"  [{mode:12s} n={n:5d}] cnn={agg['cnn_raw'][0]*100:.2f} "
                  f"fusion={agg['fusion'][0]*100:.2f} rf={agg['struct_rf'][0]*100:.2f} "
                  f"stacked={agg['stacked'][0]*100:.2f}", flush=True)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7, 4.3))
    xs = budgets
    def line(mode, key, **kw):
        m = [results[mode][n][key][0]*100 for n in xs]
        e = [results[mode][n][key][1]*100 for n in xs]
        ax.errorbar(xs, m, yerr=e, capsize=2, **kw)
    line("random", "stacked", marker="o", color="#1a5f9e", label="Combined, random")
    if "hard" in results:
        line("hard", "stacked", marker="^", color="#27ae60", label="Combined, hard/atypical")
    line("prototypical", "stacked", marker="v", color="#e67e22", ls="--", label="Combined, prototypical")
    line("random", "best_single", marker="s", color="#c0392b", ls=":", label="Best single member (random)")
    ax.set_xscale("log"); ax.set_xlabel("Labeled MNIST images (budget n)")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title("Track 7: combined system + exemplar selection (data efficiency)")
    ax.legend(loc="lower right", fontsize=8); ax.grid(True, ls="--", alpha=0.4)
    fig_path = os.path.join(ROOT, "experiments", "reports", "figures", "fig6_combined_efficiency.png")
    ensure_dir(os.path.dirname(fig_path)); plt.tight_layout()
    plt.savefig(fig_path, dpi=180, facecolor="white"); plt.close()
    print(f"[combined] saved {fig_path}")

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "combined_track_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"budgets": budgets, "seeds": seeds, "pool_n": int(len(y_pool)),
                       "members": ["cnn_raw", "fusion", "struct_rf"],
                       "results": results}, f, indent=2)
        print(f"[combined] saved {out}")

    # headline summary
    n0 = budgets[0]
    print("\n=== TRACK 7 SUMMARY (lowest budget) ===")
    print(f"n={n0}: best single {results['random'][n0]['best_single'][0]*100:.2f}%  "
          f"-> combined(random) {results['random'][n0]['stacked'][0]*100:.2f}%  "
          f"-> combined(prototypical) {results['prototypical'][n0]['stacked'][0]*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--pool-n", type=int, default=60000)
    main(p.parse_args())
