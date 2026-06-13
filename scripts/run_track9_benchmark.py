"""
Track 9 — Data-efficiency benchmark: MNIST record holder vs. my grand ensemble.

Three contestants, compared across stratified training-set budgets (exactly n/10 per
digit), each evaluated on the full 10k MNIST test set over several seeds:

  1. record_holder   — An et al. 2020 heterogeneous M3/M5/M7 majority vote (light repro,
                       the published record is 99.87%). src/track9/record_models.py
  2. grand_ensemble  — my best: stack of skeleton-fusion CNN + structural RF + a
                       diffusion-augmented CNN (diffusion member active only for n>=500).
  3. grand_ensemble_aug — contestant 2 with morphological augmentation added to members.

The grand ensemble stacks members leakage-free (members fit on a 70% slice of the budget,
the logistic-regression stacker fits on held-out 30% member probabilities), reusing the
Track 7 infrastructure (src/ensemble). The per-budget DDPM is trained once and its
generated bank is reused across seeds (a documented compute compromise).

Usage:
    python scripts/run_track9_benchmark.py --smoke      # fast wiring check (~minutes)
    python scripts/run_track9_benchmark.py              # full run (~hours, GPU)
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

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir, set_seed
from src.ensemble.members import (
    CNNMember, FusionCNNMember, StructuralRFMember, thin_batch, rich_features,
)
from src.ensemble.stacking import StackingEnsemble
from src.variants17.augment import augment_dataset
from src.track9.record_models import RecordHolderEnsemble
from src.track9.diffusion_member import train_and_generate

BUDGETS = [20, 50, 100, 200, 500, 1000, 2000, 5000]
DIFFUSION_THRESHOLD = 500          # diffusion member active only for n >= this
PUBLISHED_RECORD = 99.87           # An et al. 2020 heterogeneous ensemble, full 60k


# --------------------------------------------------------------------------- utils
def _select(labels, n, rng):
    """n stratified indices, exactly n//10 per class (asserted divisible by 10)."""
    assert n % 10 == 0, f"budget {n} must be divisible by 10 for balanced classes"
    per = n // 10
    sel = []
    for c in range(10):
        idx = np.where(labels == c)[0]
        sel.extend(rng.choice(idx, size=per, replace=False).tolist())
    return np.array(sel)


def _split(sel, labels, seed):
    """Per-class 70/30 split guaranteeing >=1 train and >=1 meta sample per class."""
    rng = np.random.default_rng(seed)
    tr, meta = [], []
    for c in range(10):
        idx = sel[labels[sel] == c]
        rng.shuffle(idx)
        k = max(1, min(len(idx) - 1, int(round(len(idx) * 0.7))))
        tr.extend(idx[:k].tolist())
        meta.extend(idx[k:].tolist())
    return np.array(tr), np.array(meta)


def _proba10(member, images_u8, **kw):
    """predict_proba padded to 10 columns (RF may drop absent classes)."""
    p = member.predict_proba(images_u8, **kw)
    if p.shape[1] == 10:
        return p
    classes = member.clf.classes_ if hasattr(member, "clf") else np.arange(p.shape[1])
    full = np.zeros((len(p), 10), dtype=np.float32)
    full[:, classes] = p
    return full


def _aug(raw_u8, labels, repeats, seed):
    """Original images + morphological (elastic+stroke) augmentations of them."""
    aug_f, aug_y = augment_dataset(raw_u8, labels, repeats=repeats, seed=seed)
    aug_u8 = (aug_f * 255).clip(0, 255).astype(np.uint8)
    return (np.concatenate([raw_u8, aug_u8], axis=0),
            np.concatenate([labels, aug_y], axis=0))


# ----------------------------------------------------------------- grand ensemble
def _grand_ensemble(sel, raw, thin, feats, y, test, device, args, gen_bank, augment):
    """Train the grand ensemble (optionally augmented) and return test accuracy."""
    tr, meta = _split(sel, y, seed=0)

    if augment:
        # expand the member-training slice; recompute skeleton/features for it
        a_raw, a_y = _aug(raw[tr], y[tr], args.aug_repeats, seed=hash(("aug", len(tr))) % 2**31)
        a_thin = thin_batch(a_raw)
        a_feats = rich_features(a_thin)
    else:
        a_raw, a_y, a_thin, a_feats = raw[tr], y[tr], thin[tr], feats[tr]

    members, order = {}, []
    members["fusion"] = FusionCNNMember(device, args.cnn_epochs).fit(a_raw, a_y, thin_u8=a_thin)
    members["struct_rf"] = StructuralRFMember().fit(a_raw, a_y, feats=a_feats)
    order += ["fusion", "struct_rf"]

    if gen_bank is not None:
        g_img, g_lbl = gen_bank
        d_img = np.concatenate([a_raw, g_img], axis=0)
        d_lbl = np.concatenate([a_y, g_lbl], axis=0)
        members["diffusion"] = CNNMember(device, args.cnn_epochs).fit(d_img, d_lbl)
        order.append("diffusion")

    def probs(name, imgs, **kw):
        m = members[name]
        if name == "fusion":
            return m.predict_proba(imgs, **kw)
        if name == "struct_rf":
            return _proba10(m, imgs, **kw)
        return m.predict_proba(imgs)

    meta_probs = {
        "fusion": members["fusion"].predict_proba(raw[meta], thin_u8=thin[meta]),
        "struct_rf": _proba10(members["struct_rf"], raw[meta], feats=feats[meta]),
    }
    test_probs = {
        "fusion": members["fusion"].predict_proba(test["raw"], thin_u8=test["thin"]),
        "struct_rf": _proba10(members["struct_rf"], test["raw"], feats=test["feats"]),
    }
    if "diffusion" in members:
        meta_probs["diffusion"] = members["diffusion"].predict_proba(raw[meta])
        test_probs["diffusion"] = members["diffusion"].predict_proba(test["raw"])

    st = StackingEnsemble(meta="logreg").fit([meta_probs[k] for k in order], y[meta])
    return st.score([test_probs[k] for k in order], test["y"])


# ----------------------------------------------------------------------- main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[track9] device={device}  smoke={args.smoke}", flush=True)

    raw_pool, y_pool = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_test, y_test = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")

    budgets = [100, 1000] if args.smoke else BUDGETS
    seeds = 1 if args.smoke else args.seeds

    # precompute test skeletons + features once (members reuse them every eval)
    t0 = time.perf_counter()
    test_thin = thin_batch(raw_test)
    test_feats = rich_features(test_thin)
    test = {"raw": raw_test, "thin": test_thin, "feats": test_feats, "y": y_test}
    print(f"[track9] precomputed test thin+features in {time.perf_counter()-t0:.0f}s", flush=True)

    results = {"record_holder": {}, "grand_ensemble": {}, "grand_ensemble_aug": {}}

    for n in budgets:
        # --- per-budget diffusion bank (trained once on a fixed budget-n subset) ---
        gen_bank = None
        diff_active = n >= DIFFUSION_THRESHOLD
        if diff_active:
            rng_b = np.random.default_rng(777)
            bank_sel = _select(y_pool, n, rng_b)
            t1 = time.perf_counter()
            gen_bank = train_and_generate(
                raw_pool[bank_sel], y_pool[bank_sel], device,
                n_per_class=args.gen_per_class, epochs=args.ddpm_epochs,
                dim=16, timesteps=args.ddpm_timesteps, sampling_steps=args.ddpm_sampling,
                seed=0, verbose=True,
            )
            print(f"  [n={n}] trained per-budget DDPM + bank "
                  f"({len(gen_bank[0])} imgs) in {time.perf_counter()-t1:.0f}s", flush=True)

        acc = {k: [] for k in results}
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            sel = _select(y_pool, n, rng)
            raw_s, y_s = raw_pool, y_pool   # index via sel
            thin_s = thin_batch(raw_pool[sel])
            feats_s = rich_features(thin_s)
            # remap sel -> local arrays so member code can index contiguously
            raw_loc = raw_pool[sel]; y_loc = y_pool[sel]
            loc = np.arange(len(sel))

            # contestant 1 — record holder
            set_seed(s)
            rec = RecordHolderEnsemble(device, epochs=args.record_epochs).fit(raw_loc, y_loc, seed=s)
            acc["record_holder"].append(rec.score(raw_test, y_test))

            # contestants 2 & 3 — grand ensemble (no-aug / aug)
            acc["grand_ensemble"].append(
                _grand_ensemble(loc, raw_loc, thin_s, feats_s, y_loc, test, device,
                                args, gen_bank, augment=False))
            acc["grand_ensemble_aug"].append(
                _grand_ensemble(loc, raw_loc, thin_s, feats_s, y_loc, test, device,
                                args, gen_bank, augment=True))

        for k in results:
            m, sd = float(np.mean(acc[k])), float(np.std(acc[k]))
            results[k][n] = {"mean": m, "std": sd, "diffusion": diff_active}
        print(f"  [n={n:5d}] record={results['record_holder'][n]['mean']*100:.2f}  "
              f"grand={results['grand_ensemble'][n]['mean']*100:.2f}  "
              f"grand_aug={results['grand_ensemble_aug'][n]['mean']*100:.2f}  "
              f"(diffusion={'on' if diff_active else 'off'})", flush=True)

    # --- full-data record-holder reference (once) ---
    full_ref = None
    if not args.smoke:
        print("[track9] training full-60k record-holder reference ...", flush=True)
        set_seed(0)
        rec_full = RecordHolderEnsemble(device, epochs=args.record_epochs).fit(raw_pool, y_pool, seed=0)
        full_ref = {"ensemble": rec_full.score(raw_test, y_test),
                    "members": rec_full.member_scores(raw_test, y_test)}
        print(f"[track9] full-data record holder: {full_ref['ensemble']*100:.2f}% "
              f"(members {full_ref['members']}); published record {PUBLISHED_RECORD}%", flush=True)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {"record_holder": ("#c0392b", "s", "-", "Record holder (An et al. M3/M5/M7)"),
              "grand_ensemble": ("#1a5f9e", "o", "--", "Grand ensemble (skeleton+BoF+diffusion)"),
              "grand_ensemble_aug": ("#27ae60", "^", "-", "Grand ensemble + augmentation")}
    for k, (c, mk, ls, lab) in styles.items():
        m = [results[k][n]["mean"] * 100 for n in budgets]
        e = [results[k][n]["std"] * 100 for n in budgets]
        ax.errorbar(budgets, m, yerr=e, marker=mk, color=c, ls=ls, capsize=2, label=lab)
    if full_ref is not None:
        ax.axhline(full_ref["ensemble"] * 100, color="#c0392b", ls=":", alpha=0.6,
                   label=f"Record holder, full 60k ({full_ref['ensemble']*100:.2f}%)")
    ax.axvline(DIFFUSION_THRESHOLD, color="gray", ls=":", alpha=0.4)
    ax.annotate(f"published record {PUBLISHED_RECORD}%", xy=(budgets[-1], PUBLISHED_RECORD),
                fontsize=7, color="#c0392b", ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Labeled MNIST images (budget n, balanced n/10 per digit)")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title("Track 9: data efficiency — record holder vs. grand ensemble")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, ls="--", alpha=0.4)
    fig_path = os.path.join(ROOT, "experiments", "reports", "figures", "fig7_track9_efficiency.png")
    ensure_dir(os.path.dirname(fig_path))
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180, facecolor="white")
    plt.close()
    print(f"[track9] saved {fig_path}", flush=True)

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "track9_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"budgets": budgets, "seeds": seeds,
                       "diffusion_threshold": DIFFUSION_THRESHOLD,
                       "published_record": PUBLISHED_RECORD,
                       "full_data_record_holder": full_ref,
                       "results": {k: {str(n): v for n, v in d.items()} for k, d in results.items()}},
                      f, indent=2)
        print(f"[track9] saved {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--record-epochs", type=int, default=40)
    p.add_argument("--cnn-epochs", type=int, default=12)
    p.add_argument("--aug-repeats", type=int, default=4)
    p.add_argument("--gen-per-class", type=int, default=200)
    p.add_argument("--ddpm-epochs", type=int, default=80)
    p.add_argument("--ddpm-timesteps", type=int, default=250)
    p.add_argument("--ddpm-sampling", type=int, default=50)
    args = p.parse_args()
    if args.smoke:
        args.record_epochs = 3
        args.cnn_epochs = 2
        args.aug_repeats = 2
        args.gen_per_class = 20
        args.ddpm_epochs = 5
        args.ddpm_sampling = 10
    main(args)
