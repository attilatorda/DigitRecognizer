"""
Track 9 pivot — can structural/diffusion DIVERSITY improve the MNIST record holder?

Rather than trying to beat An et al. (the main benchmark showed it wins everywhere), we
ask the opposite: does ADDING my decorrelated members to the record holder lift it,
especially at low data? Three lines per stratified budget (n/10 per digit, 5 seeds, eval
on the 10k test set):

  record_vote  - An et al. M3/M5/M7 hard majority vote (the published method; baseline)
  record_stack - logreg stack of M3/M5/M7 probabilities (control: stacking vs voting)
  record_plus  - logreg stack of M3/M5/M7 + structural-RF + diffusion-CNN (the pivot)

If record_plus > record_vote (and > record_stack), structural/diffusion diversity genuinely
helps the SOTA in the low-data regime. Members are combined leakage-free (members fit on a
70% slice, stacker on held-out 30% member-probs). The per-budget DDPM bank (n>=500) is
reused across seeds. fp32 only.

Usage:
    python scripts/run_track9_augment.py --smoke
    python scripts/run_track9_augment.py
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
from src.ensemble.members import CNNMember, StructuralRFMember, thin_batch, rich_features
from src.ensemble.stacking import StackingEnsemble
from src.track9.record_models import RecordModel, train_record_model, _predict_probs
from src.track9.diffusion_member import train_and_generate

BUDGETS = [20, 50, 100, 200, 500, 1000, 2000, 5000]
DIFFUSION_THRESHOLD = 500
RECORD_KERNELS = (3, 5, 7)


def _select(labels, n, rng):
    assert n % 10 == 0
    per = n // 10
    sel = []
    for c in range(10):
        sel.extend(rng.choice(np.where(labels == c)[0], size=per, replace=False).tolist())
    return np.array(sel)


def _split(sel, labels, seed):
    rng = np.random.default_rng(seed)
    tr, meta = [], []
    for c in range(10):
        idx = sel[labels[sel] == c]
        rng.shuffle(idx)
        k = max(1, min(len(idx) - 1, int(round(len(idx) * 0.7))))
        tr.extend(idx[:k].tolist()); meta.extend(idx[k:].tolist())
    return np.array(tr), np.array(meta)


def _proba10(member, images_u8, **kw):
    p = member.predict_proba(images_u8, **kw)
    if p.shape[1] == 10:
        return p
    full = np.zeros((len(p), 10), dtype=np.float32)
    full[:, member.clf.classes_] = p
    return full


def _vote(probs_list):
    """Hard majority over argmax labels; 3-way ties broken by summed probability."""
    preds = np.stack([p.argmax(1) for p in probs_list], axis=0)
    summed = np.sum(probs_list, axis=0)
    out = np.empty(preds.shape[1], dtype=np.int64)
    for i in range(preds.shape[1]):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        top = counts.max()
        if top >= 2:
            cands = vals[counts == top]
            out[i] = cands[summed[i, cands].argmax()] if len(cands) > 1 else cands[0]
        else:
            out[i] = summed[i].argmax()
    return out


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[t9aug] device={device} smoke={args.smoke}", flush=True)

    raw_pool, y_pool = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_test, y_test = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    budgets = [100, 1000] if args.smoke else BUDGETS
    seeds = 1 if args.smoke else args.seeds

    t0 = time.perf_counter()
    test_thin = thin_batch(raw_test)
    test_feats = rich_features(test_thin)
    print(f"[t9aug] precomputed test thin+features in {time.perf_counter()-t0:.0f}s", flush=True)

    results = {"record_vote": {}, "record_stack": {}, "record_plus": {}}

    for n in budgets:
        gen_bank = None
        if n >= DIFFUSION_THRESHOLD:
            rng_b = np.random.default_rng(777)
            bsel = _select(y_pool, n, rng_b)
            gen_bank = train_and_generate(
                raw_pool[bsel], y_pool[bsel], device, n_per_class=args.gen_per_class,
                epochs=args.ddpm_epochs, dim=16, timesteps=args.ddpm_timesteps,
                sampling_steps=args.ddpm_sampling, seed=0)
            print(f"  [n={n}] DDPM bank ready ({len(gen_bank[0])} imgs)", flush=True)

        acc = {k: [] for k in results}
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            sel = _select(y_pool, n, rng)
            raw_loc, y_loc = raw_pool[sel], y_pool[sel]
            thin_loc = thin_batch(raw_loc)
            feats_loc = rich_features(thin_loc)
            loc = np.arange(len(sel))
            tr, meta = _split(loc, y_loc, seed=0)

            # record-holder members: train M3/M5/M7 on tr, collect meta + test probs
            set_seed(s)
            mp, tp = {}, {}
            for k in RECORD_KERNELS:
                m = RecordModel(k)
                train_record_model(m, raw_loc[tr], y_loc[tr], device,
                                   epochs=args.record_epochs, seed=s + k)
                mp[f"M{k}"] = _predict_probs(m, raw_loc[meta], device)
                tp[f"M{k}"] = _predict_probs(m, raw_test, device)
            rec_keys = [f"M{k}" for k in RECORD_KERNELS]

            # structural RF member
            rf = StructuralRFMember().fit(raw_loc[tr], y_loc[tr], feats=feats_loc[tr])
            mp["struct_rf"] = _proba10(rf, raw_loc[meta], feats=feats_loc[meta])
            tp["struct_rf"] = _proba10(rf, raw_test, feats=test_feats)

            # diffusion-augmented CNN member (n>=threshold)
            plus_keys = rec_keys + ["struct_rf"]
            if gen_bank is not None:
                g_img, g_lbl = gen_bank
                dcnn = CNNMember(device, args.cnn_epochs).fit(
                    np.concatenate([raw_loc[tr], g_img]), np.concatenate([y_loc[tr], g_lbl]))
                mp["diffusion"] = dcnn.predict_proba(raw_loc[meta])
                tp["diffusion"] = dcnn.predict_proba(raw_test)
                plus_keys = plus_keys + ["diffusion"]

            # three lines
            acc["record_vote"].append(float((_vote([tp[k] for k in rec_keys]) == y_test).mean()))
            st = StackingEnsemble(meta="logreg").fit([mp[k] for k in rec_keys], y_loc[meta])
            acc["record_stack"].append(st.score([tp[k] for k in rec_keys], y_test))
            stp = StackingEnsemble(meta="logreg").fit([mp[k] for k in plus_keys], y_loc[meta])
            acc["record_plus"].append(stp.score([tp[k] for k in plus_keys], y_test))

        for k in results:
            results[k][n] = {"mean": float(np.mean(acc[k])), "std": float(np.std(acc[k]))}
        d_plus = (results["record_plus"][n]["mean"] - results["record_vote"][n]["mean"]) * 100
        print(f"  [n={n:5d}] vote={results['record_vote'][n]['mean']*100:.2f}  "
              f"stack={results['record_stack'][n]['mean']*100:.2f}  "
              f"plus={results['record_plus'][n]['mean']*100:.2f}  "
              f"(plus-vote={d_plus:+.2f}pp)", flush=True)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sty = {"record_vote": ("#c0392b", "s", "-", "Record holder (majority vote)"),
           "record_stack": ("#999999", "x", ":", "Record holder (stacked, M3/M5/M7 only)"),
           "record_plus": ("#27ae60", "^", "-", "Record holder + structural + diffusion")}
    for k, (c, mk, ls, lab) in sty.items():
        m = [results[k][nn]["mean"] * 100 for nn in budgets]
        e = [results[k][nn]["std"] * 100 for nn in budgets]
        ax.errorbar(budgets, m, yerr=e, marker=mk, color=c, ls=ls, capsize=2, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Labeled MNIST images (budget n, balanced n/10 per digit)")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title("Track 9 pivot: does structural/diffusion diversity improve the record holder?")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, ls="--", alpha=0.4)
    fig_path = os.path.join(ROOT, "experiments", "reports", "figures", "fig8_track9_augment.png")
    ensure_dir(os.path.dirname(fig_path))
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180, facecolor="white")
    plt.close()
    print(f"[t9aug] saved {fig_path}", flush=True)

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "track9_augment_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"budgets": budgets, "seeds": seeds,
                       "diffusion_threshold": DIFFUSION_THRESHOLD,
                       "results": {k: {str(n): v for n, v in d.items()} for k, d in results.items()}},
                      f, indent=2)
        print(f"[t9aug] saved {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--record-epochs", type=int, default=40)
    p.add_argument("--cnn-epochs", type=int, default=12)
    p.add_argument("--gen-per-class", type=int, default=200)
    p.add_argument("--ddpm-epochs", type=int, default=80)
    p.add_argument("--ddpm-timesteps", type=int, default=250)
    p.add_argument("--ddpm-sampling", type=int, default=50)
    args = p.parse_args()
    if args.smoke:
        args.record_epochs = 3
        args.cnn_epochs = 2
        args.gen_per_class = 20
        args.ddpm_epochs = 5
        args.ddpm_sampling = 10
    main(args)
