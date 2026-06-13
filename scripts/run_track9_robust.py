"""
Track 9c — does confidence-weighted structural ensembling improve the MNIST record
holder's LOW-DATA CORRUPTION ROBUSTNESS?

Members are trained on the CLEAN full budget (the Track 9b fairness fix: no held-out
split, so the baseline keeps all its labels) and evaluated on a suite of corrupted test
sets. Skeleton features are stroke-thickness invariant by construction, so the structural
member is expected to be robust where the augmentation-trained CNN is brittle; a per-sample
confidence-weighted combiner lets it help on those corruptions while being ignored on the
ones it cannot handle (noise/occlusion).

Lines per (corruption, budget), 10 seeds, eval on the 10k test set:
  record_vote      - M3/M5/M7 majority vote (fair full-budget baseline)
  record_aug       - record holder retrained with corruptions in its augmentation (stronger
                     baseline; low budgets only)
  conf_all         - confidence-weighted ensemble of M3/M5/M7 + structural-RF + diffusion-CNN
  soft_all         - equal-weight ensemble of the same (combiner ablation: fixed vs adaptive)
  conf_drop_struct - conf_all without the structural member (member ablation)
  conf_drop_diff   - conf_all without the diffusion member (member ablation)

Headline metric: mean corruption accuracy (mCA) over the 4 non-clean corruptions.

Usage:
    python scripts/run_track9_robust.py --smoke
    python scripts/run_track9_robust.py
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
from src.common.utils import ensure_dir, set_seed
from src.ensemble.members import CNNMember, StructuralRFMember, thin_batch, rich_features
from src.track9.record_models import RecordModel, train_record_model, _predict_probs
from src.track9.diffusion_member import train_and_generate
from src.track9.corruptions import apply_corruption, denoise_for_skeleton, CORRUPTIONS
from src.track9 import combiners as cb

BUDGETS = [20, 50, 100, 200, 500, 1000, 2000]
DIFFUSION_THRESHOLD = 500
RECORD_AUG_MAX_N = 500          # corruption-augmented baseline only at low budgets
ERR_BUDGET = 100                # budget for the per-digit error analysis
RECORD_KERNELS = (3, 5, 7)
NONCLEAN = [c for c in CORRUPTIONS if c != "clean"]


def _select(labels, n, rng):
    assert n % 10 == 0
    sel = []
    for c in range(10):
        sel.extend(rng.choice(np.where(labels == c)[0], size=n // 10, replace=False).tolist())
    return np.array(sel)


def _proba10(member, **kw):
    p = member.predict_proba(**kw)
    if p.shape[1] == 10:
        return p
    full = np.zeros((len(p), 10), dtype=np.float32)
    full[:, member.clf.classes_] = p
    return full


def _majority(probs_list):
    """Hard majority over argmax; 3-way ties broken by summed probability."""
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


def _corrupt_train(raw_u8, labels, rng):
    """Corruption-aware training set: ~half the images get a random corruption."""
    out = raw_u8.copy()
    for i in range(len(out)):
        if rng.random() < 0.5:
            k = NONCLEAN[rng.integers(len(NONCLEAN))]
            out[i] = apply_corruption(out[i:i + 1], k, rng=rng)[0]
    return out


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[t9c] device={device} smoke={args.smoke}", flush=True)
    raw_pool, y_pool = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_test, y_test = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    if args.smoke:
        raw_test, y_test = raw_test[:1500], y_test[:1500]

    budgets = [100, 1000] if args.smoke else BUDGETS
    seeds = 1 if args.smoke else args.seeds
    corruptions = ["clean", "stroke_dilate", "gaussian_noise"] if args.smoke else CORRUPTIONS

    # --- cache corrupted test sets + their structural features (once) ---
    t0 = time.perf_counter()
    crng = np.random.default_rng(12345)
    test_imgs, test_feats = {}, {}
    for k in corruptions:
        ci = apply_corruption(raw_test, k, rng=crng)
        test_imgs[k] = ci
        test_feats[k] = rich_features(thin_batch(denoise_for_skeleton(ci)))
        print(f"[t9c] cached corrupted test '{k}' (+features)", flush=True)
    print(f"[t9c] test caches in {time.perf_counter()-t0:.0f}s", flush=True)

    lines = ["record_vote", "record_aug", "conf_all", "soft_all",
             "conf_drop_struct", "conf_drop_diff"]
    # acc[line][corruption][n] -> list over seeds ; mca[line][n] -> list over seeds
    acc = {ln: {c: {n: [] for n in budgets} for c in corruptions} for ln in lines}
    mca = {ln: {n: [] for n in budgets} for ln in lines}
    # per-digit accuracy for error analysis at ERR_BUDGET: errdig[line][corruption] -> (seeds,10)
    errdig = {ln: {c: [] for c in corruptions} for ln in ("record_vote", "conf_all")}

    for n in budgets:
        gen_bank = None
        if n >= DIFFUSION_THRESHOLD:
            bsel = _select(y_pool, n, np.random.default_rng(777))
            gen_bank = train_and_generate(
                raw_pool[bsel], y_pool[bsel], device, n_per_class=args.gen_per_class,
                epochs=args.ddpm_epochs, dim=16, timesteps=args.ddpm_timesteps,
                sampling_steps=args.ddpm_sampling, seed=0)
            print(f"  [n={n}] DDPM bank ready", flush=True)

        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            sel = _select(y_pool, n, rng)
            raw_c, y_c = raw_pool[sel], y_pool[sel]
            feats_c = rich_features(thin_batch(raw_c))

            # --- train members on the CLEAN full budget ---
            set_seed(s)
            rec = {}
            for k in RECORD_KERNELS:
                m = RecordModel(k)
                train_record_model(m, raw_c, y_c, device, epochs=args.record_epochs, seed=s + k)
                rec[k] = m
            rf = StructuralRFMember().fit(raw_c, y_c, feats=feats_c)
            diff = None
            if gen_bank is not None:
                g_img, g_lbl = gen_bank
                diff = CNNMember(device, args.cnn_epochs).fit(
                    np.concatenate([raw_c, g_img]), np.concatenate([y_c, g_lbl]))

            # --- corruption-aware baseline (low budgets only) ---
            rec_aug = None
            if n <= RECORD_AUG_MAX_N:
                arng = np.random.default_rng(900 + s)
                raw_aug = _corrupt_train(raw_c, y_c, arng)
                rec_aug = {}
                for k in RECORD_KERNELS:
                    m = RecordModel(k)
                    train_record_model(m, raw_aug, y_c, device, epochs=args.record_epochs, seed=s + k)
                    rec_aug[k] = m

            # --- evaluate every line on every corruption ---
            per_seed_mca = {ln: [] for ln in lines}
            for c in corruptions:
                rp = [_predict_probs(rec[k], test_imgs[c], device) for k in RECORD_KERNELS]
                rfp = _proba10(rf, images_u8=test_imgs[c], feats=test_feats[c])
                members = rp + [rfp]
                no_struct = list(rp)
                no_diff = rp + [rfp]
                if diff is not None:
                    dp = diff.predict_proba(test_imgs[c])
                    members = members + [dp]
                    no_struct = no_struct + [dp]

                line_pred = {
                    "record_vote": _majority(rp),
                    "conf_all": cb.preds(cb.conf_vote(members)),
                    "soft_all": cb.preds(cb.soft_vote(members)),
                    "conf_drop_struct": cb.preds(cb.conf_vote(no_struct)) if diff is not None else _majority(rp),
                    "conf_drop_diff": cb.preds(cb.conf_vote(no_diff)),
                }
                if rec_aug is not None:
                    line_pred["record_aug"] = _majority(
                        [_predict_probs(rec_aug[k], test_imgs[c], device) for k in RECORD_KERNELS])

                for ln, pred in line_pred.items():
                    a = float((pred == y_test).mean())
                    acc[ln][c][n].append(a)
                    if c != "clean":
                        per_seed_mca[ln].append(a)
                    if n == ERR_BUDGET and ln in errdig:
                        errdig[ln][c].append([float((pred[y_test == d] == d).mean()) for d in range(10)])

            for ln in lines:
                if per_seed_mca[ln]:
                    mca[ln][n].append(float(np.mean(per_seed_mca[ln])))

            tag = " ".join(
                f"{ln.split('_')[0][:4]}{ln.split('_')[-1][:3]}={np.mean(mca[ln][n][-1:]) * 100:.1f}"
                for ln in ("record_vote", "conf_all") if mca[ln][n])
            print(f"  [n={n:4d} s={s}] mCA {tag}  (conf-rec="
                  f"{(np.mean(mca['conf_all'][n][-1:]) - np.mean(mca['record_vote'][n][-1:]))*100:+.2f}pp)",
                  flush=True)

        d = (np.mean(mca["conf_all"][n]) - np.mean(mca["record_vote"][n])) * 100
        win = int(np.sum(np.array(mca["conf_all"][n]) > np.array(mca["record_vote"][n])))
        print(f"  [n={n:4d}] mCA conf_all={np.mean(mca['conf_all'][n])*100:.2f}  "
              f"record_vote={np.mean(mca['record_vote'][n])*100:.2f}  "
              f"delta={d:+.2f}pp  ({win}/{len(mca['conf_all'][n])} seeds improved)", flush=True)

    # --- aggregate + save ---
    def agg(d):
        return {"mean": float(np.mean(d)), "std": float(np.std(d)), "n": len(d)} if d else None
    out = {
        "budgets": budgets, "seeds": seeds, "corruptions": corruptions,
        "diffusion_threshold": DIFFUSION_THRESHOLD, "err_budget": ERR_BUDGET,
        "accuracy": {ln: {c: {str(n): agg(acc[ln][c][n]) for n in budgets} for c in corruptions}
                     for ln in lines},
        "mca": {ln: {str(n): agg(mca[ln][n]) for n in budgets} for ln in lines},
        "mca_paired": {ln: {str(n): mca[ln][n] for n in budgets} for ln in lines},
        "err_per_digit": {ln: {c: (np.mean(errdig[ln][c], axis=0).tolist() if errdig[ln][c] else None)
                                for c in corruptions} for ln in errdig},
    }
    if not args.smoke:
        path = os.path.join(ROOT, "experiments", "reports", "track9_robust_results.json")
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[t9c] saved {path}", flush=True)
    else:
        print("[t9c] smoke OK; results not written", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--record-epochs", type=int, default=40)
    p.add_argument("--cnn-epochs", type=int, default=12)
    p.add_argument("--gen-per-class", type=int, default=200)
    p.add_argument("--ddpm-epochs", type=int, default=80)
    p.add_argument("--ddpm-timesteps", type=int, default=250)
    p.add_argument("--ddpm-sampling", type=int, default=50)
    a = p.parse_args()
    if a.smoke:
        a.seeds = 1
        a.record_epochs = 3
        a.cnn_epochs = 2
        a.gen_per_class = 20
        a.ddpm_epochs = 5
        a.ddpm_sampling = 10
    main(a)
