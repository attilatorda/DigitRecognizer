"""
Track 9c severity sweep: does the agnostic-ensemble robustness gain hold across corruption
SEVERITIES (not just the single moderate level used in the main study)?

Members train once on the clean budget (n=2000, where the gain is largest and diffusion is
active); we then evaluate record_vote vs the agnostic conf_all ensemble across a grid of
severities per corruption. Test skeleton features are cached per (corruption, severity).

Usage:  python scripts/run_track9_severity.py [--smoke]
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
from src.track9.record_models import RecordModel, train_record_model, _predict_probs
from src.track9.diffusion_member import train_and_generate
from src.track9.corruptions import apply_corruption, denoise_for_skeleton
from src.track9 import combiners as cb

BUDGET = 2000
KERNELS = (3, 5, 7)
SEV = {
    "gaussian_noise": [0.25, 0.5, 0.75, 1.0],
    "blur": [0.7, 1.1, 1.5, 2.0],
    "stroke_dilate": [1, 2, 3],
    "occlusion": [6, 10, 14, 18],
}


def _select(labels, n, rng):
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


def _majority(pl):
    preds = np.stack([p.argmax(1) for p in pl], 0); summed = np.sum(pl, 0)
    out = np.empty(preds.shape[1], np.int64)
    for i in range(preds.shape[1]):
        v, c = np.unique(preds[:, i], return_counts=True)
        out[i] = (v[c == c.max()][np.argmax(summed[i, v[c == c.max()]])] if c.max() >= 2
                  else summed[i].argmax())
    return out


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_pool, y_pool = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_test, y_test = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    seeds = 1 if args.smoke else args.seeds
    grid = {k: (v[:2] if args.smoke else v) for k, v in SEV.items()}
    if args.smoke:
        raw_test, y_test = raw_test[:1500], y_test[:1500]
        grid = {"stroke_dilate": grid["stroke_dilate"], "gaussian_noise": grid["gaussian_noise"]}

    # cache corrupted test sets + features per (corruption, severity)
    t0 = time.perf_counter()
    crng = np.random.default_rng(12345)
    timg, tfeat = {}, {}
    for k, sevs in grid.items():
        for sv in sevs:
            ci = apply_corruption(raw_test, k, severity=sv, rng=crng)
            timg[(k, sv)] = ci
            tfeat[(k, sv)] = rich_features(thin_batch(denoise_for_skeleton(ci)))
    print(f"[sev] cached {len(timg)} (corruption,severity) test sets in {time.perf_counter()-t0:.0f}s", flush=True)

    res = {k: {sv: {"record_vote": [], "conf_all": []} for sv in sevs} for k, sevs in grid.items()}
    bsel = _select(y_pool, BUDGET, np.random.default_rng(777))
    bank = train_and_generate(raw_pool[bsel], y_pool[bsel], device, n_per_class=args.gen_per_class,
                              epochs=args.ddpm_epochs, dim=16, timesteps=args.ddpm_timesteps,
                              sampling_steps=args.ddpm_sampling, seed=0)

    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        sel = _select(y_pool, BUDGET, rng)
        raw_c, y_c = raw_pool[sel], y_pool[sel]
        feats_c = rich_features(thin_batch(raw_c))
        set_seed(s)
        rec = {k: train_record_model(RecordModel(k), raw_c, y_c, device, epochs=args.record_epochs, seed=s + k)
               for k in KERNELS}
        rf = StructuralRFMember().fit(raw_c, y_c, feats=feats_c)
        g_img, g_lbl = bank
        diff = CNNMember(device, args.cnn_epochs).fit(np.concatenate([raw_c, g_img]),
                                                      np.concatenate([y_c, g_lbl]))
        for k, sevs in grid.items():
            for sv in sevs:
                rp = [_predict_probs(rec[j], timg[(k, sv)], device) for j in KERNELS]
                members = rp + [_proba10(rf, images_u8=timg[(k, sv)], feats=tfeat[(k, sv)]),
                                diff.predict_proba(timg[(k, sv)])]
                res[k][sv]["record_vote"].append(float((_majority(rp) == y_test).mean()))
                res[k][sv]["conf_all"].append(cb.accuracy(cb.conf_vote(members), y_test))
        print(f"[sev] seed {s} done", flush=True)

    agg = {k: {str(sv): {ln: float(np.mean(res[k][sv][ln])) for ln in res[k][sv]} for sv in sevs}
           for k, sevs in grid.items()}

    # figure
    fig, axes = plt.subplots(1, len(grid), figsize=(3.4 * len(grid), 3.6), squeeze=False)
    for ax, (k, sevs) in zip(axes[0], grid.items()):
        ax.plot(sevs, [agg[k][str(sv)]["record_vote"] * 100 for sv in sevs], "s-", color="#c0392b", label="record holder")
        ax.plot(sevs, [agg[k][str(sv)]["conf_all"] * 100 for sv in sevs], "^-", color="#27ae60", label="+ struct + diff")
        ax.set_title(k, fontsize=10); ax.set_xlabel("severity"); ax.grid(True, ls="--", alpha=0.4)
    axes[0][0].set_ylabel("test accuracy (%)"); axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Track 9c severity sweep (n={BUDGET})", fontsize=11)
    plt.tight_layout()
    fp = os.path.join(ROOT, "experiments", "reports", "figures", "fig12_track9c_severity.png")
    ensure_dir(os.path.dirname(fp)); plt.savefig(fp, dpi=180, facecolor="white"); plt.close()
    print(f"[sev] saved {fp}", flush=True)

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "track9_severity_results.json")
        json.dump({"budget": BUDGET, "seeds": seeds, "results": agg}, open(out, "w"), indent=2)
        print(f"[sev] saved {out}", flush=True)


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
    a = p.parse_args()
    if a.smoke:
        a.seeds = 1; a.record_epochs = 3; a.cnn_epochs = 2
        a.gen_per_class = 20; a.ddpm_epochs = 5; a.ddpm_sampling = 10
    main(a)
