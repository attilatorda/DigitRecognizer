"""
Diagnostic: is an ensemble of the three one-shot strategies worth building?

Gets per-image MNIST-test predictions from three diverse predictors:
  A. Morphological proto  (neural, learned features, morphological augmentation)
  B. DDPM proto           (neural, learned features, diffusion-generated data)
  C. Structural RF        (skeleton-graph bag-of-features, no deep learning)

Then reports the numbers that decide whether to build an ensemble track:
  - per-method accuracy
  - pairwise agreement and "both wrong" rates (error correlation)
  - ORACLE ceiling: accuracy if a perfect combiner always picked a correct member
  - realizable 3-way majority vote (ties -> strongest member)

If oracle >> best single method, the methods err on different images and an
ensemble has real headroom. If oracle ~= best single, they are too correlated.

Usage:  python scripts/analyze_ensemble_potential.py
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.data_io import load_mnist_idx
from src.variants17.label_schema import CLASS17_TO_DIGIT10
from src.variants17.train_variants17_proto import (
    EmbeddingCNN, compute_prototypes, get_mnist_predictions,
)
from run_structural_v2_experiment import _featurize, _load_bank


def _proto_method_preds(ckpt_dir, support_u8_or01, support_labels17, test_images, device):
    """Per-image majority-vote prediction over a proto method's 5 seed checkpoints."""
    support01 = support_u8_or01.astype(np.float32)
    if support01.max() > 1.0:
        support01 = support01 / 255.0
    seed_preds = []
    for fn in sorted(f for f in os.listdir(ckpt_dir) if f.startswith("best_seed")):
        model = EmbeddingCNN(emb_dim=64).to(device)
        ck = torch.load(os.path.join(ckpt_dir, fn), map_location=device)
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        protos = compute_prototypes(model, support01, support_labels17, device)
        pred10, _ = get_mnist_predictions(model, test_images, protos, device)
        seed_preds.append(pred10)
    seed_preds = np.stack(seed_preds, axis=0)              # (5, N)
    # per-image majority vote across seeds
    maj = np.array([np.bincount(seed_preds[:, i], minlength=10).argmax()
                    for i in range(seed_preds.shape[1])])
    return maj


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ensemble-diag] device={device}")
    test_images, y = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    N = len(y)

    # --- A. Morphological proto (support = 17 CultiVar templates) ---
    tmpl = np.load(os.path.join(ROOT, "data/processed/mnist17_variants/train_images.npy"))
    tmpl_lab = np.load(os.path.join(ROOT, "data/processed/mnist17_variants/train_labels17.npy"))
    predA = _proto_method_preds(
        os.path.join(ROOT, "experiments/checkpoints/oneshot_comparison/proto"),
        tmpl, tmpl_lab, test_images, device)
    print(f"[A] morphological proto : {(predA==y).mean()*100:.2f}%")

    # --- B. DDPM proto (support = generated images) ---
    gen = np.load(os.path.join(ROOT, "experiments/checkpoints/diffusion_aug/generated_images.npy"))
    gen_lab = np.load(os.path.join(ROOT, "experiments/checkpoints/diffusion_aug/generated_labels.npy"))
    predB = _proto_method_preds(
        os.path.join(ROOT, "experiments/checkpoints/diffusion_experiment/ddpm_proto"),
        gen, gen_lab, test_images, device)
    print(f"[B] DDPM proto          : {(predB==y).mean()*100:.2f}%")

    # --- C. Structural RF (features from the combined bank) ---
    bank_imgs, bank_y = _load_bank("both")
    Xtr = _featurize(bank_imgs, tag="bank")
    Xte = _featurize(test_images, tag="test")
    rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0))
    rf.fit(Xtr, bank_y)
    predC = rf.predict(Xte).astype(int)
    print(f"[C] structural RF       : {(predC==y).mean()*100:.2f}%")

    preds = {"A_morph": predA, "B_ddpm": predB, "C_struct": predC}
    accs = {k: (v == y).mean() for k, v in preds.items()}
    best = max(accs.values())

    # --- pairwise error correlation ---
    print("\n[ensemble-diag] pairwise (agreement | both-wrong):")
    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = preds[names[i]], preds[names[j]]
            agree = (a == b).mean()
            both_wrong = ((a != y) & (b != y)).mean()
            print(f"  {names[i]} vs {names[j]}: agree={agree*100:.1f}%  both_wrong={both_wrong*100:.1f}%")

    # --- oracle ceiling: at least one member correct ---
    any_correct = ((predA == y) | (predB == y) | (predC == y)).mean()

    # --- realizable 3-way majority vote (ties -> strongest member = best single) ---
    strongest = max(accs, key=accs.get)
    stack = np.stack([predA, predB, predC], axis=0)         # (3, N)
    maj = np.empty(N, dtype=int)
    for i in range(N):
        counts = np.bincount(stack[:, i], minlength=10)
        top = counts.max()
        maj[i] = counts.argmax() if top >= 2 else preds[strongest][i]
    maj_acc = (maj == y).mean()

    print("\n=== VERDICT ===")
    print(f"best single method     : {best*100:.2f}%  ({max(accs,key=accs.get)})")
    print(f"3-way majority vote    : {maj_acc*100:.2f}%  ({(maj_acc-best)*100:+.2f}pp vs best)")
    print(f"ORACLE ceiling         : {any_correct*100:.2f}%  (at least one member correct)")
    headroom = any_correct - best
    print(f"oracle headroom        : {headroom*100:+.2f}pp")
    if headroom > 0.05:
        print("-> Methods err on DIFFERENT images: an ensemble track is worth building.")
    elif headroom > 0.02:
        print("-> Modest decorrelation: an ensemble may give a small gain.")
    else:
        print("-> Methods too correlated: an ensemble is unlikely to help.")


if __name__ == "__main__":
    main()
