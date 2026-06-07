"""
Step 2 toward an ensemble track: seed-ensemble the proto members and export each
member's per-image probability matrix (the inputs a stacking combiner will consume).

For each proto method we compare:
  - single-seed mean accuracy (what the paper reports)
  - hard majority vote over the 5 seeds
  - distance-averaged soft ensemble (average the 17-way distances, then 17->10)

Exports experiments/reports/member_probs.npz with (10000, 10) probability matrices
for morphological proto, DDPM proto, and structural RF, plus the test labels.

Usage:  python scripts/build_member_predictions.py
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
from src.variants17.train_variants17_proto import EmbeddingCNN, compute_prototypes
from run_structural_v2_experiment import _featurize, _load_bank

DIGIT_OF = np.array([CLASS17_TO_DIGIT10[c] for c in range(17)])  # (17,) class->digit


def _seed_distances(ckpt_dir, support01, support_y17, test_images, device):
    """Return list of (N,17) distance matrices, one per seed checkpoint."""
    loader_imgs = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1).to(device) / 255.0
    out = []
    for fn in sorted(f for f in os.listdir(ckpt_dir) if f.startswith("best_seed")):
        model = EmbeddingCNN(emb_dim=64).to(device)
        ck = torch.load(os.path.join(ckpt_dir, fn), map_location=device)
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        model.eval()
        protos = compute_prototypes(model, support01, support_y17, device)
        with torch.no_grad():
            dists = []
            for i in range(0, len(loader_imgs), 512):
                z = model(loader_imgs[i:i+512])
                dists.append(torch.cdist(z, protos).cpu().numpy())
        out.append(np.concatenate(dists, axis=0))   # (N,17)
    return out


def _dist17_to_digit10_scores(dist17):
    """(N,17) distances -> (N,10) scores (higher=better) via min-distance per digit."""
    N = dist17.shape[0]
    scores = np.full((N, 10), -np.inf, dtype=np.float32)
    for c in range(17):
        d = DIGIT_OF[c]
        scores[:, d] = np.maximum(scores[:, d], -dist17[:, c])
    return scores


def _softmax(x, scale=10.0):
    z = x * scale
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _proto_member(name, ckpt_dir, support_u8_or01, support_y17, test_images, y, device):
    support01 = support_u8_or01.astype(np.float32)
    if support01.max() > 1.0:
        support01 = support01 / 255.0
    per_seed = _seed_distances(ckpt_dir, support01, support_y17, test_images, device)

    # single-seed mean accuracy
    single_accs = [(_dist17_to_digit10_scores(d).argmax(1) == y).mean() for d in per_seed]
    # hard majority vote
    seed_preds = np.stack([_dist17_to_digit10_scores(d).argmax(1) for d in per_seed], 0)
    hard = np.array([np.bincount(seed_preds[:, i], minlength=10).argmax() for i in range(len(y))])
    # distance-averaged soft ensemble
    avg_scores = _dist17_to_digit10_scores(np.mean(per_seed, axis=0))
    soft = avg_scores.argmax(1)
    probs = _softmax(avg_scores)

    print(f"[{name}] single-seed mean {np.mean(single_accs)*100:.2f}%  "
          f"hard-vote {(hard==y).mean()*100:.2f}%  soft-ens {(soft==y).mean()*100:.2f}%")
    return probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_images, y = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")

    tmpl = np.load(os.path.join(ROOT, "data/processed/mnist17_variants/train_images.npy"))
    tmpl_lab = np.load(os.path.join(ROOT, "data/processed/mnist17_variants/train_labels17.npy"))
    probsA = _proto_member("morph_proto",
        os.path.join(ROOT, "experiments/checkpoints/oneshot_comparison/proto"),
        tmpl, tmpl_lab, test_images, y, device)

    gen = np.load(os.path.join(ROOT, "experiments/checkpoints/diffusion_aug/generated_images.npy"))
    gen_lab = np.load(os.path.join(ROOT, "experiments/checkpoints/diffusion_aug/generated_labels.npy"))
    probsB = _proto_member("ddpm_proto",
        os.path.join(ROOT, "experiments/checkpoints/diffusion_experiment/ddpm_proto"),
        gen, gen_lab, test_images, y, device)

    # structural RF (predict_proba)
    bank_imgs, bank_y = _load_bank("both")
    Xtr = _featurize(bank_imgs, tag="bank")
    Xte = _featurize(test_images, tag="test")
    rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0))
    rf.fit(Xtr, bank_y)
    probsC = rf.predict_proba(Xte).astype(np.float32)
    print(f"[struct_rf] {(probsC.argmax(1)==y).mean()*100:.2f}%")

    out = os.path.join(ROOT, "experiments", "reports", "member_probs.npz")
    np.savez_compressed(out, morph=probsA, ddpm=probsB, struct=probsC, y=y)
    print(f"\n[members] saved {out}  shapes: morph{probsA.shape} ddpm{probsB.shape} struct{probsC.shape}")


if __name__ == "__main__":
    main()
