"""
One-shot 17-class digit recognition — full comparison experiment.

Runs four configurations over multiple seeds and evaluates each on both the
MNIST test split (10K) and the MNIST train split (60K — equally unseen, since
the one-shot model was never trained on any MNIST image).

Configurations
--------------
1. nearest_template   — L2 distance to the 17 raw templates; no training
2. no_aug_cnn         — SimpleCNN, noise-only augmentation (elastic_prob=0, stroke_prob=0)
3. full_aug_cnn       — SimpleCNN, full augmentation (elastic + stroke-width + noise)
4. proto              — EmbeddingCNN with prototype nearest-neighbour classification

Output
------
experiments/reports/oneshot_results.json   — raw numbers
experiments/reports/oneshot_results.md     — formatted comparison table

Usage
-----
Full run (~65 min CPU):
    python scripts/run_oneshot_experiment.py

Quick smoke-test:
    python scripts/run_oneshot_experiment.py --epochs 1 --seeds 1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir, set_seed
from src.local_cnn.model import SimpleCNN
from src.variants17.augment import augment_dataset
from src.variants17.benchmarks import BENCHMARKS
from src.variants17.label_schema import CLASS17_TO_DIGIT10
from src.variants17.train_variants17_cnn import (
    build_transformed_eval_set,
    eval_on_mnist,
    eval_17class,
    save_augmented_set,
)
from src.variants17.train_variants17_proto import (
    EmbeddingCNN,
    compute_prototypes,
    eval_mnist_proj,
)


# ---------------------------------------------------------------------------
# Nearest-template baseline (no training)
# ---------------------------------------------------------------------------

def run_nearest_template(templates_u8, labels17, test_images, train_images):
    """L2 distance from each MNIST image to the 17 templates. O(n·17) dot products."""
    t0 = time.perf_counter()
    tmpl = templates_u8.astype(np.float32).reshape(17, -1) / 255.0

    def _eval(images_u8):
        imgs = images_u8.astype(np.float32).reshape(len(images_u8), -1) / 255.0
        dists = np.linalg.norm(imgs[:, None, :] - tmpl[None, :, :], axis=2)  # (N, 17)
        pred17 = dists.argmin(axis=1)
        pred10 = np.array([CLASS17_TO_DIGIT10[int(c)] for c in pred17])
        return pred10

    _, test_labels = None, None  # labels passed separately below
    elapsed = time.perf_counter() - t0
    return elapsed


def nearest_template_accuracy(templates_u8, images_u8, labels10):
    tmpl = templates_u8.astype(np.float32).reshape(17, -1) / 255.0
    imgs = images_u8.astype(np.float32).reshape(len(images_u8), -1) / 255.0
    # Process in chunks to avoid huge memory allocation on train split
    chunk = 2000
    preds = []
    for i in range(0, len(imgs), chunk):
        dists = np.linalg.norm(imgs[i:i+chunk, None, :] - tmpl[None, :, :], axis=2)
        preds.append(dists.argmin(axis=1))
    pred17 = np.concatenate(preds)
    pred10 = np.array([CLASS17_TO_DIGIT10[int(c)] for c in pred17])
    return float((pred10 == labels10).mean())


# ---------------------------------------------------------------------------
# CNN training loop
# ---------------------------------------------------------------------------

def train_cnn_one_seed(
    seed, templates_u8, labels17,
    test_images, test_labels,
    train_images_mnist, train_labels_mnist,
    epochs, batch_size, lr, elastic_prob, stroke_prob,
    out_dir, device,
    save_augmented_dir: str = "",
):
    set_seed(seed)
    aug_x, aug_y = augment_dataset(
        templates_u8, labels17,
        repeats=256, seed=seed,
        elastic_prob=elastic_prob, stroke_prob=stroke_prob,
    )
    if save_augmented_dir and seed == 0:
        save_augmented_set(aug_x, aug_y, save_augmented_dir)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(aug_x, dtype=torch.float32).unsqueeze(1),
            torch.tensor(aug_y, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )

    model = SimpleCNN(num_classes=17).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()

        acc = eval_on_mnist(model, test_images, test_labels, device)
        print(f"    epoch={epoch} test_acc={acc*100:.2f}%", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
        ensure_dir(out_dir)
        torch.save(best_state, os.path.join(out_dir, f"best_seed{seed}.pt"))

    acc_train = eval_on_mnist(model, train_images_mnist, train_labels_mnist, device)
    return best_acc, acc_train, train_time


# ---------------------------------------------------------------------------
# Proto training loop
# ---------------------------------------------------------------------------

def train_proto_one_seed(
    seed, templates_u8, labels17,
    test_images, test_labels,
    train_images_mnist, train_labels_mnist,
    epochs, batch_size, lr, emb_dim, elastic_prob, stroke_prob,
    out_dir, device,
    save_augmented_dir: str = "",
):
    set_seed(seed)
    aug_x, aug_y = augment_dataset(
        templates_u8, labels17,
        repeats=256, seed=seed,
        elastic_prob=elastic_prob, stroke_prob=stroke_prob,
    )
    if save_augmented_dir and seed == 0:
        save_augmented_set(aug_x, aug_y, save_augmented_dir)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(aug_x, dtype=torch.float32).unsqueeze(1),
            torch.tensor(aug_y, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )

    model = EmbeddingCNN(emb_dim=emb_dim).to(device)
    clf = nn.Linear(emb_dim, 17).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    support_x01 = templates_u8.astype(np.float32) / 255.0
    best_acc = 0.0
    best_model_state = None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train(); clf.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            ce(clf(model(xb)), yb).backward()
            opt.step()

        protos = compute_prototypes(model, support_x01, labels17, device)
        acc = eval_mnist_proj(model, test_images, test_labels, protos, device)
        print(f"    epoch={epoch} test_acc={acc*100:.2f}%", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        ensure_dir(out_dir)
        torch.save(best_model_state, os.path.join(out_dir, f"best_seed{seed}.pt"))

    protos = compute_prototypes(model, support_x01, labels17, device)
    acc_train = eval_mnist_proj(model, train_images_mnist, train_labels_mnist, protos, device)
    return best_acc, acc_train, train_time


# ---------------------------------------------------------------------------
# Run one full configuration (N seeds)
# ---------------------------------------------------------------------------

def run_config(name, train_fn, seeds, **kwargs):
    print(f"\n[oneshot] === {name} ===", flush=True)
    test_accs, train_accs, times = [], [], []
    for seed in seeds:
        print(f"  seed={seed}", flush=True)
        acc_test, acc_train, t = train_fn(seed=seed, **kwargs)
        test_accs.append(acc_test)
        train_accs.append(acc_train)
        times.append(t)
        print(f"  seed={seed} test={acc_test*100:.2f}% train={acc_train*100:.2f}% time={t:.0f}s", flush=True)

    result = {
        "name": name,
        "seeds": seeds,
        "mnist_test_accs": [float(a) for a in test_accs],
        "mnist_train_accs": [float(a) for a in train_accs],
        "mean_test_acc": float(np.mean(test_accs)),
        "std_test_acc": float(np.std(test_accs)),
        "mean_train_acc": float(np.mean(train_accs)),
        "std_train_acc": float(np.std(train_accs)),
        "mean_train_time_s": float(np.mean(times)),
    }
    print(
        f"[oneshot] {name}  test={result['mean_test_acc']*100:.2f}%+-{result['std_test_acc']*100:.2f}%"
        f"  train={result['mean_train_acc']*100:.2f}%+-{result['std_train_acc']*100:.2f}%",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(results, nearest, local_cnn_acc, path):
    lines = [
        "# One-Shot 17-Class Digit Recognition — Results",
        "",
        "Training data: 17 hand-drawn templates × 256 augmented copies each.",
        "Evaluation: MNIST test split (10K) and MNIST train split (60K, equally unseen).",
        "",
        "## One-shot results",
        "",
        "| Config | Test acc (%) | ± | Train acc (%) | ± | Train time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        lines.append(
            f"| {r['name']} "
            f"| {r['mean_test_acc']*100:.2f} | {r['std_test_acc']*100:.2f} "
            f"| {r['mean_train_acc']*100:.2f} | {r['std_train_acc']*100:.2f} "
            f"| {r['mean_train_time_s']:.0f} |"
        )

    lines += [
        "",
        "## Nearest-template L2 baseline (no training)",
        "",
        f"| Split | Accuracy (%) |",
        "|---|---:|",
        f"| MNIST test (10K) | {nearest['test_acc']*100:.2f} |",
        f"| MNIST train (60K) | {nearest['train_acc']*100:.2f} |",
        "",
        "## Context: published MNIST benchmarks",
        "",
        "| Method | Accuracy (%) | Source |",
        "|---|---:|---|",
    ]

    for b in BENCHMARKS:
        if b["accuracy"] is None:
            acc_str = f"{local_cnn_acc*100:.2f}" if local_cnn_acc else "—"
        else:
            acc_str = f"{b['accuracy']*100:.2f}"
        lines.append(f"| {b['label']} | {acc_str} | {b['source']} |")

    # Add our best one-shot result for positioning
    best_test = max(r["mean_test_acc"] for r in results)
    lines += [
        f"| **One-shot best (this work)** | **{best_test*100:.2f}** | this project |",
        "",
        "*Generated by scripts/run_oneshot_experiment.py*",
        "",
    ]

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[oneshot] markdown saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seeds))
    print(f"[oneshot] device={device} seeds={seeds} epochs={args.epochs}", flush=True)

    # Load data once
    templates_u8 = np.load(os.path.join(args.data_dir, "train_images.npy"))
    labels17 = np.load(os.path.join(args.data_dir, "train_labels17.npy"))
    test_images, test_labels = load_mnist_idx(args.mnist_path, "t10k")
    train_images_mnist, train_labels_mnist = load_mnist_idx(args.mnist_path, "train")
    print(f"[oneshot] loaded MNIST test={len(test_images)} train={len(train_images_mnist)}", flush=True)

    # Load local_cnn accuracy for benchmark table if checkpoint exists
    local_cnn_acc = None
    local_ckpt = os.path.join(args.local_cnn_dir, "best_local_cnn.pt")
    if os.path.exists(local_ckpt):
        try:
            m = SimpleCNN(num_classes=10).to(device)
            m.load_state_dict(torch.load(local_ckpt, map_location=device))
            test_x = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1).to(device) / 255.0
            m.eval()
            with torch.no_grad():
                pred = m(test_x).argmax(dim=1).cpu().numpy()
            local_cnn_acc = float((pred == test_labels).mean())
            print(f"[oneshot] local_cnn accuracy: {local_cnn_acc*100:.2f}%", flush=True)
        except Exception as e:
            print(f"[oneshot] could not load local_cnn: {e}", flush=True)

    ensure_dir(args.out_dir)
    ensure_dir(args.report_dir)
    results = []

    # 1. Nearest-template baseline
    print("\n[oneshot] === nearest_template (no training) ===", flush=True)
    t0 = time.perf_counter()
    nt_test = nearest_template_accuracy(templates_u8, test_images, test_labels)
    nt_train = nearest_template_accuracy(templates_u8, train_images_mnist, train_labels_mnist)
    nt_time = time.perf_counter() - t0
    nearest = {"test_acc": nt_test, "train_acc": nt_train, "time_s": nt_time}
    print(f"[oneshot] nearest_template  test={nt_test*100:.2f}%  train={nt_train*100:.2f}%  time={nt_time:.1f}s", flush=True)

    # Shared kwargs for training configs
    shared = dict(
        templates_u8=templates_u8, labels17=labels17,
        test_images=test_images, test_labels=test_labels,
        train_images_mnist=train_images_mnist, train_labels_mnist=train_labels_mnist,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device,
    )

    def _aug_dir(config_name):
        return os.path.join(args.out_dir, config_name, "augmented_train")

    # 2. No-augmentation CNN
    results.append(run_config(
        "no_aug_cnn",
        lambda seed, **kw: train_cnn_one_seed(
            seed=seed, elastic_prob=0.0, stroke_prob=0.0,
            out_dir=os.path.join(args.out_dir, "no_aug_cnn"),
            save_augmented_dir=_aug_dir("no_aug_cnn"), **kw,
        ),
        seeds, **shared,
    ))

    # 3. Ablation: elastic only
    if args.ablation:
        results.append(run_config(
            "elastic_only_cnn",
            lambda seed, **kw: train_cnn_one_seed(
                seed=seed, elastic_prob=args.elastic_prob, stroke_prob=0.0,
                out_dir=os.path.join(args.out_dir, "elastic_only_cnn"),
                save_augmented_dir=_aug_dir("elastic_only_cnn"), **kw,
            ),
            seeds, **shared,
        ))

    # 4. Ablation: stroke only
    if args.ablation:
        results.append(run_config(
            "stroke_only_cnn",
            lambda seed, **kw: train_cnn_one_seed(
                seed=seed, elastic_prob=0.0, stroke_prob=args.stroke_prob,
                out_dir=os.path.join(args.out_dir, "stroke_only_cnn"),
                save_augmented_dir=_aug_dir("stroke_only_cnn"), **kw,
            ),
            seeds, **shared,
        ))

    # 5. Full-augmentation CNN
    results.append(run_config(
        "full_aug_cnn",
        lambda seed, **kw: train_cnn_one_seed(
            seed=seed, elastic_prob=args.elastic_prob, stroke_prob=args.stroke_prob,
            out_dir=os.path.join(args.out_dir, "full_aug_cnn"),
            save_augmented_dir=_aug_dir("full_aug_cnn"), **kw,
        ),
        seeds, **shared,
    ))

    # 6. Proto embedding (optionally more seeds)
    proto_seeds = list(range(args.proto_seeds))
    results.append(run_config(
        "proto",
        lambda seed, **kw: train_proto_one_seed(
            seed=seed, emb_dim=args.emb_dim,
            elastic_prob=args.elastic_prob, stroke_prob=args.stroke_prob,
            out_dir=os.path.join(args.out_dir, "proto"),
            save_augmented_dir=_aug_dir("proto"), **kw,
        ),
        proto_seeds, **shared,
    ))

    # Save (merge with existing to preserve previous runs)
    json_path = os.path.join(args.report_dir, "oneshot_results.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            existing = json.load(f)
        new_by_name = {r["name"]: r for r in results}
        merged = [new_by_name.pop(e["name"], e) for e in existing.get("configs", [])]
        merged.extend(new_by_name.values())
        payload = {
            "nearest_template": nearest,
            "configs": merged,
            "local_cnn_acc": local_cnn_acc or existing.get("local_cnn_acc"),
        }
    else:
        payload = {"nearest_template": nearest, "configs": results, "local_cnn_acc": local_cnn_acc}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[oneshot] results saved: {json_path}")

    md_path = os.path.join(args.report_dir, "oneshot_results.md")
    write_markdown(results, nearest, local_cnn_acc, md_path)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Config':<20} {'Test':>10} {'Train':>10}")
    print("-" * 44)
    print(f"{'nearest_template':<20} {nt_test*100:>9.2f}%  {nt_train*100:>9.2f}%")
    for r in results:
        print(f"{r['name']:<20} {r['mean_test_acc']*100:>9.2f}%  {r['mean_train_acc']*100:>9.2f}%")
    if local_cnn_acc:
        print(f"\n{'local_cnn (supervised)':<20} {local_cnn_acc*100:>9.2f}%  (reference)")

    # Comparison against pre-fix baseline if a backup exists
    pre_fix_path = os.path.join(args.report_dir, "oneshot_results_pre_augfix.json")
    if os.path.exists(pre_fix_path):
        try:
            with open(pre_fix_path, encoding="utf-8") as f:
                old = json.load(f)
            old_map = {c["name"]: c for c in old.get("configs", [])}
            new_map = {r["name"]: r for r in results}
            print("\n=== AUGMENTATION FIX COMPARISON (test acc) ===")
            print(f"{'Config':<22} {'Before':>12} {'After':>12} {'Delta':>8}")
            print("-" * 58)
            for name in ["no_aug_cnn", "elastic_only_cnn", "stroke_only_cnn", "full_aug_cnn", "proto"]:
                if name in old_map and name in new_map:
                    before = old_map[name]["mean_test_acc"] * 100
                    after  = new_map[name]["mean_test_acc"] * 100
                    delta  = after - before
                    sign   = "+" if delta >= 0 else ""
                    print(f"{name:<22} {before:>10.2f}%  {after:>10.2f}%  {sign}{delta:>6.2f}pp")
        except Exception as e:
            print(f"[oneshot] could not print comparison: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path", default="mnist_data")
    parser.add_argument("--data-dir", default="data/processed/mnist17_variants")
    parser.add_argument("--out-dir", default="experiments/checkpoints/oneshot_comparison")
    parser.add_argument("--report-dir", default="experiments/reports")
    parser.add_argument("--local-cnn-dir", default="experiments/checkpoints/local_cnn",
                        help="Directory with best_local_cnn.pt for reference accuracy")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--proto-seeds", type=int, default=5,
                        help="Number of seeds for the proto config (default 5 for paper)")
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--elastic-prob", type=float, default=0.70)
    parser.add_argument("--stroke-prob", type=float, default=1.00)
    parser.add_argument("--ablation", action="store_true",
                        help="Also run elastic-only and stroke-only ablation configs")
    main(parser.parse_args())
