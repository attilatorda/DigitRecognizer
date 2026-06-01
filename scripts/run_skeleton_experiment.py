"""
Full skeleton-vs-raw comparison experiment.

Trains SimpleCNN on raw MNIST and on each combination of
(skeletonization method) × (channel mode) over multiple seeds,
then writes a JSON result file and a markdown comparison table.

Usage (full run, ~20–40 min depending on hardware):
    python scripts/run_skeleton_experiment.py

Quick smoke-test:
    python scripts/run_skeleton_experiment.py --epochs 1 --seeds 1 --methods zhang
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
from src.novel_skeleton.skeleton_cache import load_or_build
from src.novel_skeleton.train_skeleton_cnn import (
    evaluate_detailed,
    to_loader as skeleton_to_loader,
)


# ---------------------------------------------------------------------------
# Raw-MNIST helpers (no skeletonization)
# ---------------------------------------------------------------------------

def raw_to_loader(images, labels, batch_size=128, shuffle=False):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Core training loop (shared by raw and skeleton configs)
# ---------------------------------------------------------------------------

def train_one_seed(
    train_loader: DataLoader,
    test_loader: DataLoader,
    seed: int,
    epochs: int,
    lr: float,
    in_channels: int,
    device: torch.device,
    checkpoint_path: str,
) -> tuple[float, dict[int, float], float]:
    """Train for one seed. Returns (best_test_acc, per_class_acc, train_time_s)."""
    set_seed(seed)
    model = SimpleCNN(num_classes=10, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc, _ = evaluate_detailed(model, test_loader, device)
        print(f"    epoch={epoch} test_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, checkpoint_path)

    _, per_class = evaluate_detailed(model, test_loader, device)
    return best_acc, per_class, train_time


# ---------------------------------------------------------------------------
# Run one configuration (method + channel_mode) over all seeds
# ---------------------------------------------------------------------------

def run_config(
    config_name: str,
    train_loader_fn,        # callable(seed) -> (train_loader, test_loader)
    seeds: list[int],
    epochs: int,
    lr: float,
    in_channels: int,
    device: torch.device,
    out_dir: str,
    skeletonize_time_s: float,
) -> dict:
    print(f"\n[experiment] === {config_name} ===", flush=True)
    ensure_dir(out_dir)

    best_accs, per_class_list, train_times = [], [], []

    for seed in seeds:
        print(f"  seed={seed}", flush=True)
        train_loader, test_loader = train_loader_fn(seed)
        ckpt = os.path.join(out_dir, f"best_{config_name}_seed{seed}.pt")
        acc, per_class, t = train_one_seed(
            train_loader, test_loader, seed, epochs, lr, in_channels, device, ckpt
        )
        best_accs.append(acc)
        per_class_list.append(per_class)
        train_times.append(t)
        print(f"  seed={seed} best_acc={acc:.4f} train_time={t:.1f}s", flush=True)

    mean_acc = float(np.mean(best_accs))
    std_acc = float(np.std(best_accs))

    # Mean per-class accuracy over seeds
    mean_per_class = {}
    for c in range(10):
        mean_per_class[c] = float(np.mean([pc[c] for pc in per_class_list]))

    print(
        f"[experiment] {config_name}  mean={mean_acc*100:.2f}%  std={std_acc*100:.2f}%",
        flush=True,
    )

    return {
        "name": config_name,
        "seeds": seeds,
        "best_test_accs": [float(a) for a in best_accs],
        "mean_test_acc": mean_acc,
        "std_test_acc": std_acc,
        "skeletonize_time_s": skeletonize_time_s,
        "mean_train_time_s": float(np.mean(train_times)),
        "per_class_accs_per_seed": [{int(k): v for k, v in pc.items()} for pc in per_class_list],
        "mean_per_class_acc": {int(k): v for k, v in mean_per_class.items()},
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_markdown(results: list[dict], path: str):
    lines = [
        "# Skeleton CNN Comparison",
        "",
        "SimpleCNN trained on raw MNIST pixels vs. skeletonized inputs.",
        f"Seeds per config: {results[0]['seeds']}",
        "",
        "## Overall accuracy",
        "",
        "| Config | Channel | Mean acc (%) | Std (%) | Skel. time (s) | Train time (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for r in results:
        name = r["name"]
        parts = name.split("_")
        # method is everything before the last channel token
        if name == "raw":
            method, channel = "raw", "raw"
        elif name.endswith("_hough"):
            method = "_".join(parts[:-2])
            channel = "skeleton+hough"
        else:
            method = "_".join(parts[:-1])
            channel = "skeleton"
        skel_t = f"{r['skeletonize_time_s']:.0f}" if r["skeletonize_time_s"] > 0 else "—"
        lines.append(
            f"| {method} | {channel} | {r['mean_test_acc']*100:.2f} "
            f"| {r['std_test_acc']*100:.2f} | {skel_t} | {r['mean_train_time_s']:.0f} |"
        )

    lines += [
        "",
        "## Per-class accuracy (mean over seeds, %)",
        "",
        "| Config | " + " | ".join(str(c) for c in range(10)) + " |",
        "|---|" + "|".join("---:" for _ in range(10)) + "|",
    ]
    for r in results:
        pc = r["mean_per_class_acc"]
        cells = " | ".join(f"{pc[c]*100:.1f}" for c in range(10))
        lines.append(f"| {r['name']} | {cells} |")

    lines += ["", f"*Generated by scripts/run_skeleton_experiment.py*", ""]

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[experiment] markdown saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seeds))
    methods = args.methods.split(",")
    channel_modes = args.channel_modes.split(",")

    print(f"[experiment] device={device} seeds={seeds} methods={methods} channels={channel_modes}")
    print(f"[experiment] epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}")

    train_raw, train_labels = load_mnist_idx(args.mnist_path, "train")
    test_raw, test_labels = load_mnist_idx(args.mnist_path, "t10k")

    ensure_dir(args.out_dir)
    ensure_dir(args.report_dir)
    results = []

    # --- raw baseline ---
    print("\n[experiment] === raw (baseline) ===", flush=True)

    def raw_loaders(_seed):
        return (
            raw_to_loader(train_raw, train_labels, args.batch_size, shuffle=True),
            raw_to_loader(test_raw, test_labels, args.batch_size, shuffle=False),
        )

    results.append(run_config(
        "raw", raw_loaders, seeds, args.epochs, args.lr,
        in_channels=1, device=device,
        out_dir=os.path.join(args.out_dir, "raw"),
        skeletonize_time_s=0.0,
    ))

    # --- skeleton configs ---
    for method in methods:
        # Skeletonize once (cached); time only counts if cache is cold
        train_skel, t_train = load_or_build(
            train_raw, method, "train", args.cache_dir, args.progress_every
        )
        test_skel, t_test = load_or_build(
            test_raw, method, "test", args.cache_dir, args.progress_every
        )
        skel_time = t_train + t_test

        for ch_mode in channel_modes:
            config_name = f"{method}_{'hough' if ch_mode == 'skeleton_hough' else 'skeleton'}"
            in_ch = 1 if ch_mode == "skeleton" else 2

            def make_loaders(_seed, _tr=train_skel, _te=test_skel, _ch=ch_mode):
                return (
                    skeleton_to_loader(_tr, train_labels, args.batch_size, shuffle=True,
                                       channel_mode=_ch,
                                       hough_threshold=args.hough_threshold,
                                       hough_line_length=args.hough_line_length,
                                       hough_line_gap=args.hough_line_gap),
                    skeleton_to_loader(_te, test_labels, args.batch_size, shuffle=False,
                                       channel_mode=_ch,
                                       hough_threshold=args.hough_threshold,
                                       hough_line_length=args.hough_line_length,
                                       hough_line_gap=args.hough_line_gap),
                )

            results.append(run_config(
                config_name, make_loaders, seeds, args.epochs, args.lr,
                in_channels=in_ch, device=device,
                out_dir=os.path.join(args.out_dir, config_name),
                skeletonize_time_s=skel_time,
            ))

    # --- save results ---
    json_path = os.path.join(args.report_dir, "skeleton_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[experiment] results saved: {json_path}")

    md_path = os.path.join(args.report_dir, "skeleton_comparison.md")
    write_markdown(results, md_path)

    # --- summary table to stdout ---
    print("\n=== SUMMARY ===")
    print(f"{'Config':<30} {'Mean acc':>10} {'Std':>8}")
    print("-" * 52)
    for r in results:
        print(f"{r['name']:<30} {r['mean_test_acc']*100:>9.2f}%  +-{r['std_test_acc']*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path", default="mnist_data")
    parser.add_argument("--cache-dir", default="data/processed/mnist_skeleton")
    parser.add_argument("--out-dir", default="experiments/checkpoints/skeleton_comparison")
    parser.add_argument("--report-dir", default="experiments/reports")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (uses seeds 0, 1, ..., N-1)")
    parser.add_argument("--methods", default="zhang,lee,thin,medial_axis",
                        help="Comma-separated skeleton methods to include")
    parser.add_argument("--channel-modes", default="skeleton,skeleton_hough",
                        help="Comma-separated channel modes: skeleton, skeleton_hough")
    parser.add_argument("--hough-threshold", type=int, default=8)
    parser.add_argument("--hough-line-length", type=int, default=5)
    parser.add_argument("--hough-line-gap", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=500)
    main(parser.parse_args())
