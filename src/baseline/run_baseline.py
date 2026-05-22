import argparse
import os
import subprocess
import sys


def main(args):
    cmd = [
        sys.executable,
        "-m",
        "src.local_cnn.train_local_cnn",
        "--mnist-path",
        args.mnist_path,
        "--out-dir",
        args.out_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ]
    print("[baseline] Running baseline CNN pipeline...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path", default="mnist_data")
    parser.add_argument("--out-dir", default="experiments/checkpoints/baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
