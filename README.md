# DigitRecognizer Research Workspace

This repository is organized for multiple digit-recognition research tracks:

1. **Baseline** (`src/baseline`) - quick reference pipeline.
2. **Local CNN** (`src/local_cnn`) - your main local training setup.
3. **Novel Skeleton CNN** (`src/novel_skeleton`) - Zhang-Suen style skeletonized inputs.
4. **Variants14** (`src/variants14`) - 14-class setup where digits 0/1/4/7 are split into variant labels.

Your original NumPy MLP training file is preserved at:

- `legacy/main_numpy_mlp.py`

## Install

```bash
pip install -r requirements.txt
```

## Run

### 1) Baseline

```bash
python -m src.baseline.run_baseline --mnist-path mnist_data --out-dir experiments/checkpoints/baseline --epochs 3
```

### 2) Local CNN

```bash
python -m src.local_cnn.train_local_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/local_cnn --epochs 5
```

### 3) Skeleton CNN (Zhang-Suen)

```bash
python -m src.novel_skeleton.train_skeleton_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/skeleton --epochs 5
```

### 4) Variants14 pipeline

Generate 14-class labels from MNIST first:

```bash
python -m src.variants14.generate_variants14 --mnist-path mnist_data --out-dir data/processed/mnist14_variants
```

Then train:

```bash
python -m src.variants14.train_variants14_cnn --data-dir data/processed/mnist14_variants --out-dir experiments/checkpoints/variants14 --epochs 5
```

## 14-class schema

- `0_plain`, `0_crossed`
- `1_american`, `1_european`
- `2`, `3`
- `4_open`, `4_closed`
- `5`, `6`
- `7_crossed`, `7_uncrossed`
- `8`, `9`

## Legacy Note

The original pure NumPy MLP implementation is preserved in:

- `legacy/main_numpy_mlp.py`
