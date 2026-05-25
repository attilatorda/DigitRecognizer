# DigitRecognizer Research Workspace

This repository is organized for multiple digit-recognition research tracks:

1. **Baseline** (`src/baseline`) - quick reference pipeline.
2. **Local CNN** (`src/local_cnn`) - primary local training setup.
3. **Novel Skeleton CNN** (`src/novel_skeleton`) - Zhang-Suen style skeletonized inputs.
4. **Variants14** (`src/variants14`) - 14-class setup where digits 0/1/4/7 are split into variant labels.
5. **Variants17** (`src/variants17`) - one-shot 17-class style-aware setup from custom templates.

## Results & Analysis

- [Reports Index (HTML)](experiments/reports/index.html)
- [Skeleton CNN vs Local CNN - Analysis (HTML)](experiments/reports/skeleton_vs_local_analysis.html)
- [Project Changes & Reproduction Tutorial (HTML)](experiments/reports/project_changes_and_repro_tutorial.html)

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

## 17-class naming convention (one-shot variants17)

For the one-shot style-aware track, labels are normalized as:
- `digit_variant_x` when multiple writing styles exist for the same digit in the curated template set,
- plain digit names (e.g. `6`) when no variant split is used.

Current variants17 labels:

`4_variant_a, 0_variant_a, 1_variant_a, 2_variant_a, 4_variant_b, 7_variant_a, 9_variant_a, 0_variant_b, 1_variant_b, 2_variant_b, 3, 4_variant_c, 5, 6, 7_variant_b, 8, 9_variant_b`

### Why these variants exist

The variant split captures culturally common handwriting differences (for example American vs European conventions):
- `0`: crossed vs uncrossed forms
- `1`: serifed/angled top vs simpler vertical forms
- `7`: crossed vs uncrossed forms
- additional alternate handwritten styles for `2`, `4`, and `9`

This makes the label system explicit and reproducible in code, generated files, and reports.
