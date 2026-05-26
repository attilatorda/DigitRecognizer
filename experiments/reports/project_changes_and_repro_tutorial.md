# DigitRecognizer — Changes Summary & Reproduction Tutorial

Date: 2026-05-22

## 1) What changed in this update

This project was restructured from a single training script into a multi-track research workspace.

### New structure

- `legacy/main_numpy_mlp.py`
  - Preserved original NumPy MLP script (moved from previous `main.py`).

- `src/common/`
  - Shared helpers:
    - `data_io.py` (MNIST IDX loading)
    - `utils.py` (seed setup, directory creation)
    - `metrics.py`
    - `labels.py`

- `src/local_cnn/`
  - `model.py` (SimpleCNN)
  - `train_local_cnn.py` (standard grayscale CNN training on MNIST)

- `src/baseline/`
  - `run_baseline.py` (baseline runner, aligned with local CNN pipeline)

- `src/novel_skeleton/`
  - `skeletonize.py` (Zhang–Suen style thinning via `skimage.morphology.skeletonize`)
  - `train_skeleton_cnn.py` (CNN training on skeletonized images)

- `experiments/configs/`
  - `baseline.yaml`, `local_cnn.yaml`, `skeleton_cnn.yaml`, `variants17.yaml`

- `experiments/logs/`
  - Run logs from executed experiments.

- `experiments/reports/`
  - `skeleton_vs_local_analysis.md`
  - this tutorial document.

### Dependency file

- `requirements.txt` includes:
  - `numpy`, `torch`, `torchvision`, `scikit-image`, `tqdm`, `matplotlib`, `scikit-learn`, `pyyaml`

---

## 2) Reproducing the results (step-by-step)

## Prerequisites

- Python 3.10+ (tested in your environment with Python 3.14)
- Terminal in repository root:

```bash
d:/Developer/DigitRecognizer
```

MNIST IDX files already present in:

```text
mnist_data/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run local grayscale CNN baseline

```bash
python -m src.local_cnn.train_local_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/local_cnn --epochs 5 --batch-size 128 --lr 0.001
```

Expected artifact:

- `experiments/checkpoints/local_cnn/best_local_cnn.pt`

### Step 3 — Run novel skeleton CNN experiment

```bash
python -m src.novel_skeleton.train_skeleton_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/skeleton --epochs 5 --batch-size 128 --lr 0.001
```

Expected artifact:

- `experiments/checkpoints/skeleton/best_skeleton_cnn.pt`

### Step 4 — (Optional) Save logs while running

PowerShell example:

```powershell
python -m src.novel_skeleton.train_skeleton_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/skeleton --epochs 5 --batch-size 128 --lr 0.001 2>&1 | Tee-Object -FilePath experiments/logs/skeleton_run.log
python -m src.local_cnn.train_local_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/local_cnn --epochs 5 --batch-size 128 --lr 0.001 2>&1 | Tee-Object -FilePath experiments/logs/local_run.log
```

### Step 5 — Compare results

Read:

- `experiments/logs/skeleton_run.log`
- `experiments/logs/local_run.log`
- `experiments/reports/skeleton_vs_local_analysis.md`

Observed in this run:

- Skeleton best test accuracy: **0.9778**
- Local baseline best test accuracy: **0.9911**
- Gap: **1.33 percentage points**

---

## 3) Reproducing the 17-class one-shot workflow

### Generate 17-class template dataset

```bash
python -m src.variants17.generate_variants17 --image-path data/processed/17digits_fixed_equal_height_thickness.png --out-dir data/processed/mnist17_variants
```

### Train 17-class one-shot CNN

```bash
python -m src.variants17.train_variants17_cnn --data-dir data/processed/mnist17_variants --out-dir experiments/checkpoints/variants17 --epochs 8 --batch-size 128 --lr 0.001
```

---

## 4) Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`**
  - Run `pip install -r requirements.txt` in the same Python environment used by VS Code.

- **Pylance unresolved import warnings**
  - Ensure VS Code is using the interpreter where dependencies were installed.

- **PowerShell heredoc (`<<`) fails**
  - Use `python -c "..."` or run a `.py` script file; bash heredoc syntax is not valid in default PowerShell/cmd flow.

---

## 5) Suggested next improvements

1. Use 2-channel input: original grayscale + skeleton.
2. Add morphological denoise before skeletonization.
3. Add confusion matrices for class-level error analysis.
4. Extend training to 15–25 epochs with LR scheduler.
