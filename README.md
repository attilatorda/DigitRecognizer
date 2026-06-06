# DigitRecognizer Research Workspace

A multi-track research repository exploring digit recognition from several angles —
supervised CNNs, topological/skeleton features, one-shot learning from a culturally-motivated
17-class template set, diffusion-based data generation, and explicit structural feature
extraction.

## Research tracks

| # | Track | Path | Idea |
|---|-------|------|------|
| 1 | **Baseline** | `src/baseline` | Quick reference supervised pipeline |
| 2 | **Local CNN** | `src/local_cnn` | Primary supervised CNN (the accuracy ceiling) |
| 3 | **Novel Skeleton CNN** | `src/novel_skeleton` | Zhang-Suen / Lee / Guo-Hall skeletonized inputs |
| 4 | **Variants17** | `src/variants17` | One-shot 17-class style-aware learning from 17 templates |
| 5 | **Diffusion** | `src/diffusion` | Class-conditional DDPM generates training data |
| 6 | **Structural** | `src/structural` | Bag-of-features from a skeleton graph (no learning) |

---

## Results summary

### Supervised (full MNIST training set)

| Method | MNIST test acc |
|--------|---------------:|
| Local CNN (supervised reference) | **99.11%** |
| Raw + skeleton fusion (Track 3) | 98.98% |
| Skeleton-only, Guo-Hall (Track 3) | 98.11% |
| Skeleton-only, Zhang-Suen (Track 3) | 97.78% |

**Track 3 finding:** skeletonization is *information loss* for MNIST — it discards
stroke-width gradients the CNN relies on, and the topology it preserves is already
learnable from raw pixels. Fusion nearly matches raw; skeleton-only always underperforms.

### One-shot / few-example (trained from 17 templates only)

| Method | MNIST test acc | Track |
|--------|---------------:|-------|
| **Proto embedding (DDPM data, dim=32 + EMA)** | **78.50% ± 1.76** | 5 |
| Proto embedding (morphological aug) | 77.46% ± 2.39 | 4 |
| DDPM full-aug CNN | 74.40% ± 1.99 | 5 |
| DDPM no-aug CNN | 73.94% ± 1.34 | 5 |
| **Structural v3 (93-dim features + 8704-image bank, Random Forest)** | **71.98%** | 6 |
| Full-aug CNN (morphological) | 65.19% | 4 |
| No-aug CNN (morphological) | 51.20% | 4 |
| Nearest-template (L2) | 38.60% | 4 |
| Structural v1 (1-NN, 17 templates) | 35.81% | 6 |
| Random chance | 10.00% | — |

**Track 4 finding:** a prototype-embedding model reaches 78% of supervised accuracy from
just 17 hand-drawn templates. Augmentation is essential (+14pp over no-aug); metric
learning beats classification.

**Track 5 finding:** DDPM-generated training images are dramatically better raw data than
17 templates — a plain CNN jumps **+22.7pp** (51.20→73.94%). At full quality (dim=32,
timesteps=1000, **EMA** weight averaging, 512 imgs/class, 250 DDIM steps, GPU) the proto
config reaches **78.50% ± 1.76**, which **exceeds** the hand-crafted morphological baseline
(77.46% ± 2.39) by +1.04pp (4/5 seeds above it) — though overlapping error bars make it
matches-to-exceeds rather than a decisive win. The trajectory 73.7% (dim=16) → 76.6%
(dim=32) → 78.5% (dim=32+EMA) shows the gain tracks generation quality. Learned augmentation
now rivals expert-crafted augmentation with zero domain-specific design.

**Track 6 finding:** explicit structural features (endpoints, junctions, loops, typed
segments) reach 35.81% from 17 templates with 1-NN and zero learning. Three upgrades —
richer 93-dim descriptors (orientation histogram, curvature/inflection, geometry,
endpoint/loop position, and signed-curvature stroke shape), a large reference bank (features
from 8704 augmented+DDPM images), and a Random Forest classifier — take it to **71.98%**,
within ~5.5pp of the CNN baseline. The reference bank was the dominant lever (17 → 8704
reference vectors); signed-curvature descriptors added ~4pp by separating open single-stroke
digits (1/2/3/5/7). A fully interpretable, no-deep-learning method.

---

## Install

```bash
pip install -r requirements.txt
```

## Run

### Track 1 — Baseline
```bash
python -m src.baseline.run_baseline --mnist-path mnist_data --out-dir experiments/checkpoints/baseline --epochs 3
```

### Track 2 — Local CNN
```bash
python -m src.local_cnn.train_local_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/local_cnn --epochs 5
```

### Track 3 — Skeleton CNN
```bash
python -m src.novel_skeleton.train_skeleton_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/skeleton --epochs 5
```

### Track 4 — Variants17 one-shot (full comparison + ablation)
```bash
python scripts/run_oneshot_experiment.py --ablation --proto-seeds 5
```

### Track 5 — Diffusion augmentation
```bash
# Phase 1: learn digit structure from MNIST (single-variant classes 3,5,6,8)
python scripts/train_diffusion.py --phase 1 --epochs 50 --dim 16 --timesteps 250

# Phase 2: fine-tune on all 17 CultiVar classes
python scripts/train_diffusion.py --phase 2 \
    --resume experiments/checkpoints/diffusion/phase1_final.pt \
    --epochs 100 --dim 16 --timesteps 250

# Generate training images, then evaluate
python scripts/generate_diffusion_aug.py --checkpoint experiments/checkpoints/diffusion/phase2_final.pt --dim 16 --timesteps 250
python scripts/run_diffusion_experiment.py
```

### Track 6 — Structural bag-of-features
```bash
python scripts/run_structural_experiment.py            # full 10K test (~10s)
python scripts/run_structural_experiment.py --smoke    # 100-image quick check
```

### Interactive browser
```bash
launch_browser.bat        # or: python -m streamlit run scripts/mnist_browser_app.py
```
Browse MNIST/`.npy` datasets, tag images, detect duplicates, view a paginated gallery.

---

## Reports & analysis

- [Reports Index (HTML)](experiments/reports/index.html)
- **CultiVar-17 paper (canonical):** `experiments/reports/cultivar17_paper.pdf` (IEEEtran, 5pp) — source `cultivar17_paper.tex`; [short summary](experiments/reports/one_shot_17digit_paper_draft.md)
- [Diffusion track status & results](experiments/reports/diffusion_track_status.md)
- [Structural early findings](experiments/reports/structural_early_findings.md)
- [Skeleton CNN vs Local CNN analysis (HTML)](experiments/reports/skeleton_vs_local_analysis.html)

---

## CultiVar-17 — the 17-class taxonomy (Tracks 4–6)

Labels are normalized as:
- `digit_variant_x` when multiple writing styles exist for the same digit,
- plain digit names (e.g. `6`) when no variant split is used.

Full label set:

`4_variant_a, 0_variant_a, 1_variant_a, 2_variant_a, 4_variant_b, 7_variant_a, 9_variant_a, 0_variant_b, 1_variant_b, 2_variant_b, 3, 4_variant_c, 5, 6, 7_variant_b, 8, 9_variant_b`

### Why these variants exist

The split captures culturally common handwriting differences (e.g. American vs European):
- `0`: crossed vs uncrossed
- `1`: serifed/angled top vs simple vertical
- `7`: crossed vs uncrossed
- alternate handwritten styles for `2`, `4`, `9`

For MNIST evaluation the 17 style classes collapse to 10 canonical digits via a fixed,
**surjective** map (`src/variants17/label_schema.py`) — each variant maps to its base digit
(e.g. `0_variant_a`, `0_variant_b` → 0).

### Dataset availability

The CultiVar-17 dataset ships **in-repo** (it is small — ~27 KB total):
- `data/processed/mnist17_variants/train_images.npy` — 17 × 28 × 28 uint8 templates
- `data/processed/mnist17_variants/train_labels17.npy` — labels 0–16
- `data/processed/17digits_fixed_equal_height_thickness.png` — the source composite

The templates are regenerable from the source image:

```bash
python -m src.variants17.generate_variants17
```

(All other `data/` artifacts — `mnist_data/`, augmented/generated `.npy` sets — are large
and gitignored; they are reproduced on demand by the run commands above.)

### A note on epochs

`experiments/configs/*.yaml` (`epochs: 3`/`5`) configure the **supervised** Tracks 1–3.
The one-shot paper experiment (Track 4) is driven by `scripts/run_oneshot_experiment.py`,
which trains for **8 epochs** (its `--epochs` default) — this is the value the paper reports.
