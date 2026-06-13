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
| 3 | **Skeleton CNN** | `src/skeleton` | Zhang-Suen / Lee / Guo-Hall skeletonized inputs |
| 4 | **Variants17** | `src/variants17` | One-shot 17-class style-aware learning from 17 templates |
| 5 | **Diffusion** | `src/diffusion` | Class-conditional DDPM generates training data |
| 6 | **Structural** | `src/structural` | Bag-of-features from a skeleton graph (no learning) |
| 7 | **Combined** | `src/ensemble` | Best-of-everything supervised system + exemplar selection (*not* one-shot) |
| 8 | **Introspection** | `scripts/probe_variant_recovery.py`, `run_subclass_expansion.py` | Extracting latent style-variant structure from CNNs |
| 9 | **Data Efficiency Benchmark (PROPOSED)** | `src/track9_benchmark` | **BestNet (SOTA CNN) vs. Heterogeneous Ensemble** on small training sets (100–10,000 labels) |

Tracks 1–6 are supervised or one-shot; **Tracks 7–8 deliberately step outside one-shot** —
Track 7 fuses the strongest pieces of the prior tracks on MNIST training data; Track 8
investigates whether a CNN's internal representation encodes unlabeled style variants and
whether that can improve training.
**Track 9 (proposed)** extends the data‑efficiency analysis of Track 7 by comparing a
modern SOTA CNN architecture against the best ensemble method, systematically varying the
amount of labeled training data.


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
| **Proto embedding (DDPM data, dim=32 + EMA)** | **79.08% ± 1.67** | 5 |
| Proto embedding (morphological aug) | 77.40% ± 2.24 | 4 |
| DDPM full-aug CNN | 75.33% ± 1.13 | 5 |
| DDPM no-aug CNN | 73.27% ± 1.36 | 5 |
| **Structural v3 (93-dim features + 8704-image bank, Random Forest)** | **72.32%** | 6 |
| Full-aug CNN (morphological) | 64.40% | 4 |
| No-aug CNN (morphological) | 49.87% | 4 |
| Nearest-template (L2) | 38.60% | 4 |
| Structural v1 (1-NN, 17 templates) | 35.81% | 6 |
| Random chance | 10.00% | — |

**Track 4 finding:** a prototype-embedding model reaches 78% of supervised accuracy from
just 17 hand-drawn templates. Augmentation is essential (+14pp over no-aug); metric
learning beats classification.

**Track 5 finding:** DDPM-generated training images are dramatically better raw data than
17 templates — a plain CNN jumps **+23.4pp** (49.87→73.27%). At full quality (dim=32,
timesteps=1000, **EMA** weight averaging, 512 imgs/class, 250 DDIM steps, GPU) the proto
config reaches **79.08% ± 1.67**, which **exceeds** the hand-crafted morphological baseline
(77.40% ± 2.24) by +1.69pp (4/5 seeds above it) — though overlapping error bars make it
matches-to-exceeds rather than a decisive win. The trajectory 73.7% (dim=16) → 76.6%
(dim=32) → 79.1% (dim=32+EMA) shows the gain tracks generation quality. Learned augmentation
now rivals expert-crafted augmentation with zero domain-specific design.

**Track 6 finding:** explicit structural features (endpoints, junctions, loops, typed
segments) reach 35.81% from 17 templates with 1-NN and zero learning. Three upgrades —
richer 93-dim descriptors (orientation histogram, curvature/inflection, geometry,
endpoint/loop position, and signed-curvature stroke shape), a large reference bank (features
from 8704 augmented+DDPM images), and a Random Forest classifier — take it to **72.32%**,
within ~5pp of the CNN baseline. The reference bank was the dominant lever (17 → 8704
reference vectors); signed-curvature descriptors added ~4pp by separating open single-stroke
digits (1/2/3/5/7). A fully interpretable, no-deep-learning method.

### Supervised — Track 7 (combined system + exemplar selection, *not* one-shot)

Track 7 fuses the strongest pieces of the prior tracks on the MNIST **training** set —
**CNN (raw)** + **Fusion CNN (raw + Guo-Hall "thin" skeleton)** + **structural RF (93-dim)** —
stacked by a meta-classifier, plus an **exemplar selector** that finds the examples that
"fit a class strongest". Shown in the low-label regime (at 60k a plain CNN is already ~99%):

| Budget n | Best single member | **Combined (stacked)** | gain |
|---------:|-------------------:|-----------------------:|-----:|
| 250 | 82.1% (RF) | **85.5%** | +3.4 |
| 500 | 85.5% (RF) | **91.1%** | +5.6 |
| 1000 | 87.4% (RF) | **92.8%** | +5.4 |
| 5000 | 95.5% (CNN) | **96.9%** | +1.4 |

**Track 7 findings:**
1. **Combining helps most when labels are scarce.** The structural RF is far more
   label-efficient than the CNN (75.9% vs 52.6% at n=100); the CNN overtakes by n≈2500; the
   stack exploits both and leads by +3–5.6pp at n=250–1000, narrowing as the CNN saturates.
2. **The "strongest-fitting" examples are the *worst* training coreset.** Selecting the most
   prototypical examples (highest confidence + closest to the class centroid) *underperforms
   random* at every budget (n=1000: 80.8% vs 92.8%) and plateaus early — they are easy,
   central, redundant, with no decision-boundary coverage. Their value is *representing* a
   class / seeding generation (the deferred diffusion step), not training. Hard/atypical
   examples are the opposite: catastrophic at tiny n (noise) but best at large n (97.4% @ 5k).

See `fig6_combined_efficiency.png`, `combined_track_results.json`. The earlier
semi-supervised stacking of the one-shot recognisers (86.6% from 50 labels, +17pp over raw
pixels; `ensemble_findings.md`, `fig5_label_efficiency.png`) is a precursor of this track.

### Introspection — Track 8 (extracting latent variants from CNNs)

Does a 10-class CNN encode style variants it was never trained on (crossed vs uncrossed 7)?

**Finding 1 (justified):** yes — a linear probe recovers the crossed-7 variant from the CNN
embedding at **83.8%** (base rate 57.3%, +26.5pp; beats the raw-pixel probe). The variant is
a faint linear sub-axis, *not* found by embedding-KMeans (ARI 0.02) but cleanly recovered in
the **structural-feature space** (ARI 0.83).

**Finding 2 (falsified, with cause):** splitting digits into structure-defined sub-classes,
expanding the head, and mapping back 17→10-style **hurts** low-label training (−12pp at n=100,
shrinking to −0.2pp at n=5000) — hard relabeling fragments scarce data. The principled fix —
an **auxiliary sub-class head** (multi-task, no fragmentation) — is the open direction. See
`cnn_introspection_findings.md`.

### Track 9 — Data Efficiency Benchmark

**Question:** in the low-data regime, can a hand-built ensemble of structural priors beat
the MNIST CNN *record holder*, and at what data budget? The opponent is **An et al. 2020,
"An Ensemble of Simple Convolutional Neural Network Models"**
([arXiv:2008.10400](https://arxiv.org/abs/2008.10400)) — a majority vote of three deep
"simple" CNNs (kernels 3/5/7, no pooling/padding, BN+ReLU, single FC), the reproducible
MNIST record at **99.87%** (heterogeneous) / 99.91% (two-level), trained with
rotation+translation augmentation. It is re-implemented faithfully but **lightly** (one
model per kernel, ~40 epochs) in `src/track9/record_models.py`; our light repro reaches
**99.65%** at full 60k (M3 99.53, M5 99.57, M7 99.51), below the published record as
expected.

**Three contestants**, trained on the *same* stratified budget (exactly n/10 per digit),
5 seeds, evaluated on the full 10k test set (`scripts/run_track9_benchmark.py`):
1. **Record holder** — An et al. M3/M5/M7 majority vote (light repro).
2. **Grand ensemble** — my best: leakage-free stack of skeleton-fusion CNN (Track 3) +
   structural bag-of-features RF (Track 6) + a diffusion-augmented CNN (Track 5, active
   only for n≥500 where a DDPM can be trained).
3. **Grand ensemble + augmentation** — contestant 2 with morphological augmentation.

**Result (MNIST test acc %, mean over 5 seeds):**

| n (labeled) | Record holder | Grand ensemble | Grand + aug |
|------------:|--------------:|---------------:|------------:|
| 20    | **64.2** | 44.1 | 46.7 |
| 50    | **83.4** | 58.4 | 61.3 |
| 100   | **90.2** | 80.9 | 81.1 |
| 200   | **96.2** | 86.6 | 89.6 |
| 500   | **98.2** | 92.6 | 95.5 |
| 1000  | **98.8** | 95.0 | 96.9 |
| 2000  | **99.1** | 96.7 | 98.0 |
| 5000  | **99.4** | 97.8 | 98.7 |
| 60000 | **99.65** | — | — |

**Track 9 finding (a clean negative):** the record holder wins at **every** budget,
including the extreme low-data regime where structural priors were expected to help most
(64% vs 47% at n=20). Simple rotation/translation augmentation on a deep CNN is more
data-efficient than explicit skeleton/bag-of-features structure. Two consolation positives:
augmentation reliably helps (`grand+aug > grand` at every n), and the gap is purely a
low-data phenomenon — by n≥1000 the grand ensemble is within ~2pp, converging toward (but
never overtaking) the record holder. Figure: `experiments/reports/figures/fig7_track9_efficiency.png`;
raw numbers in `experiments/reports/track9_results.json`.

> Note: the proposal's original "BestNet ≥99.585%" single CNN was a vanilla 2-conv net
> mislabelled as SOTA; an online search established the *actual* reproducible record
> (An et al.), which is what Track 9 benchmarks against.

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
python -m src.skeleton.train_skeleton_cnn --mnist-path mnist_data --out-dir experiments/checkpoints/skeleton --epochs 5
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

### Track 7 — Combined supervised system + exemplar selection
```bash
python scripts/run_combined_track.py          # data-efficiency curve + exemplar study + figure
python scripts/run_combined_track.py --smoke  # quick
# precursor (semi-supervised stacking of one-shot members):
python scripts/build_member_predictions.py && python scripts/run_ensemble_track.py
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
