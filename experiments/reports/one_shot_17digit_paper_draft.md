# One-Shot Learning with a 17-Digit Style System: A Compact Experimental Study

**Draft v0.1**  
**Project:** DigitRecognizer  
**Date:** 2026-05-25

## Abstract

This paper draft presents a one-shot learning study using a custom 17-class digit style taxonomy derived from a single composite source image. Each class is represented by one template, centered into 28×28 format, and used as the sole base supervision signal for training. To improve robustness, we apply controlled low-magnitude transformations (rotation, anisotropic stretch, thickness perturbation, and additive noise) and evaluate both in-domain transformed recognition and out-of-domain transfer to MNIST. Results show a strong contrast: high in-domain transformed accuracy (up to 99.12%) and low MNIST transfer accuracy (best 9.62%), highlighting the central challenge of domain generalization in strict one-shot setups.

## 1. Introduction

One-shot learning seeks to generalize from extremely limited labeled examples. In handwritten digit recognition, standard benchmarks typically assume thousands of samples per class, while one-shot settings rely on prototype-level supervision. This work explores an intentionally strict regime: 17 style-aware classes learned from one template each.

The objective is twofold:
1. assess whether a CNN can learn stable discriminative signals from one-shot templates under mild synthetic variation,
2. test transfer from this synthetic/style-specific space to MNIST handwritten digits.

## 2. Problem Formulation

Let \(\mathcal{C}_{17}\) be a set of 17 digit-style classes:

`4_variant_a, 0_variant_a, 1_variant_a, 2_variant_a, 4_variant_b, 7_variant_a, 9_variant_a, 0_variant_b, 1_variant_b, 2_variant_b, 3, 4_variant_c, 5, 6, 7_variant_b, 8, 9_variant_b`.

The naming policy is intentionally normalized:
- `digit_variant_x` for digits with multiple stylistic conventions in the dataset,
- plain digit labels (e.g., `6`) where no variant split is used.

This reflects common handwriting differences across regions and educational traditions. For example, many writers use crossed vs uncrossed `0`, different stroke styles for `1` (serif/angled vs simple vertical forms), and crossed vs uncrossed `7`, often associated with American vs European writing habits. Similar alternative forms are represented for `2`, `4`, and `9`.

Given one template \(x_c\) per class \(c \in \mathcal{C}_{17}\), the training objective is to learn classifier \(f_\theta: \mathbb{R}^{28\times28} \to \mathcal{C}_{17}\). For MNIST evaluation, predictions are projected from 17-style classes to canonical digit classes \(\mathcal{D}_{10}=\{0,...,9\}\).

## 3. Data and Preprocessing

### 3.1 Template extraction
- Source: `data/processed/17digits_fixed_equal_height_thickness.png`
- Procedure:
  - isolate central strip,
  - split into 17 equal horizontal slots,
  - foreground-tight crop and center each digit to 28×28 canvas.
- Output:
  - `train_images.npy` with shape `(17, 28, 28)`
  - `train_labels17.npy` with labels `0..16`

### 3.2 Controlled transformed evaluation set

To probe robustness without large semantic drift, we generate exactly 20 transformed samples per class (340 total), each with:
- rotation: ±8°,
- stretch: x/y factors in [0.92, 1.08],
- thickness operation: none/dilate/erode,
- Gaussian noise: std in [0.01, 0.03].

All parameters are logged in:
`experiments/checkpoints/variants17/eval_transformed/transformations.json`.

## 4. Method

### 4.1 Model
- Backbone: `SimpleCNN` (2 conv blocks + MLP head)
- Output dimension: 17 classes

### 4.2 One-shot training strategy

Because one template per class is too sparse for stable optimization, we use low-noise repeated perturbation during training to create stochastic batches while preserving class identity.

### 4.3 Evaluation protocols
1. **MNIST transfer evaluation**:
   - test on MNIST t10k,
   - map 17-style predictions to digit-10 labels,
   - report top-1 accuracy.

2. **Transformed in-domain evaluation**:
   - evaluate 17-way classification on the generated transformed set,
   - report top-1 accuracy.

## 5. Results

From the latest run:
- **Best MNIST accuracy:** 9.62%
- **Best transformed 17-way accuracy:** 99.12%

Baseline references from project reports:
- One-shot variants17 MNIST transfer remains at 9.62% under the current strict setup.

### 5.1 Multi-seed stability (n=5)

Best-per-seed MNIST transfer (%): `9.62, 10.39, 12.98, 7.59, 9.22`  
Best-per-seed transformed17 (%): `99.12, 99.12, 97.06, 99.41, 97.65`

| Metric | Mean (%) | Std (%) |
|---|---:|---:|
| MNIST best accuracy (5 seeds) | 9.96 | 1.97 |
| Transformed17 best accuracy (5 seeds) | 98.47 | 1.05 |

### 5.2 Nearest-template one-shot baseline

| Method | MNIST (%) | Transformed17 (%) |
|---|---:|---:|
| Nearest-template (L2) | 11.94 | 97.06 |
| CNN one-shot (5-seed mean best) | 9.96 | 98.47 |
| Prototype-embedding baseline (single run, seed=42) | 11.37 | 97.35 |

Interpretation:
- On MNIST transfer, direct nearest-template matching is competitive and even higher than the current CNN mean.
- On transformed in-domain evaluation, CNN achieves stronger robustness than nearest-template matching.
- The prototype-embedding baseline acts as an intermediate learned-metric method and should be extended to multi-seed reporting in the next revision.

## 6. Discussion

The observed pattern is consistent with expected one-shot behavior under domain mismatch:

1. **In-domain success**: small, controlled distortions preserve the template manifold; classification remains highly separable.
2. **Cross-domain failure**: MNIST exhibits broader writer variation, stroke dynamics, and topology that are not captured by a single prototype per style class.
3. **Label granularity mismatch**: 17 style classes collapse into 10 MNIST identities, introducing ambiguity in projected evaluation.

Hence, this system currently behaves as a strong style-prototype recognizer rather than a general handwritten digit recognizer.

## 7. Threats to Validity

- **Single-source template bias**: all classes originate from one composite image.
- **Limited augmentation family**: mild affine+morphological perturbations may under-represent true handwriting variability.
- **Projection artifacts**: 17→10 mapping may hide class-specific errors.

## 8. Future Work

1. Add realistic elastic deformations and local nonlinear warps.
2. Move to metric-learning objectives (prototypical/triplet loss).
3. Use pretrained visual encoders and nearest-prototype decision rules.
4. Introduce few-shot support from real handwritten samples per style cluster.
5. Report confusion matrices (17-way and projected 10-way) and calibration metrics.

## 9. Reproducibility Notes

Key scripts:
- `src/variants17/generate_variants17.py`
- `src/variants17/train_variants17_cnn.py`

Main run command:

```bash
python src/variants17/train_variants17_cnn.py
```

Generated report artifacts:
- `experiments/reports/variants17_one_shot_analysis.md`
- `experiments/reports/variants17_one_shot_analysis.html`

---

### Draft Statement

This is a project draft intended as an internal technical paper scaffold. It can be expanded into formal conference format (e.g., IEEE/ACM) by adding full experimental tables, confidence intervals, ablations, and related work citations.
