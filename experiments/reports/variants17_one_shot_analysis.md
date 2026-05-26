# Variants17 One-Shot Learning — Result Analysis

Date: 2026-05-25  
Author: Cline

## 1) Objective

Evaluate a strict one-shot setup where the model is trained from a custom **17-template** digit strip (one template per class) and then tested on MNIST.

New extension in this run:
- controlled transformed evaluation set (20 samples per class)
- transformation logging for reproducibility

## 2) Setup summary

### Training source (one-shot base)
- `data/processed/17digits_fixed_equal_height_thickness.png`
- split into 17 equal slots and centered into 28x28 templates
- classes:
  `4_variant_a, 0_variant_a, 1_variant_a, 2_variant_a, 4_variant_b, 7_variant_a, 9_variant_a, 0_variant_b, 1_variant_b, 2_variant_b, 3, 4_variant_c, 5, 6, 7_variant_b, 8, 9_variant_b`

Variant naming rationale:
- `digit_variant_x` is used when culturally common handwriting forms differ (for example: crossed vs uncrossed 0, serifed/angled 1 forms, and crossed vs uncrossed 7 often associated with different regional writing habits).
- Plain digit names (`3`, `5`, `6`, `8`) are used where this dataset did not separate meaningful style variants.

### Pipeline files
- `src/variants17/generate_variants17.py`
- `src/variants17/train_variants17_cnn.py`
- `src/variants17/label_schema.py`

### Controlled transformed eval set
- exactly **20** transformed samples per class
- total transformed samples: **340**
- per-sample operations (mild):
  - rotation: ±8°
  - stretch: x/y scale in [0.92, 1.08]
  - thickness: none / dilate / erode
  - noise: Gaussian std in [0.01, 0.03]

Artifacts:
- `experiments/checkpoints/variants17/eval_transformed/images.npy`
- `experiments/checkpoints/variants17/eval_transformed/labels17.npy`
- `experiments/checkpoints/variants17/eval_transformed/transformations.json`

## 3) Results from latest run

Command:

```bash
python src/variants17/train_variants17_cnn.py
```

Observed:
- **Best MNIST test accuracy**: **9.62%**
- **Transformed 17-class eval accuracy**: up to **99.12%**

Interpretation:
- The model learns the synthetic/style-preserving transformed set very well.
- Transfer to MNIST remains low, indicating strong domain shift between custom templates and handwritten MNIST distribution.

## 4) One-shot performance summary

Core one-shot outcomes:
- One-shot variants17 MNIST best: **9.62%**
- Transformed 17-class eval: up to **99.12%**

This highlights a strong in-domain robustness profile with weak cross-domain transfer to MNIST.

### 5-seed reproducibility (Day-1 publishability run)

Best-per-seed MNIST accuracies (%): `9.62, 10.39, 12.98, 7.59, 9.22`  
Best-per-seed transformed17 accuracies (%): `99.12, 99.12, 97.06, 99.41, 97.65`

| Metric | Mean (%) | Std (%) |
|---|---:|---:|
| MNIST best accuracy (5 seeds) | 9.96 | 1.97 |
| Transformed17 best accuracy (5 seeds) | 98.47 | 1.05 |

### One-shot baselines (nearest-template and learned metric)

| Method | MNIST (%) | Transformed17 (%) |
|---|---:|---:|
| Nearest-template (L2 to 17 templates) | 11.94 | 97.06 |
| Prototype-embedding baseline (5-seed mean ± std) | 9.35 ± 2.06 | 97.00 ± 0.49 |
| CNN one-shot (5-seed mean ± std of best) | 9.96 ± 1.97 | 98.47 ± 1.05 |

Prototype-embedding per-seed finals (MNIST %): `11.37, 8.53, 10.05, 10.62, 6.18`  
Prototype-embedding per-seed finals (transformed17 %): `97.35, 97.65, 96.76, 96.47, 96.76`

Baseline takeaway:
- Template matching is competitive on MNIST transfer in this strict setup.
- CNN improves in-domain transformed robustness over nearest-template matching.
- Prototype-embedding baseline provides a stronger learned-metric comparator and should be retained as a core comparison track.

## 5) Why this happens (technical analysis)

1. **Extremely low base diversity**: one prototype per class cannot span handwritten variability.
2. **Style-class mismatch**: multiple class variants map to the same canonical MNIST digit (e.g., two “1” styles), while MNIST labels only digit identity.
3. **Domain gap**: stroke shape, thickness, and writing dynamics in MNIST are broader than controlled synthetic perturbations.
4. **Evaluation projection loss**: 17-way predictions are projected to 10-way digits, which can mask style confusion but does not solve representation mismatch.

## 6) Practical conclusion

This is a successful **proof-of-concept one-shot style recognizer**, but not yet a competitive MNIST recognizer.

- Strong in-domain robustness (99%+ on transformed variants)
- Weak cross-domain generalization to MNIST (~10%)

## 7) Recommended next steps

1. Add stronger but still realistic augmentations (elastic distortions, local affine jitter).
2. Replace plain supervised one-shot with **metric learning** (prototypical/triplet loss).
3. Use a frozen pretrained feature extractor and nearest-prototype classification.
4. Add a hybrid training strategy: small curated real handwritten support set per class.
