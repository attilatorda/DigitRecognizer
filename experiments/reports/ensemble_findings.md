# Ensemble investigation — findings

**Question:** does an ensemble of the three one-shot strategies (including the
skeleton-based structural recogniser) make sense?

**Short answer:** the diversity is real and the skeleton-based member genuinely
contributes orthogonal correctness — but in the *one-shot* regime the payoff is small,
because the large latent headroom can only be unlocked with labels, which exits one-shot.

## 1. Diagnostic — is there diversity? (`analyze_ensemble_potential.py`)

Per-image predictions from the three members on the 10k MNIST test:

| Member | Accuracy |
|--------|---------:|
| Morphological proto | 77.4% |
| DDPM proto | 80.6% |
| Structural RF (skeleton-based) | 73.2% |
| **Oracle** (≥1 member correct) | **91.1%** |

The structural member agrees with the neural members only 68–73% of the time (vs 81%
between the two neural members) and the oracle ceiling is +10pp above the best single
method — so the methods err on genuinely different images. Diversity confirmed.

## 2. Seed-ensembling — a free, label-free win (`build_member_predictions.py`)

Combining each proto method's 5 already-trained seeds (label-free):

| Method | single-seed mean | hard vote | **soft (dist-avg)** |
|--------|----:|----:|----:|
| Morphological proto | 75.2% | 77.4% | 77.9% |
| DDPM proto | 78.5% | 80.6% | **81.5%** |

DDPM proto gains **+3pp** for free. Fully in-regime (no MNIST labels).

## 3. Combining the three members (`run_stacking_ensemble.py`)

Evaluated leakage-free (base members never saw MNIST; any meta-classifier is trained on
half the MNIST test, evaluated on the disjoint half).

| Combiner | Accuracy | One-shot valid? |
|----------|---------:|:---------------:|
| Best single member | 81.3% | ✅ |
| **Soft-average (label-free)** | **82.8%** (+1.5pp) | ✅ |
| Stacking meta-classifier (logreg / histgb) | 92.8% / 96.6% | ❌ |

## The boundary finding

The supervised stacker reaches 96.6% — but it is trained on 5,000 labeled MNIST images,
so it is **not a one-shot method**; it is a small supervised classifier over the members'
probability vectors (whose top-3 accuracy is ~95%, so labels easily recover the truth).
It *exceeds the argmax oracle* precisely because it exploits soft top-k information with
supervision.

**Conclusion.** In the one-shot regime, ensembling the diverse recognisers gives only a
modest label-free gain (best single 81.3% → soft-average 82.8%, plus the +3pp from
seed-ensembling). The large 10pp oracle headroom is real but unlockable only with
supervision — at which point the task is no longer one-shot. The skeleton-based structural
member *does* add orthogonal signal (validating the original intuition), but the regime,
not the method, caps the payoff. A standalone "ensemble track" is therefore not worth a
separate paper section beyond reporting the small label-free gain and this boundary.

Raw numbers: `stacking_ensemble_results.json`. Reproduce:
```bash
python scripts/analyze_ensemble_potential.py     # diagnostic (oracle headroom)
python scripts/build_member_predictions.py        # seed-ensemble + export member probs
python scripts/run_stacking_ensemble.py           # label-free vs supervised combiners
```
