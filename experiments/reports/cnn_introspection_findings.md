# Track 8 — extracting latent variant structure from CNNs (investigation)

**Question.** (1) Can the style *variants* of a digit (e.g. crossed vs uncrossed 7) be
recovered from a CNN that was only trained on 10-class labels? (2) If so, can that be used
to improve training — by weighting "basic truths" more, or by splitting a class into
sub-classes (add neurons + a 17→10-style map-back)?

## Finding 1 — the variant IS recoverable from the CNN (JUSTIFIED)

A plain 10-class CNN (98.98% test) never sees variant labels. We derive a cheap variant
pseudo-label from the Track 6 skeleton graph (crossed-7 = has a junction/crossbar) and probe.

| digit 7 (crossed/uncrossed, base rate 57.3%) | accuracy |
|----------------------------------------------|---------:|
| linear probe on **CNN embedding** | **83.8%** (+26.5pp over base) |
| linear probe on raw pixels (reference) | 81.6% |
| KMeans(2) on embedding vs variant (ARI) | 0.02 |

The CNN **preserves** the crossed/uncrossed distinction linearly — the embedding probe even
beats the raw-pixel probe. So style information the network was never asked to learn survives
in its representation and is extractable.

**Crucial nuance — where the variant lives.** Unsupervised KMeans on the *CNN embedding*
fails to find the variant (ARI 0.02): the embedding is organised by digit identity, and the
variant is only a faint linear sub-axis. But KMeans in the **structural-feature space**
(Track 6's 93-dim) recovers it strongly: **ARI 0.83** vs the crossbar label. So variants must
be defined in *structural* space, not by clustering CNN embeddings.

(The digit-1 probe was inconclusive: 94% of MNIST 1s are "plain", so the imbalance pins the
probe at the base rate — a heuristic/data artifact, not evidence of absence.)

## Finding 2 — naive sub-class expansion HURTS (FALSIFIED, with a clear cause)

Idea #2b (the user's "add a neuron + 17→10 map-back"): split each digit into 2
structure-defined sub-classes (structural-space KMeans), train a 2k-way CNN, sum sub-class
probabilities back to 10 digits. Compared to a plain 10-way CNN at low budgets (3 seeds):

| labels n | plain 10-way | sub-class (map-back) | delta |
|---------:|-------------:|---------------------:|------:|
| 100 | 60.6% | 48.3% | **−12.2pp** |
| 250 | 73.9% | 69.4% | −4.5pp |
| 500 | 83.6% | 80.4% | −3.2pp |
| 1000 | 90.9% | 89.1% | −1.8pp |
| 2500 | 95.3% | 94.5% | −0.7pp |
| 5000 | 96.9% | 96.8% | −0.2pp |

Hard sub-class labels **fragment scarce data** — splitting each digit's examples in two halves
the data per class, so the expanded head overfits. The penalty shrinks monotonically with `n`
(−12pp → −0.2pp), converging to break-even once data is plentiful. It never helps in range.

Idea #2a (weight prototypical "basic truths" more in backprop) was not run, but Track 7
already showed prototypical = easy/central/redundant; up-weighting them would reduce
boundary focus and is expected to hurt for the same reason.

## Conclusion & open direction

- The latent variant structure is **real and extractable** (#1 ✓), and structural features —
  not CNN-embedding clustering — are the right lens.
- Exploiting it by **hard relabeling** backfires (#2b ✗): it starves each sub-class of data.
- **The principled fix (untested): an auxiliary sub-class head.** Keep the full 10-way loss
  on all data (no fragmentation) and add a shared-backbone sub-class head as a regularising
  auxiliary task (`loss = CE_10 + λ·CE_subclass`). This injects the structural-variant signal
  without splitting the main task's data, and is the most likely route to a *positive* result.

Reproduce: `python scripts/probe_variant_recovery.py` (Finding 1),
`python scripts/run_subclass_expansion.py` (Finding 2). Raw numbers:
`subclass_expansion_results.json`.

## Future ideas (not yet run)

1. **Auxiliary sub-class head (most promising).** Multi-task CNN: shared backbone, a 10-way
   main head trained on all data, plus a sub-class head (structure-defined variants) as a
   regularising auxiliary loss `CE_10 + λ·CE_subclass`. Injects the variant signal without
   fragmenting the main task — the route to a *positive* #2 result. Reuses
   `assign_subclasses` (structural KMeans) from `run_subclass_expansion.py`.
2. **Idea #2a — weight prototypical "basic truths" more in backprop.** Per-example loss
   weights ∝ prototypicality (Track 7 `exemplar.py`). Expected to *hurt* (Track 7: prototypical
   = easy/redundant/low boundary value); worth a quick confirmatory A/B and possibly the
   *inverse* (up-weight hard/atypical examples).
3. **Reconstruct, not just decode.** Visualise the variant axis (e.g. condition the Track 5
   DDPM on the probe direction, or feature-vis the crossbar sub-axis) to *generate* the two
   7-variants from the net — turning Finding 1 from "decodable" into "reconstructable".
