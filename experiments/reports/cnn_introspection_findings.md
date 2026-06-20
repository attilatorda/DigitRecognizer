# Track 8 — extracting latent variant structure from CNNs (investigation)

> **Status: REOPENED (2026-06).** Findings 1–2 stand (variant is decodable; hard relabeling
> hurts). The track was extended from *decoding* to *reconstruction + logic extraction*
> (Findings 3–5 below), answering three questions: can we rebuild digits, rebuild variants, and
> extract logic (XOR) from a network? Short answers: **yes**, **yes at the representation level**,
> and **yes (cleanly, for minimal Boolean nets)**. The auxiliary sub-class head remains
> documented future work, not pursued.

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

## Finding 3 — digits ARE graphically reconstructable from the net (JUSTIFIED)

Activation maximization (gradient ascent on the input to maximize a class logit, with
total-variation / L2 / periodic-blur regularizers) rebuilds a recognizable image for **all 10
digits**; each reconstruction is **self-classified as its target class at 100% confidence**.
The images are ghostly and tiled (the activation-max signature — the CNN's
translation-equivariance produces repeated motifs) rather than photorealistic, but the digit
forms (0,1,2,3,5,6,7,8,9 especially) are clearly legible. So a discriminative CNN carries a
recoverable *generative* template of each class in its weights. Figure:
`figures/fig_track8_reconstruct.png`. Reproduce: `scripts/run_introspection_reconstruct.py`.

## Finding 4 — the variant axis is GENERATIVE in representation space (JUSTIFIED, with a caveat)

Taking the digit-7 crossed/uncrossed probe direction $w$ (Finding 1) and reconstructing a 7
while pushing its embedding along $\pm w$ moves the embedding's variant score **monotonically
from $-8.2$ (uncrossed) to $+25.0$ (crossed)** across the sweep, and the pixel reconstruction
visibly shifts toward crossbar structure at the crossed end. So Finding 1's *decodable* axis is
also *steerable/generative* — the network can be driven to render either variant.
**Caveat:** the reconstructions are too noisy for a clean pixel-level crossbar metric — skeleton
junction counts ($\sim$10–14 across the sweep) are dominated by activation-max speckle, so the
claim rests on the monotone probe score (quantitative) plus visual inspection, not a junction
count. Figure: `figures/fig_track8_variants.png`.

## Finding 5 — logic (XOR) IS exactly extractable from minimal nets (JUSTIFIED)

For the six 2-input Boolean gates, a minimal 2-2-1 MLP fits each truth table and the logic is
**exactly recovered** by enumerating the four inputs (**6/6 gates**). The XOR mechanism reads
straight off the weights: the two hidden units become single-corner detectors $\neg a\wedge b$
and $a\wedge\neg b$, OR-combined at the output — i.e. **XOR $=(\neg a\wedge b)\vee(a\wedge\neg
b)$**, the textbook sum-of-minterms. The control confirms necessity of depth: a 0-hidden linear
unit solves the linearly separable gates (AND/OR/NAND/NOR at 100%) but **cannot fit XOR/XNOR
(50%)**. Logic extraction is exact for small nets; extracting XOR-like structure from the full
MNIST CNN would be approximate (not pursued). Figure: `figures/fig_track8_logic.png`; raw:
`track8_logic_results.json`. Reproduce: `scripts/run_logic_extraction.py`.

## Finding 6 — model differences encode INVARIANCE, read from robustness not accuracy (JUSTIFIED)

A panel of three classifiers with increasing spatial structure — an MLP, the 2-conv SimpleCNN,
and the An et al.\ record-holder single model (M3) — spans only **1.4 pp on clean MNIST**
($98.08 / 99.12 / 99.47$, 3 seeds) yet diverges by **~46 pp under a 4-pixel translation**
($39.7 / 71.7 / 85.8$): the architectural gap is an *invariance* gap nearly invisible on clean
data. The ordering MLP $\ll$ SimpleCNN $\ll$ record-M3 holds for both translation and rotation.
So "model A beats model B" is best read not from clean accuracy but from **perturbation
robustness**, which exposes what each architecture computes (translation tolerance from the
convolutional prior; more from depth/no-pooling). **Honest null:** an unsupervised style-variant
probe (crossed-7) reaches ~80% on all three representations (base 57%), so structure preservation
is architecture-independent — the discriminating axis is robustness, not representation richness.
Figure: `figures/fig_track8_panel_robust.png`; raw: `track8_panel_results.json`. Reproduce:
`scripts/run_track8_model_panel.py`.

## Finding 7 — depth (not kernel size) builds clean internal class prototypes (JUSTIFIED, qualitative)

Activation-maximizing each class for the shallow SimpleCNN vs.\ the record holder's deep M3/M5/M7
sub-models shows the deep models render **smooth, prototype-like canonical digits** (mean total
variation $\sim\!0.08$) where the shallow CNN renders **high-frequency texture** that barely
resembles a digit (TV $\sim\!0.5$). The hypothesized "different receptive-field scales" story is
**not** supported — smoothness tracks *depth*, not kernel size. What the record holder's design
buys, readable here, is (i) depth that forms clean class prototypes and (ii) three decorrelated
sub-models whose templates differ — the basis of the ensemble. Figure:
`figures/fig_track8_receptive_fields.png`. Reproduce: `scripts/run_track8_introspect_best.py`.

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
