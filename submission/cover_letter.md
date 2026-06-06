# Cover letter — ICFHR 2026 submission

> Paste into the submission portal's cover-letter field, or adapt as an email
> to the program chairs. Fill the bracketed fields.

---

Dear ICFHR 2026 Program Chairs,

Please consider our paper, **"CultiVar-17: Three Data-Centric Strategies for One-Shot
Digit Recognition from a Culturally-Motivated Template Set,"** for presentation at ICFHR
2026.

The paper studies an extreme one-shot regime — building complete recognition pipelines
from exactly **17 hand-drawn digit templates**, one per culturally-motivated style class
(crossed vs. uncrossed 0/7, serif vs. plain 1, and other documented regional conventions).
We contribute:

1. **CultiVar-17**, a small, reproducible dataset whose class structure is grounded in
   real handwriting conventions rather than arbitrary splits;
2. a head-to-head comparison of **three complementary data-centric strategies** —
   hand-crafted morphological augmentation, learned augmentation via a class-conditional
   diffusion model, and an interpretable structural bag-of-features — on a common MNIST
   transfer benchmark;
3. the finding that **the augmentation process, not model capacity, is the dominant lever**:
   learned diffusion augmentation matches-to-exceeds expert-crafted augmentation
   (78.5% vs. 77.5%) with no domain-specific design, and an error analysis localises the
   residual gap to a 17-template style-coverage ceiling.

We believe the work fits ICFHR's scope directly: it concerns handwriting style, cultural
variation in digit forms, and data-centric recognition from minimal supervision. All code,
data, and the full experimental pipeline are publicly released for reproducibility.

The work is original and not under review elsewhere.

Thank you for your consideration.

Sincerely,
Attila Torda (Independent Researcher)
[email] · [repo URL]
