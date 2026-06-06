# ICFHR 2026 submission checklist

A pre-submission checklist for `cultivar17_paper.pdf`. Items marked **[you]** need a
human decision/action; the rest are verifiable in-repo.

## Paper content
- [x] Title, abstract, keywords present
- [x] All three strategies (Tracks 4/5/6) described with results
- [x] 6 figures/tables cross-referenced in text; 0 undefined references on compile
- [x] Related Work + 11 references (BibTeX), error analysis, threats to validity
- [x] Reproducibility section with exact commands
- [ ] **[you]** Fill author email + final affiliation (currently "Independent Researcher")
- [ ] **[you]** Add acknowledgements if any (placeholder text is generic; fine as-is)
- [ ] **[you]** Confirm page limit — currently **6 pages**; ICFHR allows up to 8 (IEEE
      two-column). Within limit.

## Formatting
- [x] IEEEtran `conference` document class
- [ ] **[you]** Confirm ICFHR 2026 uses the standard IEEE conference template (it has
      historically). If they mandate a specific style file, swap `\documentclass`.
- [ ] **[you]** Anonymise for double-blind review IF required — remove the author block,
      the "Independent Researcher" line, and any self-identifying repo URLs/paths. Check
      the 2026 CFP for whether review is double-blind.

## Reproducibility / artifacts
- [x] Dataset (CultiVar-17) tracked in-repo (`data/processed/mnist17_variants/`)
- [x] All experiment scripts committed; commands in the paper + README
- [x] Results JSON committed (`oneshot_results.json`,
      `diffusion_experiment_results.json`, `structural_v3_results.json`)
- [ ] **[you]** Set the repo URL in `CITATION.cff`, `arxiv_submission.txt`,
      `cover_letter.md` (currently `REPLACE_WITH_REPO_URL` / `[repo URL]`)
- [ ] **[you]** Decide whether to release the repo public before or after acceptance

## Submission steps (ICFHR portal)
1. [ ] **[you]** Create/confirm the submission-system account (CMT/EasyChair — per CFP)
2. [ ] **[you]** Upload `cultivar17_paper.pdf`
3. [ ] **[you]** Paste abstract (plain text from `arxiv_submission.txt`)
4. [ ] **[you]** Select subject area / keywords
5. [ ] **[you]** Cover letter (from `cover_letter.md`) if the portal asks for one
6. [ ] **[you]** Confirm "original, not under review elsewhere"

## Optional: arXiv preprint
- [ ] **[you]** Decide timing (many venues allow non-anonymous preprints; check CFP)
- [ ] **[you]** Submit via `arxiv_submission.txt` metadata; upload `.tex` + `.bib` + `.bbl`
      + `figures/` (NOT the compiled PDF as source)

## Recompile-from-clean sanity check
```bash
cd experiments/reports
pdflatex cultivar17_paper && bibtex cultivar17_paper \
  && pdflatex cultivar17_paper && pdflatex cultivar17_paper
# expect: "Output written ... (6 pages)", 0 undefined references
```
