#!/usr/bin/env bash
# Build self-contained arXiv submission bundles for each paper.
# For every paper: copy the .tex + its .bib + neurips_2024.sty (if used) + the figures it
# \includegraphics, then COMPILE the bundle in isolation (the real test that it is complete),
# leaving a fresh .bbl, and tar it. Output: experiments/reports/arxiv/<name>/ and <name>.tar.gz
set -u
cd "$(dirname "$0")/.." || exit 1
REP=experiments/reports
OUT=$REP/arxiv
rm -rf "$OUT"; mkdir -p "$OUT"

papers=(cultivar17_paper track9c_paper_neurips track9_benchmark_paper skeleton_paper track8_paper)

for tex in "${papers[@]}"; do
  d=$OUT/$tex; mkdir -p "$d/figures"
  cp "$REP/$tex.tex" "$d/"
  # bib
  bib=$(grep -oE 'bibliography\{[^}]+\}' "$REP/$tex.tex" | sed -E 's/bibliography\{|\}//g')
  [ -n "$bib" ] && cp "$REP/$bib.bib" "$d/"
  # neurips style if used
  grep -q 'neurips_2024' "$REP/$tex.tex" && cp "$REP/neurips_2024.sty" "$d/"
  # figures actually referenced
  for f in $(grep -oE 'figures/[A-Za-z0-9_./-]+\.(png|pdf|jpg)' "$REP/$tex.tex" | sort -u); do
    cp "$REP/$f" "$d/figures/" 2>/dev/null || echo "  !! missing $f for $tex"
  done
  # compile in isolation
  ( cd "$d" && pdflatex -interaction=nonstopmode "$tex.tex" >/dev/null 2>&1 \
       && bibtex "$tex" >/dev/null 2>&1 \
       && pdflatex -interaction=nonstopmode "$tex.tex" >/dev/null 2>&1 \
       && pdflatex -interaction=nonstopmode "$tex.tex" > _final.log 2>&1 )
  pages=$(grep -oE 'Output written.*\([0-9]+ pages' "$d/_final.log" | grep -oE '[0-9]+ pages')
  undef=$(grep -ciE 'undefined (citation|reference)' "$d/_final.log")
  # keep only submission files (tex, bib, bbl, sty, figures); drop aux
  ( cd "$d" && rm -f ./*.aux ./*.log ./*.out ./*.blg _final.log )
  ( cd "$OUT" && tar -czf "$tex.tar.gz" "$tex" )
  nfig=$(ls "$d/figures" | wc -l)
  echo "$tex: ${pages:-FAILED}, undefined=$undef, figs=$nfig -> $tex.tar.gz"
done
echo "--- bundles in $OUT ---"; ls -1 "$OUT"/*.tar.gz