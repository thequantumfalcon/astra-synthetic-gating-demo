# Zenodo deposit: ASTRA synthetic gating demo (preprint + reproducibility, v5)

Author: Thomas Albrecht (Independent Researcher)
License: CC-BY-4.0
Release: v5 (2026-01-16)

This deposit is a citable archive of a preprint and its reproducibility materials for a **synthetic** demonstration of an amplitude-based gating step and its effect on downstream scoring metrics.

**What it is:** a controlled, fixed-seed toy experiment + audited artifacts that reproduce the same outputs end-to-end.

**What it is not:** a detector-data result or a claim of any astrophysical detection; no proprietary detector data are included.

## Included files
- `paper.pdf` — compiled manuscript PDF
- `ASTRA_Source_v5.zip` — LaTeX source bundle (clean; includes `paper.bbl` and `references.bib`)
- `ASTRA_Ancillary_v5.zip` — reproducibility artifacts (logs/tables)
- `UPLOAD_AUDIT_v5.txt` — packaging audit notes
- `COMPILE_AUDIT_25X_v5.txt` — 25× clean-room compile audit (25/25 passes)
- `SHA256SUMS.txt` — checksums for integrity

## Reproduce the PDF
Extract `ASTRA_Source_v5.zip` and run:
- `pdflatex paper.tex`
- `bibtex paper`
- `pdflatex paper.tex`
- `pdflatex paper.tex`

## Integrity
You can verify file integrity by comparing SHA-256 digests against `SHA256SUMS.txt`.
