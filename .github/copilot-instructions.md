# Copilot instructions (ASTRA reproducibility repo)

## Big picture
This repo is a runnable, public-facing reproducibility package for the **ASTRA synthetic gating demo** (v5, 2026-01-16): LaTeX manuscript sources plus scripts that regenerate the key artifacts and verify them against an immutable Zenodo snapshot.

Treat `zenodo_snapshot/` as **release evidence**: don’t edit its contents during development.

## Layout / entrypoints
- `engine/harmonic_matter_engine_v6/`: MIT-licensed Python package; ASTRA entrypoint is `python -m harmonic_matter_engine_v6.astra`.
- `paper/`: LaTeX manuscript sources (`paper.tex`, `sections/`, etc.).
- `repro/`: portable scripts:
  - `repro/run_astra.py` generates `astra_submission_bundle/` and `astra_output/`.
  - `repro/verify_astra.py` verifies outputs against goldens extracted from the snapshot’s `ASTRA_Ancillary_v5.zip`.
  - `repro/build_paper.py` builds the PDF from `astra_submission_bundle/`.
- `zenodo_snapshot/zenodo_deposit_2026-01-16_v5_final/`: immutable snapshot of the Zenodo deposit (goldens + audits + checksums).
- Legacy/original deposit packaging helpers may still exist at repo root (e.g., `UPLOAD_TO_ZENODO.txt`), but development should target the folders above.

## Critical workflows (don’t guess; follow the recorded ones)
- Regenerate + verify artifacts:
  - `python repro/run_astra.py`
  - `python repro/verify_astra.py`
- Build manuscript PDF from the assembled bundle:
  - `python repro/build_paper.py` (runs `pdflatex → bibtex → pdflatex → pdflatex`)
- Keep snapshot integrity consistent:
  - Don’t edit `zenodo_snapshot/` during normal development.
  - If you intentionally cut a new release snapshot, regenerate checksums and update audits as part of that release process.

## Project-specific conventions
- Scope language is intentional: the deposit is a **synthetic** demo and explicitly “not a detection claim” (mirrors `README_ZENODO.md` + `zenodo.json`). Preserve this framing in edits.
- Metadata is duplicated between deposit layouts; if you update `CITATION.cff`, `zenodo.json`, `README_ZENODO.md`, or `SHA256SUMS.txt`, keep the corresponding files in both snapshot and any packaging mirrors in sync.
- LaTeX source structure:
  - `paper/paper.tex` is the root; content lives in `paper/sections/*.tex`.
  - The bundle build uses `astra_submission_bundle/` so the manuscript can include generated artifacts.
- Python scripts:
  - ASTRA proof code is intentionally NumPy-only for minimal dependencies; keep it lightweight.

## When making changes
Prefer edits in `engine/`, `paper/`, and `repro/`.

Avoid editing `zenodo_snapshot/` except when intentionally producing a new release snapshot.
