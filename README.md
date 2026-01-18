# ASTRA synthetic gating demo

## Scope / Non-Claim Statement
This repository supports the ASTRA synthetic gating demo: a controlled, fixed-seed toy experiment showing how an amplitude-based gating (preprocessing) step can suppress a short transient prior to downstream scoring. It is not a detector-data result and makes no astrophysical detection claim. No proprietary detector data are included.

## Licensing (Split by Artifact Type)
- Manuscript text and figures (the preprint and LaTeX prose): CC-BY-4.0 (see the Zenodo record metadata).
- Software code in this repository (Python package and helper scripts): MIT License (see LICENSE).
- Generated reproduction artifacts (e.g., logs, CSV, TeX tables, NPZ outputs produced by the scripts): CC0-1.0 (public domain dedication) unless otherwise noted.

Where a file contains its own license notice, that notice takes precedence for that file.

## Verification Contract (What “Same Results” Means)
Reproduced outputs must match the `zenodo_snapshot/` goldens exactly: logs/CSV/TeX are string-identical, and `astra_injection.npz` is byte-identical under the pinned environment (Python 3.11.9 + locked dependencies + deterministic settings); any mismatch is treated as a failure and reported with a manifest and numeric diffs.

PDF files are not expected to be byte-for-byte identical across platforms or TeX distributions (timestamps and PDF object IDs vary); verification is based on the numerical artifacts and their inclusion in the rebuilt manuscript.

## Directory structure
- `engine/` — MIT-licensed Python package source (`harmonic_matter_engine_v6`), including the ASTRA entrypoint (`python -m harmonic_matter_engine_v6.astra`).
- `paper/` — LaTeX manuscript sources used to rebuild the preprint (includes `paper.tex` and `sections/`).
- `repro/` — Cross-platform scripts to regenerate artifacts, build the PDF, and verify outputs against goldens (writes a run manifest for traceability).
- `zenodo_snapshot/` — Immutable, verbatim copy of the Zenodo deposit used as release evidence and golden-reference outputs; do not edit during development.
- `astra_output/` — Runtime-generated outputs from a non-MC run (created by the reproduction scripts).
- `astra_submission_bundle/` — Runtime-generated bundle layout used by the paper build and verification scripts.

## Quickstart (pinned runtime)
- Pinned runtime: Python 3.11.9

### 1) Install dependencies
Create and activate a Python 3.11.9 environment, then install:
- `pip install -r requirements.txt`

### 2) Generate artifacts
- `python repro/run_astra.py`

### 3) Verify against Zenodo goldens
- `python repro/verify_astra.py`

### 4) Build the paper PDF
- `python repro/build_paper.py`
