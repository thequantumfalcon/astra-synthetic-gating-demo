# ASTRA synthetic gating demo

[![ci](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/ci.yml?query=branch%3Amain)

## Security Health

[![CodeQL](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/codeql.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/thequantumfalcon/astra-synthetic-gating-demo/badge)](https://securityscorecards.dev/viewer/?uri=github.com/thequantumfalcon/astra-synthetic-gating-demo)
[![Dependency Review](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/dependency-review.yml)
[![SBOM](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/sbom.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/sbom.yml)
[![Release Attestation](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/release-attestation.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/release-attestation.yml)
[![Security Policy](https://img.shields.io/badge/security-policy-brightgreen)](SECURITY.md)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-025E8C?logo=dependabot)](.github/dependabot.yml)

## Scope / Non-Claim Statement

This repository supports the ASTRA synthetic gating demo: a controlled, fixed-seed toy experiment showing how an amplitude-based gating (preprocessing) step can suppress a short transient prior to downstream scoring. It is not a detector-data result and makes no astrophysical detection claim. No proprietary detector data are included.

## Licensing (Split by Artifact Type)

- Manuscript text and figures (the preprint and LaTeX prose): CC-BY-4.0.
- Software code in this repository (Python package and helper scripts): MIT License (see LICENSE).
- Generated reproduction artifacts (e.g., logs, CSV, TeX tables, NPZ outputs produced by the scripts): CC0-1.0 (public domain dedication) unless otherwise noted.

Where a file contains its own license notice, that notice takes precedence for that file.

## Citation

For citation metadata, see `CITATION.cff`.

## Verification Contract (What “Same Results” Means)

Reproduced outputs are checked for internal consistency using `repro/verify_astra.py`.

The verifier validates that expected artifacts are present and checks key content and schema constraints so runs are auditable and repeatable.

Run:

- `python repro/verify_astra.py`

PDF files are not expected to be byte-for-byte identical across platforms or TeX distributions (timestamps and PDF object IDs vary); verification is based on the numerical artifacts and their inclusion in the rebuilt manuscript.

CI (GitHub Actions) runs reproduction + verification on both Windows and Linux. The paper PDF is built in CI on Linux as a smoke test, but is not verified byte-for-byte.

## Directory structure

- `engine/` — MIT-licensed Python package source (`harmonic_matter_engine_v6`), including the ASTRA entrypoint (`python -m harmonic_matter_engine_v6.astra`).
- `paper/` — LaTeX manuscript sources used to rebuild the preprint (includes `paper.tex` and `sections/`).
- `repro/` — Cross-platform scripts to regenerate artifacts, build the PDF, and verify outputs (writes a run manifest for traceability).
- `astra_output/` — Runtime-generated outputs from a non-MC run (created by the reproduction scripts).
- `astra_submission_bundle/` — Runtime-generated bundle layout used by the paper build and verification scripts.

## Quickstart (pinned runtime)

- Pinned runtime: Python 3.11.9

### 1) Install dependencies

Create and activate a Python 3.11.9 environment, then install:

- `pip install -r requirements.txt`

### 2) Generate artifacts

- `python repro/run_astra.py`

### 3) Verify generated artifacts

- `python repro/verify_astra.py`

### 4) Build the paper PDF

- `python repro/build_paper.py`

Building the PDF requires a LaTeX distribution available on your PATH (e.g., MiKTeX on Windows, TeX Live on Linux).
