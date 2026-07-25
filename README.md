# ASTRA synthetic gating demo

[![ci](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/ci.yml?query=branch%3Amain)

## Security Health

[![CodeQL](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/codeql.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/scorecards.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/scorecards.yml)
[![Dependency Review](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/dependency-review.yml)
[![SBOM](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/sbom.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/sbom.yml)
[![ClusterFuzzLite PR fuzzing](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/cflite_pr.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/cflite_pr.yml)
[![Release Attestation](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/release-attestation.yml/badge.svg)](https://github.com/thequantumfalcon/astra-synthetic-gating-demo/actions/workflows/release-attestation.yml)
[![Security Policy](https://img.shields.io/badge/security-policy-brightgreen)](SECURITY.md)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-025E8C?logo=dependabot)](.github/dependabot.yml)

## Scope / Non-Claim Statement

This repository supports the ASTRA synthetic gating demo: a controlled, fixed-seed toy experiment showing how an amplitude-based gating (preprocessing) step can suppress a short transient prior to downstream scoring. It is not a detector-data result and makes no astrophysical detection claim. No proprietary detector data are included.

## Licensing (Split by Artifact Type)

- Software code in this repository (Python package and helper scripts): MIT License — full text in [LICENSE](LICENSE).
- Manuscript text and figures (the preprint and LaTeX prose under `paper/`): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode).
- Generated reproduction artifacts (logs, CSV, TeX tables, NPZ outputs produced by the scripts): [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode) public domain dedication.

Only the MIT text is bundled in this repository; the CC-BY-4.0 and CC0-1.0 grants above
are made by reference to the canonical texts linked. Where a file contains its own
license notice, that notice takes precedence for that file.

## Citation

For citation metadata, see `CITATION.cff`.

## Artifact Consistency Checks (What Is and Is Not Verified)

Run:

- `python repro/verify_astra.py`

`repro/verify_astra.py` checks the regenerated bundle against a fixed contract: that
the four expected artifacts exist, that `verification_log.txt` and `mc_summary.csv`
carry the required fields, that the run used seed 123 and produced 200 Monte Carlo
rows, that `snr_after < snr_before`, and that the gated fraction lands in `(0, 0.1)`.

**It does not compare output against a stored reference.** These are internal
consistency and invariant checks, not a golden-file diff — a run that satisfied every
invariant with different numbers would still pass. Peak-based statistics can differ in
the final ulp across numpy releases, which is why `requirements-lock.txt` pins numpy
and why CI installs from it.

PDF files are not expected to be byte-for-byte identical across platforms or TeX
distributions (timestamps and PDF object IDs vary), and the verifier does not inspect
the built PDF at all.

CI (GitHub Actions) runs reproduction + verification on both Windows and Linux. The
paper PDF is built in CI on Linux as a smoke test, but is not verified byte-for-byte.

Published GitHub releases attach a reproducibility bundle: `astra-release-artifacts.tgz`, its `astra-release-artifacts.tgz.sha256` checksum, a CycloneDX SBOM (`sbom.cdx.json`), and an in-toto provenance bundle (`astra-release-artifacts.tgz.intoto.jsonl`).

Pull requests are fuzzed with ClusterFuzzLite against the JSON config load/save
helpers in `engine/astra/utils.py`. The fuzzer covers those helpers only; it does not
exercise the gating code path.

## Directory structure

- `engine/` — MIT-licensed Python package source (`astra`), including the demo entrypoint (`python -m astra`).
- `paper/` — LaTeX manuscript sources used to rebuild the preprint (includes `paper.tex` and `sections/`).
- `repro/` — Cross-platform scripts to regenerate artifacts, build the PDF, and verify outputs (writes a run manifest for traceability).
- `tests/` — Unit tests for the gating functions and the reproduction driver.
- `.github/` — CI, CodeQL, SBOM, release-attestation and Scorecard workflows, plus branch-protection ruleset payloads.
- `.clusterfuzzlite/` — ClusterFuzzLite PR fuzzing target and build definition.
- `scripts/` — Maintainer helper for auditing the repository's GitHub hardening settings.
- `ops/`, `docs/` — Terraform ruleset definitions and the notes describing how they are applied.
- `astra_output/` — Runtime-generated outputs from a non-MC run (created by the reproduction scripts).
- `astra_submission_bundle/` — Runtime-generated bundle layout used by the paper build and verification scripts.

## Quickstart

- Pinned runtime: Python 3.11.9. Dependency pins for reproduction live in `requirements-lock.txt`.

### 1) Install

Create and activate a Python 3.11.9 environment, then install the pinned
dependencies and the package itself:

- `pip install -r requirements-lock.txt`
- `pip install -e .`

Installing the package is what makes `python -m astra` and the `astra` console script
available. The `repro/` scripts below add `engine/` to `PYTHONPATH` themselves, so they
also work from a clean clone with only `pip install -r requirements.txt`.

### 2) Generate artifacts

- `python repro/run_astra.py`

### 3) Verify generated artifacts

- `python repro/verify_astra.py`

### 4) Build the paper PDF

- `python repro/build_paper.py`

Building the PDF requires a LaTeX distribution available on your PATH (e.g., MiKTeX on Windows, TeX Live on Linux).
