# Changelog

All notable changes to this reproducibility package are documented in this file.

## [Unreleased]

### Removed
- Removed modules inherited from an unrelated parent project that were never part of
  this demo: the physics-surrogate stub, the agent and Gaussian-splat modules, the
  SPH solver, the LiteRT export helper, the standalone `main.py` orchestrator, and
  their bundled config and output fixtures. This drops the optional JAX dependency
  and the `engine-jax-tests` CI job along with them.
- Removed `run_astra_kernel()`, which printed a "predicted GW strain" derived from
  invented vacuum-collapse parameters and then discarded that arithmetic in favour of
  a hard-coded constant. The injected burst amplitude is now stated plainly as the
  illustrative value it always was, consistent with the repository's Non-Claim Statement.

### Changed
- Renamed the Python package from `harmonic_matter_engine_v6` to `astra`; the entrypoint
  is now `python -m astra`. Documentation, the manuscript, and the reproduction scripts
  were updated to match.
- Renamed `verify_gating_paradox()` to `run_gating_trial()` to describe what it does.
- Reproduction artifacts (`mc_summary.csv`, `mc_table.tex`, `astra_injection.npz`,
  `verification_log.txt`) are unchanged; the verification contract still holds.

### Fixed
- Fixed the OpenSSF Scorecard workflow, which had failed every scheduled run since
  2026-06-08 because top-level workflow write permissions are rejected by the
  publishing step. Permissions are now read-only at the top level and scoped to the job.
- Removed the `harmonic-matter-engine-v6` console script, which pointed at a module
  requiring JAX and so raised `ImportError` on any install.
- Corrected documentation that did not match the code: the verification section
  described a golden-reference comparison the verifier does not perform, claimed
  verification covered the rebuilt manuscript, implied the demo itself is fuzzed, and
  documented an entrypoint that failed from a clean clone because the Quickstart never
  installed the package. The OpenSSF Scorecard badge, which rendered "invalid repo
  path", was replaced with the workflow badge.
- Aligned metadata that had drifted: `CITATION.cff` declared CC-BY-4.0 for a software
  entry that LICENSE and `pyproject.toml` call MIT, the v5.0 changelog date was wrong
  by two weeks, `SECURITY.md` hedged about private vulnerability reporting that is
  enabled, and `ops/github-rulesets.tf` targeted a different ref set than the
  checked-in JSON and the live ruleset.

### Security
- `requirements-lock.txt` was installed by nothing while carrying all four open
  Dependabot advisories. It is now a real pin, consumed by the reproduction, PDF-build
  and release jobs, and holds only the runtime dependency; the superseded pip and
  setuptools pins that generated the advisories are gone.
- All eight workflows are now read-only at the workflow level with writes scoped to the
  jobs that need them. Scorecard had scored Token-Permissions 0/10.
- The published CycloneDX SBOM was generated from the unpinned requirements file and
  recorded numpy with no version; it now reads the lock file.
- Updated all pinned actions (checkout v6.0.3, codeql-action v4.36.1, upload-artifact
  v7.0.1, setup-python v6.2.0, dependency-review v5.0.0, attest-build-provenance
  v4.1.0, harden-runner v2.19.4) and pinned zizmor.

## [v5.0.3] - 2026-02-10

### Changed
- Updated release metadata and bumped the project version to `v5.0.3`.
- Clarified the appendix, bibliography, reproduction instructions, and repository metadata.

### Fixed
- Fixed taper overlap and same-length cross-correlation behavior in `paper/astra_real_verify.py`.
- Adjusted CI to build the manuscript PDF on Linux and refreshed `requirements-lock.txt`.

## [v5.0] - 2026-01-18

### Added
- Added CI status reporting and the public reproducibility package baseline used for the first tagged release.
