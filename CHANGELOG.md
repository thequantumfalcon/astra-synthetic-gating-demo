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
