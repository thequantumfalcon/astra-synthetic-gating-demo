# Changelog

All notable changes to this reproducibility package are documented in this file.

## [6.1.2] - 2026-07-25

### Fixed
- The release archive carried LaTeX build byproducts (`paper.log`, `.aux`, `.out`,
  `.bbl`, `.blg`). TeX writes absolute font and input paths into those files, so a
  bundle produced locally embedded the builder's home directory. They are now stripped
  before the archive is created; `paper.pdf` and every data artifact are unaffected.
  Published archives were built in CI and contained only the runner's own paths.

## [6.1.1] - 2026-07-25

### Fixed
- The run manifest recorded `python_executable`, the absolute path of the interpreter
  that produced the bundle. That path ships inside the release tarball, so anyone
  reproducing the work locally and sharing the result would disclose their home
  directory and username. The manifest now records the Python implementation and
  version instead, which is what reproducing the run actually needs. Published release
  tarballs were built in CI and only ever contained the runner's path, so nothing was
  disclosed. A test asserts the manifest contains no local paths.

## [6.1.0] - 2026-07-25

### Fixed
- Reported statistics now reproduce on any machine. Two defects were behind this.
  `np.std` reduces in an order that depends on SIMD width, so identical values summed
  to a different final ulp on different CPUs, shifting `threshold = 8*std` and moving
  the reported numbers; `population_std()` now reduces with `math.fsum`, which is
  correctly rounded and order-independent. Separately, numpy's normal generator is not
  bit-reproducible across C runtimes — a handful of the 245,760 draws differ by one ulp
  between machines, visible even between two Linux CI runners with different CPUs — so
  serialising these statistics to 17 significant digits was claiming precision that
  encoded which processor ran the job. `report()` rounds derived statistics to ten
  significant digits, well beyond what a peak-based SNR proxy carries. Full-precision
  arrays are still written to the NPZ.

### Added
- `repro/check_reference_artifacts.py` diffs regenerated artifacts against the reference
  copies in `paper/` and runs in CI on both operating systems. `verify_astra.py` only
  ever checked internal consistency, so a run producing different numbers passed as long
  as the invariants held. The comparison is exact, with no tolerance.

## [6.0.0] - 2026-07-25

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
- Updated all pinned actions (checkout v7.0.1, codeql-action v4.37.3, upload-artifact
  v7.0.1, setup-python v7.0.0, dependency-review v5.0.0, attest-build-provenance
  v4.1.1, scorecard-action v2.4.4, harden-runner v2.20.0) and pinned zizmor. Every pin
  now carries its resolved version as a trailing comment.
- Pinned the ClusterFuzzLite base image by digest and added a docker ecosystem to
  Dependabot so the digest does not go stale unnoticed.
- `scripts/audit-github-hardening.sh` printed `FAIL` lines but always exited 0, and
  reported "authenticated for remote security checks" without performing one. It now
  exits non-zero on failure and actually checks open Dependabot alerts, whether private
  vulnerability reporting is enabled, and whether every status check the live ruleset
  requires corresponds to a workflow that still exists. That last check would have
  caught the `engine-jax-tests` breakage in this release. It runs in CI on every PR.
- Renamed `hardened-ci-template` to `workflow-lint`. It was never a template: it is the
  repository's only workflow-security gate (zizmor) and it ran on every PR. It also
  duplicated a test run that `ci` already performs on two operating systems, which has
  been dropped. The required-status-check lists were updated in the same change.
- `requirements-lock.txt` now documents that its numpy pin is deliberately frozen to the
  version that produced the committed reference artifacts, and what raising it requires.

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
