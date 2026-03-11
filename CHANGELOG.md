# Changelog

All notable changes to this reproducibility package are documented in this file.

## [Unreleased]

### Fixed
- Removed the duplicated `paper/astra_proof.py` source and made the manuscript bundle include the engine copy directly.
- Seeded `WalrusSurrogate` with `numpy.random.default_rng()` and added a CLI seed option for deterministic engine runs.
- Replaced library `print()` calls in the engine support modules with module logging and cleaned up the LiteRT export helper.

### Added
- Added installable CLI entry points for `astra` and `harmonic-matter-engine-v6`.
- Added engine-focused regression tests for the Walrus surrogate and bundle assembly path.

## [v5.0.3] - 2026-02-10

### Changed
- Updated release metadata and bumped the project version to `v5.0.3`.
- Clarified the appendix, bibliography, reproduction instructions, and repository metadata.

### Fixed
- Fixed taper overlap and same-length cross-correlation behavior in `paper/astra_real_verify.py`.
- Adjusted CI to build the manuscript PDF on Linux and refreshed `requirements-lock.txt`.

## [v5.0] - 2026-02-03

### Added
- Added CI status reporting and the public reproducibility package baseline used for the first tagged release.
