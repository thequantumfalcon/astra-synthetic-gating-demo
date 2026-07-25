# Public Release Checklist

- Confirm the working tree is clean except for the intended release changes.
- Run `python -m pytest -q` and confirm all tests pass.
- Run `python repro/run_astra.py` and `python repro/verify_astra.py` if artifact regeneration is required for the release.
- Review `README.md`, `LICENSE`, `SECURITY.md`, and `CHANGELOG.md` for final public-facing wording.
- Confirm `.gitignore` excludes local artifacts, generated bundles, logs, and environment files.
- Verify no secrets, private identifiers, or internal-only files remain in tracked content.
- Confirm `CHANGELOG.md`, `CITATION.cff`, and `pyproject.toml` agree on the version, and that the date matches the release being cut.
- Run `ruff check .` and `ruff format --check .`.
- Open the rendered README on GitHub and confirm every badge resolves (a badge in an error state is a release blocker).
- After publishing, confirm the release page carries the tgz, its sha256, `sbom.cdx.json`, and the in-toto provenance bundle.
- Re-apply `.github/rulesets/main-protection.json` whenever CI job names change, so required checks still match reality.
- Tag the release only after the public history and release contents are final.
