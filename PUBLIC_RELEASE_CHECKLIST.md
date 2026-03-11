# Public Release Checklist

- Confirm the working tree is clean except for the intended release changes.
- Run `python -m pytest -q` and confirm all tests pass.
- Run `python repro/run_astra.py` and `python repro/verify_astra.py` if artifact regeneration is required for the release.
- Review `README.md`, `LICENSE`, `SECURITY.md`, and `CHANGELOG.md` for final public-facing wording.
- Confirm `.gitignore` excludes local artifacts, generated bundles, logs, and environment files.
- Verify no secrets, private identifiers, or internal-only files remain in tracked content.
- Verify the rewritten git history has been force-pushed and old clones are discarded.
- Tag the release only after the public history and release contents are final.
