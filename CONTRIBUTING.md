# Contributing

Thank you for your interest in the ASTRA synthetic gating demo.

## Scope

This repository is a **reproducibility package** for a specific preprint. Contributions that improve reproducibility, fix bugs, or improve documentation are welcome. Feature requests that change the scientific scope of the demo are out of scope.

## How to contribute

1. **Fork** the repository and create a branch from `main`.
2. **Install** the development environment:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install pytest ruff
   ```

3. **Make your changes.** Keep commits focused and descriptive.
4. **Run the tests** to verify nothing is broken:

   ```bash
   python -m pytest tests/
   python repro/run_astra.py
   python repro/verify_astra.py
   ```

5. **Lint** your code:

   ```bash
   ruff check .
   ```

6. **Open a pull request** against `main` with a clear description of the change.

## Sign-off requirement

Contributors must certify the Developer Certificate of Origin (DCO) by adding a `Signed-off-by:` trailer to each commit.

Example:

```bash
git commit -s -m "Describe the change"
```

## What makes a good contribution

- Bug fixes with a regression test
- Documentation improvements or clarifications
- Cross-platform compatibility fixes
- Improvements to the verification or CI pipeline

## What is out of scope

- Adding real detector data (this is a synthetic demo by design)
- Changing the scientific claims or methodology of the preprint
- Adding heavy dependencies beyond numpy

## Code style

- Python 3.11+, PEP 8
- Type hints on function signatures
- Use `pathlib.Path` instead of `os.path`
- Pin GitHub Actions to full commit SHAs in workflow files
- Keep `GITHUB_TOKEN` permissions minimal and explicit per job

## Security-aware review guidelines

- Do not introduce new third-party actions without provenance review and SHA pinning
- Do not interpolate GitHub event context directly into shell commands
- Do not add secrets, credentials, local paths, or generated artifacts to tracked files
- Highlight dependency additions and license changes in pull requests

## Reporting issues

Open an issue on GitHub with:

- What you expected to happen
- What actually happened
- Your OS and Python version
- Output of `python repro/verify_astra.py`

Security-sensitive issues should follow `SECURITY.md` instead of public issue filing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
