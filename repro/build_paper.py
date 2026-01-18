from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "astra_submission_bundle"


def _run(cmd: list[str], cwd: Path) -> None:
    print("[build] $", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    if not BUNDLE_DIR.exists():
        raise SystemExit(
            f"Bundle directory not found: {BUNDLE_DIR}. Run: python repro/run_astra.py"
        )

    tex = BUNDLE_DIR / "paper.tex"
    if not tex.exists():
        raise SystemExit(f"paper.tex not found in bundle: {tex}")

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex or not bibtex:
        raise SystemExit(
            "Missing LaTeX tools. Ensure pdflatex and bibtex are installed and on PATH."
        )

    # Canonical sequence from the Zenodo README.
    _run([pdflatex, "paper.tex"], cwd=BUNDLE_DIR)
    _run([bibtex, "paper"], cwd=BUNDLE_DIR)
    _run([pdflatex, "paper.tex"], cwd=BUNDLE_DIR)
    _run([pdflatex, "paper.tex"], cwd=BUNDLE_DIR)

    pdf = BUNDLE_DIR / "paper.pdf"
    if pdf.exists():
        print(f"[build] Wrote: {pdf}")
    else:
        print("[build] Completed, but paper.pdf not found")


if __name__ == "__main__":
    main()
