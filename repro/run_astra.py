from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "engine"
PAPER_DIR = REPO_ROOT / "paper"

ASTRA_OUTPUT_DIR = REPO_ROOT / "astra_output"
BUNDLE_DIR = REPO_ROOT / "astra_submission_bundle"

PINNED_PYTHON = "3.11.9"
SEED = 123
MC_TRIALS = 200


def _deterministic_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Use the in-repo engine package without requiring installation.
    existing = env.get("PYTHONPATH", "")
    engine_path = str(ENGINE_DIR)
    env["PYTHONPATH"] = engine_path + (os.pathsep + existing if existing else "")
    return env


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[repro] $", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=_deterministic_env())


def _write_manifest(out_dir: Path) -> None:
    try:
        import numpy as np  # type: ignore

        numpy_version = np.__version__
    except Exception:
        numpy_version = None

    manifest = {
        "pinned_python": PINNED_PYTHON,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "mc_trials": MC_TRIALS,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "numpy_version": numpy_version,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    if not PAPER_DIR.exists():
        raise SystemExit(f"paper directory not found: {PAPER_DIR}")
    if not ENGINE_DIR.exists():
        raise SystemExit(f"engine directory not found: {ENGINE_DIR}")

    # Fresh outputs (avoid stale files affecting verification).
    shutil.rmtree(ASTRA_OUTPUT_DIR, ignore_errors=True)
    shutil.rmtree(BUNDLE_DIR, ignore_errors=True)

    # 1) Create non-MC outputs.
    _run(
        [
            sys.executable,
            "-m",
            "harmonic_matter_engine_v6.astra",
            "--out",
            str(ASTRA_OUTPUT_DIR),
            "--seed",
            str(SEED),
        ]
    )

    # 2) Assemble bundle for paper build (copy LaTeX tree first).
    shutil.copytree(PAPER_DIR, BUNDLE_DIR)

    # 3) Run MC inside the bundle dir (writes verification_log.txt, astra_injection.npz, mc_*).
    _run(
        [
            sys.executable,
            "-m",
            "harmonic_matter_engine_v6.astra",
            "--out",
            str(BUNDLE_DIR),
            "--seed",
            str(SEED),
            "--mc",
            str(MC_TRIALS),
        ]
    )

    _write_manifest(BUNDLE_DIR)
    print(f"[repro] Wrote bundle: {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
