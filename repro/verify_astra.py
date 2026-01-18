from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT_DEPOSIT = (
    REPO_ROOT / "zenodo_snapshot" / "zenodo_deposit_2026-01-16_v5_final"
)
SNAPSHOT_ANCILLARY_ZIP = SNAPSHOT_DEPOSIT / "ASTRA_Ancillary_v5.zip"

BUNDLE_DIR = REPO_ROOT / "astra_submission_bundle"

EXPECTED_FILES = [
    "verification_log.txt",
    "astra_injection.npz",
    "mc_summary.csv",
    "mc_table.tex",
]


@dataclass(frozen=True)
class Mismatch:
    path: str
    kind: str
    details: str


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _load_manifest() -> dict | None:
    p = BUNDLE_DIR / "run_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compare_text_exact(expected: Path, actual: Path) -> Mismatch | None:
    eb = _read_bytes(expected)
    ab = _read_bytes(actual)
    if eb == ab:
        return None
    return Mismatch(
        path=actual.name,
        kind="byte-mismatch",
        details=f"expected_sha256={_sha256_bytes(eb)} actual_sha256={_sha256_bytes(ab)}",
    )


def _npz_diff(expected: Path, actual: Path) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    eb = _read_bytes(expected)
    ab = _read_bytes(actual)
    if eb == ab:
        return mismatches

    mismatches.append(
        Mismatch(
            path=actual.name,
            kind="byte-mismatch",
            details=f"expected_sha256={_sha256_bytes(eb)} actual_sha256={_sha256_bytes(ab)}",
        )
    )

    # Proof-grade diagnostics (do not treat as pass): numeric diffs + rounded-hash
    try:
        import numpy as np  # type: ignore

        e = np.load(expected)
        a = np.load(actual)
        keys = sorted(set(e.files) | set(a.files))
        for k in keys:
            if k not in e.files or k not in a.files:
                mismatches.append(
                    Mismatch(
                        path=f"{actual.name}:{k}",
                        kind="missing-key",
                        details=f"expected_has={k in e.files} actual_has={k in a.files}",
                    )
                )
                continue

            ea = e[k]
            aa = a[k]
            if ea.shape != aa.shape or ea.dtype != aa.dtype:
                mismatches.append(
                    Mismatch(
                        path=f"{actual.name}:{k}",
                        kind="shape/dtype-mismatch",
                        details=f"expected={ea.shape}/{ea.dtype} actual={aa.shape}/{aa.dtype}",
                    )
                )
                continue

            if not np.array_equal(ea, aa):
                max_abs = float(np.max(np.abs(ea - aa)))
                ok = bool(np.allclose(ea, aa, rtol=1e-12, atol=0.0))

                def rounded_hash(x: np.ndarray) -> str:
                    xr = np.round(x.astype(np.float64), 12)
                    return hashlib.sha256(xr.tobytes()).hexdigest()

                mismatches.append(
                    Mismatch(
                        path=f"{actual.name}:{k}",
                        kind="numeric-diff",
                        details=(
                            f"allclose_rtol1e-12={ok} max_abs_diff={max_abs} "
                            f"expected_rounded_sha256={rounded_hash(ea)} actual_rounded_sha256={rounded_hash(aa)}"
                        ),
                    )
                )
    except Exception as ex:
        mismatches.append(
            Mismatch(
                path=actual.name,
                kind="npz-diff-error",
                details=str(ex),
            )
        )

    return mismatches


def _extract_goldens(tmp: Path) -> Path:
    if not SNAPSHOT_ANCILLARY_ZIP.exists():
        raise SystemExit(
            f"Missing snapshot ancillary zip: {SNAPSHOT_ANCILLARY_ZIP} (did you copy zenodo_snapshot?)"
        )

    out = tmp / "goldens"
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(SNAPSHOT_ANCILLARY_ZIP, "r") as z:
        z.extractall(out)

    return out


def main() -> None:
    if not BUNDLE_DIR.exists():
        raise SystemExit(
            f"Bundle directory not found: {BUNDLE_DIR}. Run: python repro/run_astra.py"
        )

    manifest = _load_manifest()
    if manifest:
        pv = manifest.get("python_version")
        nv = manifest.get("numpy_version")
        print(f"[verify] manifest: python={pv} numpy={nv}")

    mismatches: list[Mismatch] = []

    with tempfile.TemporaryDirectory(prefix="astra_goldens_") as td:
        golden_root = _extract_goldens(Path(td))

        for name in EXPECTED_FILES:
            expected = golden_root / name
            actual = BUNDLE_DIR / name

            if not actual.exists():
                mismatches.append(
                    Mismatch(path=name, kind="missing-actual", details=str(actual))
                )
                continue
            if not expected.exists():
                mismatches.append(
                    Mismatch(path=name, kind="missing-expected", details=str(expected))
                )
                continue

            if name.endswith(".npz"):
                mismatches.extend(_npz_diff(expected, actual))
            else:
                mm = _compare_text_exact(expected, actual)
                if mm:
                    mismatches.append(mm)

    if mismatches:
        print("[verify] FAIL")
        for m in mismatches:
            print(f"- {m.path}: {m.kind} ({m.details})")
        raise SystemExit(1)

    print("[verify] PASS: outputs match snapshot goldens")


if __name__ == "__main__":
    main()
