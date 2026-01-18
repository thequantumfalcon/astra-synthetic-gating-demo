from __future__ import annotations

import argparse
import csv
import re
import hashlib
import json
import platform
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Immutable Zenodo evidence (kept in-repo for public reproducibility)
DEPOSIT_DIR = REPO_ROOT / "zenodo_deposit_2026-01-16_v5_final"
DEPOSIT_ANCILLARY_ZIP = DEPOSIT_DIR / "ASTRA_Ancillary_v5.zip"

BUNDLE_DIR = REPO_ROOT / "astra_submission_bundle"

EXPECTED_FILES = [
    "verification_log.txt",
    "astra_injection.npz",
    "mc_summary.csv",
    "mc_table.tex",
]


def _normalize_newlines(b: bytes) -> bytes:
    # Zenodo goldens were produced on Windows (CRLF). Normalize to LF for
    # cross-platform comparisons.
    return b.replace(b"\r\n", b"\n")


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
    eb = _normalize_newlines(_read_bytes(expected))
    ab = _normalize_newlines(_read_bytes(actual))
    if eb == ab:
        return None
    return Mismatch(
        path=actual.name,
        kind="byte-mismatch",
        details=f"expected_sha256={_sha256_bytes(eb)} actual_sha256={_sha256_bytes(ab)}",
    )


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def _compare_verification_log(expected: Path, actual: Path) -> list[Mismatch]:
    # Portable comparison: key/value with numeric tolerance.
    eb = _normalize_newlines(_read_bytes(expected)).decode("utf-8", errors="replace")
    ab = _normalize_newlines(_read_bytes(actual)).decode("utf-8", errors="replace")

    def parse(txt: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in txt.splitlines():
            if not line.strip():
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
        return out

    e = parse(eb)
    a = parse(ab)
    mismatches: list[Mismatch] = []
    keys = sorted(set(e.keys()) | set(a.keys()))
    for k in keys:
        if k not in e or k not in a:
            mismatches.append(
                Mismatch(
                    path=f"verification_log.txt:{k}",
                    kind="missing-key",
                    details=f"expected_has={k in e} actual_has={k in a}",
                )
            )
            continue

        ef = _try_float(e[k])
        af = _try_float(a[k])
        if ef is not None and af is not None:
            # Tight tolerance; values are deterministic up to floating-point.
            if abs(ef - af) > max(1e-12 * abs(ef), 1e-40):
                mismatches.append(
                    Mismatch(
                        path=f"verification_log.txt:{k}",
                        kind="numeric-diff",
                        details=f"expected={ef} actual={af}",
                    )
                )
        else:
            if e[k] != a[k]:
                mismatches.append(
                    Mismatch(
                        path=f"verification_log.txt:{k}",
                        kind="value-mismatch",
                        details=f"expected={e[k]} actual={a[k]}",
                    )
                )
    return mismatches


def _compare_mc_summary(expected: Path, actual: Path) -> list[Mismatch]:
    # Portable comparison: CSV numeric tolerance.
    et = _normalize_newlines(_read_bytes(expected)).decode("utf-8", errors="replace")
    at = _normalize_newlines(_read_bytes(actual)).decode("utf-8", errors="replace")

    def parse(txt: str) -> tuple[list[str], list[dict[str, str]]]:
        reader = csv.DictReader(txt.splitlines())
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
        return fieldnames, rows

    ef, erows = parse(et)
    af, arows = parse(at)

    mismatches: list[Mismatch] = []
    if ef != af:
        mismatches.append(
            Mismatch(
                path="mc_summary.csv",
                kind="header-mismatch",
                details=f"expected={ef} actual={af}",
            )
        )
        return mismatches

    if len(erows) != len(arows):
        mismatches.append(
            Mismatch(
                path="mc_summary.csv",
                kind="rowcount-mismatch",
                details=f"expected={len(erows)} actual={len(arows)}",
            )
        )
        return mismatches

    for i, (er, ar) in enumerate(zip(erows, arows, strict=True)):
        for k in ef:
            ev = (er.get(k) or "").strip()
            av = (ar.get(k) or "").strip()
            efv = _try_float(ev)
            afv = _try_float(av)
            if efv is not None and afv is not None:
                if abs(efv - afv) > max(1e-12 * abs(efv), 1e-40):
                    mismatches.append(
                        Mismatch(
                            path=f"mc_summary.csv:row{i}:{k}",
                            kind="numeric-diff",
                            details=f"expected={efv} actual={afv}",
                        )
                    )
                    if len(mismatches) > 20:
                        return mismatches
            else:
                if ev != av:
                    mismatches.append(
                        Mismatch(
                            path=f"mc_summary.csv:row{i}:{k}",
                            kind="value-mismatch",
                            details=f"expected={ev} actual={av}",
                        )
                    )
                    if len(mismatches) > 20:
                        return mismatches
    return mismatches


_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _compare_mc_table_tex(expected: Path, actual: Path) -> list[Mismatch]:
    # Portable comparison: compare non-numeric template exactly (after newline normalization),
    # then compare numeric tokens with tight tolerance.
    eb = _normalize_newlines(_read_bytes(expected)).decode("utf-8", errors="replace")
    ab = _normalize_newlines(_read_bytes(actual)).decode("utf-8", errors="replace")

    e_nums = [_try_float(m.group(0)) for m in _NUM_RE.finditer(eb)]
    a_nums = [_try_float(m.group(0)) for m in _NUM_RE.finditer(ab)]
    e_nums_f = [x for x in e_nums if x is not None]
    a_nums_f = [x for x in a_nums if x is not None]

    e_template = _NUM_RE.sub("<NUM>", eb)
    a_template = _NUM_RE.sub("<NUM>", ab)

    mismatches: list[Mismatch] = []
    if e_template != a_template:
        mismatches.append(
            Mismatch(
                path="mc_table.tex",
                kind="template-mismatch",
                details=(
                    f"expected_sha256={_sha256_bytes(e_template.encode('utf-8'))} "
                    f"actual_sha256={_sha256_bytes(a_template.encode('utf-8'))}"
                ),
            )
        )
        return mismatches

    if len(e_nums_f) != len(a_nums_f):
        mismatches.append(
            Mismatch(
                path="mc_table.tex",
                kind="token-count-mismatch",
                details=f"expected={len(e_nums_f)} actual={len(a_nums_f)}",
            )
        )
        return mismatches

    for i, (ev, av) in enumerate(zip(e_nums_f, a_nums_f, strict=True)):
        if abs(ev - av) > max(1e-12 * abs(ev), 1e-40):
            mismatches.append(
                Mismatch(
                    path=f"mc_table.tex:num{i}",
                    kind="numeric-diff",
                    details=f"expected={ev} actual={av}",
                )
            )
            if len(mismatches) > 20:
                break

    return mismatches


def _npz_diff(expected: Path, actual: Path, *, strict_bytes: bool) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    eb = _read_bytes(expected)
    ab = _read_bytes(actual)
    if eb == ab:
        return mismatches
    if strict_bytes:
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

                if strict_bytes or not ok:
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
    if not DEPOSIT_ANCILLARY_ZIP.exists():
        raise SystemExit(
            f"Missing deposit ancillary zip: {DEPOSIT_ANCILLARY_ZIP}"
        )

    out = tmp / "goldens"
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(DEPOSIT_ANCILLARY_ZIP, "r") as z:
        z.extractall(out)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["strict", "portable"],
        default=None,
        help=(
            "strict: require byte-identical outputs; portable: normalize newlines and compare numerics with tolerance"
        ),
    )
    args = ap.parse_args()

    # Default to portable everywhere: modern NumPy / BLAS combos can introduce
    # tiny floating-point and formatting differences that are semantically
    # equivalent but not byte-identical.
    #
    # Additionally, strict byte identity is only meaningful/reliable on the
    # Windows environment used to generate the Zenodo goldens. To avoid CI
    # brittleness, we never run strict mode on non-Windows platforms.
    requested = args.mode
    is_windows = platform.system().lower().startswith("win")
    if requested == "strict" and not is_windows:
        print("[verify] note: strict mode is Windows-only; using portable")
        mode = "portable"
    else:
        mode = requested or "portable"
    strict_bytes = mode == "strict"

    if not BUNDLE_DIR.exists():
        raise SystemExit(
            f"Bundle directory not found: {BUNDLE_DIR}. Run: python repro/run_astra.py"
        )

    manifest = _load_manifest()
    if manifest:
        pv = manifest.get("python_version")
        nv = manifest.get("numpy_version")
        print(f"[verify] manifest: python={pv} numpy={nv}")

    print(f"[verify] mode: {mode}")

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

            if name == "verification_log.txt" and not strict_bytes:
                mismatches.extend(_compare_verification_log(expected, actual))
            elif name == "mc_summary.csv" and not strict_bytes:
                mismatches.extend(_compare_mc_summary(expected, actual))
            elif name == "mc_table.tex" and not strict_bytes:
                mismatches.extend(_compare_mc_table_tex(expected, actual))
            elif name.endswith(".npz"):
                mismatches.extend(_npz_diff(expected, actual, strict_bytes=strict_bytes))
            else:
                mm = _compare_text_exact(expected, actual)
                if mm:
                    mismatches.append(mm)

    if mismatches:
        print(f"[verify] FAIL (mode={mode})")
        for m in mismatches:
            print(f"- {m.path}: {m.kind} ({m.details})")
        raise SystemExit(1)

    print(f"[verify] PASS (mode={mode}): outputs match snapshot goldens")


if __name__ == "__main__":
    main()
