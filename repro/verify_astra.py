from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "astra_submission_bundle"

EXPECTED_FILES = [
    "verification_log.txt",
    "astra_injection.npz",
    "mc_summary.csv",
    "mc_table.tex",
]


def _load_manifest() -> dict | None:
    p = BUNDLE_DIR / "run_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def _parse_key_values(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _verify_verification_log(path: Path, errors: list[str]) -> None:
    txt = path.read_text(encoding="utf-8", errors="replace")
    kv = _parse_key_values(txt)
    required = {
        "seed",
        "snr_before",
        "snr_after",
        "gated_fraction",
        "threshold",
        "threshold_sigma",
        "duration_s",
        "fs_hz",
    }
    missing = sorted(required - set(kv.keys()))
    if missing:
        errors.append(f"verification_log.txt missing keys: {missing}")

    seed = _try_float(kv.get("seed", ""))
    before = _try_float(kv.get("snr_before", ""))
    after = _try_float(kv.get("snr_after", ""))
    gated = _try_float(kv.get("gated_fraction", ""))

    if seed is None:
        errors.append("verification_log.txt seed is not numeric")
    elif int(seed) != 123:
        errors.append(f"verification_log.txt seed expected 123, got {seed}")

    if before is None or after is None:
        errors.append("verification_log.txt SNR fields are missing or non-numeric")
    elif after >= before:
        errors.append(
            f"verification_log.txt expected snr_after < snr_before, got {after} >= {before}"
        )

    if gated is None:
        errors.append("verification_log.txt gated_fraction is missing or non-numeric")
    elif not (0.0 < gated < 0.1):
        errors.append(f"verification_log.txt gated_fraction out of expected range: {gated}")


def _verify_mc_summary(path: Path, errors: list[str]) -> None:
    txt = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(txt.splitlines())
    fieldnames = list(reader.fieldnames or [])
    rows = [dict(r) for r in reader]

    required_columns = [
        "seed",
        "snr_before",
        "snr_after",
        "gated_fraction",
        "threshold",
        "threshold_sigma",
    ]
    for col in required_columns:
        if col not in fieldnames:
            errors.append(f"mc_summary.csv missing column: {col}")

    if len(rows) != 200:
        errors.append(f"mc_summary.csv expected 200 rows, got {len(rows)}")

    first_seed = _try_float((rows[0].get("seed") or "") if rows else "")
    if first_seed is None or int(first_seed) != 123:
        errors.append("mc_summary.csv first row seed is not 123")


def _verify_npz(path: Path, errors: list[str]) -> None:
    try:
        import numpy as np  # type: ignore

        data = np.load(path)
        required_keys = {"data", "gated_data", "t"}
        missing = sorted(required_keys - set(data.files))
        if missing:
            errors.append(f"astra_injection.npz missing arrays: {missing}")
            return

        h = data["data"]
        h_gated = data["gated_data"]
        t = data["t"]

        if h.shape != h_gated.shape:
            errors.append(
                f"astra_injection.npz shape mismatch: h={h.shape} h_gated={h_gated.shape}"
            )
        if h.ndim != 1 or t.ndim != 1:
            errors.append("astra_injection.npz expected 1D arrays for h and t")
        if h.shape[0] == 0:
            errors.append("astra_injection.npz arrays are empty")
    except Exception as ex:
        errors.append(f"astra_injection.npz could not be parsed: {ex}")


def _verify_tex(path: Path, errors: list[str]) -> None:
    txt = path.read_text(encoding="utf-8", errors="replace")
    checks = [
        "\\begin{tabular}",
        "Peak SNR before gating",
        "Peak SNR after gating",
        "Gated samples fraction",
    ]
    for token in checks:
        if token not in txt:
            errors.append(f"mc_table.tex missing token: {token}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["strict", "portable"],
        default="portable",
        help="Compatibility flag retained for CI scripts. Both modes run the same local consistency checks.",
    )
    args = ap.parse_args()

    mode = args.mode

    if not BUNDLE_DIR.exists():
        raise SystemExit(
            f"Bundle directory not found: {BUNDLE_DIR}. Run: python repro/run_astra.py"
        )

    manifest = _load_manifest()
    if manifest:
        pv = manifest.get("python_version")
        nv = manifest.get("numpy_version")
        print(f"[verify] manifest: python={pv} numpy={nv}")

    if mode == "strict":
        print("[verify] mode: strict (accepted for compatibility; identical checks to portable)")
    else:
        print(f"[verify] mode: {mode}")
    print("[verify] local consistency checks")

    errors: list[str] = []
    for name in EXPECTED_FILES:
        p = BUNDLE_DIR / name
        if not p.exists():
            errors.append(f"missing artifact: {name}")

    if not errors:
        _verify_verification_log(BUNDLE_DIR / "verification_log.txt", errors)
        _verify_mc_summary(BUNDLE_DIR / "mc_summary.csv", errors)
        _verify_npz(BUNDLE_DIR / "astra_injection.npz", errors)
        _verify_tex(BUNDLE_DIR / "mc_table.tex", errors)

    if errors:
        print(f"[verify] FAIL (mode={mode})")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print(f"[verify] PASS (mode={mode}): artifact bundle is internally consistent")
    print(
        "[verify] note: consistency/invariant checks only; no comparison against a stored reference"
    )


if __name__ == "__main__":
    main()
