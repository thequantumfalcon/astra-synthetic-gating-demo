"""Compare regenerated artifacts against the reference copies committed in paper/.

verify_astra.py checks that a run is internally consistent: right schema, right seed,
right invariants. It cannot tell whether the run produced the *same* numbers as the
ones the manuscript was written against, because it never looks at a reference.

This script closes that gap. It diffs the freshly generated bundle against the copies
tracked under paper/, so "reproducible" is enforced rather than asserted.

On what "match" means. The reference copies were generated on Windows. Running the
same pinned Python 3.11.9 and numpy 2.4.1 on Linux reproduces them to roughly one
part in 1e16, but not bit-for-bit: numpy's reductions dispatch to different SIMD
paths per architecture, so np.std can land a final ulp away. That is a property of
floating-point hardware, not of this code, and no amount of pinning removes it.

So the enforced contract is agreement to within RELATIVE_TOLERANCE, which is orders
of magnitude tighter than any real change to the pipeline could hide under, and the
script always reports whether the match was bit-identical. Bit-identical
reproduction is expected only on the platform the reference copies came from.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "paper"
BUNDLE_DIR = REPO_ROOT / "astra_submission_bundle"

# Observed cross-platform drift is ~1e-16 relative. A genuine algorithmic change
# moves results far above this bound.
RELATIVE_TOLERANCE = 1e-12


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    return list(reader.fieldnames or []), [dict(row) for row in reader]


def _compare_csv(name: str, errors: list[str]) -> None:
    ref_path, new_path = REFERENCE_DIR / name, BUNDLE_DIR / name
    ref_cols, ref_rows = _read_csv(ref_path)
    new_cols, new_rows = _read_csv(new_path)

    if ref_cols != new_cols:
        _fail(errors, f"{name}: columns differ\n  reference: {ref_cols}\n  regenerated: {new_cols}")
        return
    if len(ref_cols) == 0:
        _fail(errors, f"{name}: reference has no columns")
        return
    if len(ref_rows) != len(new_rows):
        _fail(errors, f"{name}: {len(ref_rows)} reference rows vs {len(new_rows)} regenerated")
        return

    inexact = 0
    worst = 0.0
    worst_cell = ""
    for i, (ref, new) in enumerate(zip(ref_rows, new_rows)):
        for col in ref_cols:
            if ref[col] == new[col]:
                continue
            try:
                ref_val, new_val = float(ref[col]), float(new[col])
            except ValueError:
                _fail(
                    errors,
                    f"{name}: row {i} column {col!r}: "
                    f"reference {ref[col]!r} != regenerated {new[col]!r}",
                )
                continue
            scale = max(abs(ref_val), abs(new_val))
            rel = abs(ref_val - new_val) / scale if scale else abs(ref_val - new_val)
            if rel > RELATIVE_TOLERANCE:
                _fail(
                    errors,
                    f"{name}: row {i} column {col!r} differs by {rel:.3e} relative, "
                    f"above the {RELATIVE_TOLERANCE:.0e} tolerance: "
                    f"reference {ref[col]} != regenerated {new[col]}",
                )
            else:
                inexact += 1
                if rel > worst:
                    worst, worst_cell = rel, f"row {i} column {col!r}"

    if inexact:
        print(
            f"[reference] {name}: {len(ref_rows)} rows match within tolerance; "
            f"{inexact} cell(s) not bit-identical "
            f"(largest relative difference {worst:.3e} at {worst_cell})"
        )
    else:
        print(f"[reference] {name}: {len(ref_rows)} rows bit-identical")


def _compare_text(name: str, errors: list[str]) -> None:
    """Compare line by line, ignoring line-ending style."""
    ref = (REFERENCE_DIR / name).read_text(encoding="utf-8").splitlines()
    new = (BUNDLE_DIR / name).read_text(encoding="utf-8").splitlines()
    if ref == new:
        print(f"[reference] {name}: identical")
        return
    for i, (r, n) in enumerate(zip(ref, new)):
        if r != n:
            _fail(
                errors, f"{name}: line {i + 1} differs\n  reference:   {r!r}\n  regenerated: {n!r}"
            )
            break
    if len(ref) != len(new):
        _fail(errors, f"{name}: {len(ref)} reference lines vs {len(new)} regenerated")


def main() -> None:
    if not BUNDLE_DIR.exists():
        raise SystemExit(f"Bundle not found: {BUNDLE_DIR}. Run: python repro/run_astra.py")

    errors: list[str] = []
    for name in ["mc_summary.csv", "mc_table.tex", "verification_log.txt"]:
        if not (REFERENCE_DIR / name).exists():
            _fail(errors, f"missing reference artifact: paper/{name}")
            continue
        if not (BUNDLE_DIR / name).exists():
            _fail(errors, f"missing regenerated artifact: {BUNDLE_DIR.name}/{name}")
            continue
        if name.endswith(".csv"):
            _compare_csv(name, errors)
        else:
            _compare_text(name, errors)

    if errors:
        print("[reference] FAIL: regenerated artifacts differ from the copies in paper/")
        for err in errors:
            print(f"- {err}")
        print(
            "\nIf this is an intended change, regenerate the reference copies in the "
            "pinned environment and commit them alongside whatever caused the change."
        )
        sys.exit(1)

    print("[reference] PASS: regenerated artifacts match the reference copies in paper/")


if __name__ == "__main__":
    main()
