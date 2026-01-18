"""Project ASTRA: Archival/Open-Data Verification Protocol

Goal
----
Provide a reproducible *protocol* for testing the gating-suppression hypothesis on
real gravitational-wave strain, when/if open data is available.

Important scope note
--------------------
- This script does NOT claim access to proprietary data.
- If open data cannot be fetched, it runs in synthetic mode and produces artifacts
  that demonstrate the analysis procedure (not a detection claim).

Usage (examples)
----------------
# Try open data (will only work for public segments)
python astra_real_verify.py --gps 1448668818 --detector H1 --try-open-data

# Synthetic-only protocol (always available)
python astra_real_verify.py --gps 1448668818 --detector H1

Outputs
-------
Writes a log file next to the script by default:
- astra_submission_bundle/real_verify_log.txt

Optionally writes a plot if matplotlib is available.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TemplateParams:
    h0: float = 3.46e-21
    f_hz: float = 200.0
    tau_s: float = 0.3
    duration_s: float = 1.0


def tukey_window(n: int, alpha: float = 0.25) -> np.ndarray:
    """Minimal Tukey window implementation (no SciPy dependency)."""
    if n <= 1:
        return np.ones((n,), dtype=np.float64)
    if alpha <= 0.0:
        return np.ones((n,), dtype=np.float64)
    if alpha >= 1.0:
        # Hann
        x = np.linspace(0.0, 1.0, n, dtype=np.float64)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * x))

    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = np.ones((n,), dtype=np.float64)
    edge = alpha / 2.0

    left = x < edge
    w[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / alpha - 1.0)))

    right = x >= (1.0 - edge)
    w[right] = 0.5 * (
        1.0 + np.cos(np.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0))
    )

    return w


def make_template(dt: float, p: TemplateParams) -> np.ndarray:
    n = max(1, int(round(p.duration_s / dt)))
    t = np.arange(n, dtype=np.float64) * dt
    return p.h0 * np.exp(-t / p.tau_s) * np.sin(2.0 * np.pi * p.f_hz * t)


def normalized_xcorr_max(x: np.ndarray, y: np.ndarray, eps: float = 1e-30) -> float:
    """Return max absolute correlation normalized by std(x)*||y||.

    This is a *proxy* score suitable for protocol comparisons (gated vs ungated).
    It is not a substitute for a calibrated matched-filter SNR.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_std = float(np.std(x))
    y_norm = float(np.linalg.norm(y))

    # Full correlation; for 64s @ 4096 Hz this is large but still manageable.
    c = np.correlate(x, y, mode="valid")
    score = float(np.max(np.abs(c)) / ((x_std + eps) * (y_norm + eps)))
    return score


def apply_energy_gate(
    strain: np.ndarray,
    gate_k: float,
    tukey_alpha: float,
) -> Tuple[np.ndarray, float]:
    """Simple amplitude/energy gating (illustrative).

    Uses threshold = gate_k * median(strain^2).
    Zeros samples beyond threshold; optionally applies a short Tukey taper to
    soften edges.
    """
    energy = strain * strain
    thr = float(gate_k) * float(np.median(energy))

    gated = strain.copy()
    mask = energy > thr
    if np.any(mask):
        gated[mask] = 0.0

        if tukey_alpha > 0.0:
            # Apply a short taper around each masked region (very simple).
            # This is NOT a replica of any specific production pipeline.
            pad = 128
            w = tukey_window(2 * pad + 1, alpha=tukey_alpha)
            idx = np.where(mask)[0]
            for i in idx:
                a = max(0, i - pad)
                b = min(len(gated) - 1, i + pad)
                ww = w[(a - (i - pad)) : (2 * pad + 1 - ((i + pad) - b))]
                gated[a : b + 1] *= ww

    return gated, thr


def fetch_open_data(
    gps: int,
    detector: str,
    duration_s: int,
    prefer_fs_hz: int,
) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    """Try to fetch public GWOSC open data via gwpy.

    Returns (strain, dt, note).
    """
    try:
        from gwpy.timeseries import TimeSeries  # type: ignore

        start = gps - duration_s // 2
        end = gps + duration_s // 2
        ts = TimeSeries.fetch_open_data(detector, start, end, verbose=True)
        # Resample for stable dt (optional); if already at prefer_fs, this is cheap.
        if prefer_fs_hz and int(round(1.0 / ts.dt.value)) != int(prefer_fs_hz):
            ts = ts.resample(prefer_fs_hz)
        return ts.value.astype(np.float64), float(ts.dt.value), "open_data"
    except Exception as e:
        return None, None, f"open_data_unavailable: {e}"


def synthetic_strain(
    fs_hz: int, duration_s: int, noise_std: float, seed: int
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = fs_hz * duration_s
    return rng.normal(0.0, noise_std, size=(n,)).astype(np.float64), 1.0 / float(fs_hz)


def write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps", type=int, default=1448668818)
    ap.add_argument("--detector", type=str, default="H1")
    ap.add_argument("--duration", type=int, default=64)
    ap.add_argument("--fs", type=int, default=4096)
    ap.add_argument("--try-open-data", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--noise-std", type=float, default=1e-23)
    ap.add_argument("--gate-k", type=float, default=25.0)
    ap.add_argument("--tukey-alpha", type=float, default=0.25)
    ap.add_argument(
        "--out", type=str, default="astra_submission_bundle/real_verify_log.txt"
    )
    args = ap.parse_args(argv)

    # 1) Data acquisition
    mode = "synthetic"
    note = ""
    strain: np.ndarray
    dt: float

    if args.try_open_data:
        real, real_dt, note = fetch_open_data(
            args.gps, args.detector, args.duration, args.fs
        )
        if real is not None and real_dt is not None:
            strain, dt = real, real_dt
            mode = "open_data"
        else:
            strain, dt = synthetic_strain(
                args.fs, args.duration, args.noise_std, args.seed
            )
            mode = "synthetic_fallback"
    else:
        strain, dt = synthetic_strain(args.fs, args.duration, args.noise_std, args.seed)

    # 2) Template
    template = make_template(dt, TemplateParams())

    # 3) Gating (illustrative)
    gated, thr = apply_energy_gate(
        strain, gate_k=args.gate_k, tukey_alpha=args.tukey_alpha
    )

    # 4) Scores (proxy)
    score_ungated = normalized_xcorr_max(strain, template)
    score_gated = normalized_xcorr_max(gated, template)

    # 5) Report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("PROJECT ASTRA — REAL/OPEN DATA VERIFICATION PROTOCOL")
    lines.append(f"gps: {args.gps}")
    lines.append(f"detector: {args.detector}")
    lines.append(f"duration_s: {args.duration}")
    lines.append(f"fs_hz_target: {args.fs}")
    lines.append(f"mode: {mode}")
    if note:
        lines.append(f"note: {note}")
    lines.append(f"dt_s: {dt}")
    lines.append(f"gate_k: {args.gate_k}")
    lines.append(f"gate_threshold_energy: {thr}")
    lines.append(f"tukey_alpha: {args.tukey_alpha}")
    lines.append(f"template_h0: {TemplateParams.h0}")
    lines.append(f"template_f_hz: {TemplateParams.f_hz}")
    lines.append(f"template_tau_s: {TemplateParams.tau_s}")
    lines.append(f"score_ungated_proxy: {score_ungated}")
    lines.append(f"score_gated_proxy: {score_gated}")
    lines.append(
        f"score_ratio_ungated_over_gated: {score_ungated / (score_gated + 1e-30)}"
    )

    write_log(out_path, lines)
    print(f"[ASTRA] Wrote: {out_path.resolve()}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # type: ignore

        t = np.arange(len(strain), dtype=np.float64) * dt
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, strain, lw=0.5, label="ungated")
        ax.plot(t, gated, lw=0.5, label="gated")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("strain (arb)")
        ax.legend(loc="upper right")
        ax.set_title(
            f"ASTRA protocol: mode={mode}, gps={args.gps}, det={args.detector}"
        )
        png = out_path.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        print(f"[ASTRA] Wrote: {png.resolve()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
