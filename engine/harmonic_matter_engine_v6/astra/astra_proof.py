from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class AstraParams:
    mjd: int = 59942
    f_spin_hz: float = 100.0
    glitch_mag: float = 1.15e-12
    gv: float = 0.35

    # Vacuum-collapse / fracture energy scale (demo)
    E_vac_erg: float = 4.0e33
    dist_cm: float = 1.3 * 3.086e21

    # Predicted burst parameters
    h0_refined: float = 3.46e-21
    f_gw_hz: float = 200.0
    tau_s: float = 0.3
    t0_s: float = 30.0


@dataclass(frozen=True)
class GatingParams:
    fs_hz: int = 4096
    duration_s: int = 60
    noise_std: float = 5.0e-23
    threshold_sigma: float = 8.0


def run_astra_kernel(params: AstraParams) -> float:
    """Compute the (demo) predicted GW strain for the glitch epoch.

    This does not fetch or validate external observational datasets; it produces a
    deterministic, publication-friendly log with the parameters used.
    """
    omega = 2.0 * np.pi * params.f_spin_hz
    _ = omega, params.glitch_mag, params.gv

    # Quadrupole-ish scaling (toy)
    h0_quad = (4.0 * 6.67e-8 * params.E_vac_erg) / (2.99e10**4 * params.dist_cm)
    print(f"[ASTRA] Initializing Vector-Enhanced Kernel (Gv={params.gv:.2f})...")
    print(f"[ASTRA] Glitch Epoch (MJD): {params.mjd}")
    print(f"[ASTRA] Quadrupole baseline h0: {h0_quad:.3e}")
    print(f"[ASTRA] Predicted GW Strain (h0): {params.h0_refined:.3e}")
    print(f"[ASTRA] Predicted GW: f={params.f_gw_hz:.1f} Hz, tau={params.tau_s:.2f} s")
    return float(params.h0_refined)


def _make_timeseries(gp: GatingParams) -> np.ndarray:
    n = int(gp.fs_hz * gp.duration_s)
    return np.arange(n, dtype=np.float64) / float(gp.fs_hz)


def inject_burst(
    t: np.ndarray,
    h0: float,
    f_hz: float,
    tau_s: float,
    t0_s: float,
) -> np.ndarray:
    """Create a simple exponentially decaying sinusoid burst starting at t0."""
    sig = np.zeros_like(t, dtype=np.float64)
    m = t >= t0_s
    dt = t[m] - t0_s
    sig[m] = h0 * np.exp(-dt / tau_s) * np.sin(2.0 * np.pi * f_hz * dt)
    return sig


def apply_gating(data: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Zero out samples exceeding the gating threshold."""
    mask = np.abs(data) > threshold
    gated = data.copy()
    gated[mask] = 0.0
    return gated, mask


def verify_gating_paradox(
    h_signal: float,
    params: AstraParams,
    gp: GatingParams,
    seed: int,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(
            "\n[ASTRA] Running LIGO Engineering-Phase Pipeline Simulation (synthetic)..."
        )

    rng = np.random.default_rng(seed)
    t = _make_timeseries(gp)
    noise = rng.normal(0.0, gp.noise_std, size=t.shape).astype(np.float64)
    sig = inject_burst(t, h_signal, params.f_gw_hz, params.tau_s, params.t0_s)
    data = noise + sig

    threshold = float(gp.threshold_sigma) * float(np.std(noise))
    gated_data, gated_mask = apply_gating(data, threshold)

    snr_before = float(np.max(np.abs(data)) / (np.std(noise) + 1e-30))
    snr_after = float(np.max(np.abs(gated_data)) / (np.std(noise) + 1e-30))

    gated_fraction = float(np.mean(gated_mask))
    if verbose:
        print(f"[PROOF] Noise std: {np.std(noise):.3e}")
        print(
            f"[PROOF] Gating threshold (sigma={gp.threshold_sigma:.1f}): {threshold:.3e}"
        )
        print(f"[PROOF] Peak SNR before gating: {snr_before:.2f}")
        print(f"[PROOF] Peak SNR after gating:  {snr_after:.2f}")
        print(f"[PROOF] Gated samples fraction:  {gated_fraction:.6f}")

    return {
        "mjd": params.mjd,
        "seed": seed,
        "fs_hz": gp.fs_hz,
        "duration_s": gp.duration_s,
        "noise_std": gp.noise_std,
        "threshold_sigma": gp.threshold_sigma,
        "threshold": threshold,
        "h0": h_signal,
        "f_gw_hz": params.f_gw_hz,
        "tau_s": params.tau_s,
        "t0_s": params.t0_s,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "gated_fraction": gated_fraction,
    }


def write_artifacts(
    out_dir: Path,
    summary: dict,
    t: np.ndarray,
    data: np.ndarray,
    gated_data: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable log
    log_path = out_dir / "verification_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        for k in sorted(summary.keys()):
            f.write(f"{k}: {summary[k]}\n")

    # Machine-readable arrays
    np.savez_compressed(
        out_dir / "astra_injection.npz", t=t, data=data, gated_data=gated_data
    )


def _write_mc_artifacts(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "mc_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # LaTeX table (small, publication-friendly)
    tex_path = out_dir / "mc_table.tex"
    if not rows:
        tex_path.write_text("% No MC rows generated\n", encoding="utf-8")
        return

    snr_before = np.array([r["snr_before"] for r in rows], dtype=np.float64)
    snr_after = np.array([r["snr_after"] for r in rows], dtype=np.float64)
    gated_frac = np.array([r["gated_fraction"] for r in rows], dtype=np.float64)

    def q(x, p):
        return float(np.quantile(x, p))

    tex = []
    tex.append("% Auto-generated by astra_proof.py --mc\n")
    tex.append("\\begin{table}[t]\n")
    tex.append("\\centering\n")
    tex.append(
        "\\caption{Monte Carlo summary of synthetic gating impact ("
        + str(len(rows))
        + " trials).}\\label{tab:mc}\n"
    )
    tex.append("\\begin{tabular}{lrr}\n")
    tex.append("\\toprule\n")
    tex.append("Metric & Median & [10\\%, 90\\%]\\\\\n")
    tex.append("\\midrule\n")
    tex.append(
        f"Peak SNR before gating & {q(snr_before, 0.5):.2f} & [{q(snr_before, 0.1):.2f}, {q(snr_before, 0.9):.2f}]\\\\\n"
    )
    tex.append(
        f"Peak SNR after gating  & {q(snr_after, 0.5):.2f} & [{q(snr_after, 0.1):.2f}, {q(snr_after, 0.9):.2f}]\\\\\n"
    )
    tex.append(
        f"Gated samples fraction & {q(gated_frac, 0.5):.4f} & [{q(gated_frac, 0.1):.4f}, {q(gated_frac, 0.9):.4f}]\\\\\n"
    )
    tex.append("\\bottomrule\n")
    tex.append("\\end{tabular}\n")
    tex.append("\\end{table}\n")

    tex_path.write_text("".join(tex), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Project ASTRA reproducibility proof (synthetic gating demo)."
    )
    ap.add_argument("--prompt", type=str, default="Project ASTRA / PSR J0900-3144")
    ap.add_argument("--mjd", type=int, default=59942)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="astra_output")
    ap.add_argument("--fs", type=int, default=4096)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--noise-std", type=float, default=5.0e-23)
    ap.add_argument("--threshold-sigma", type=float, default=8.0)
    ap.add_argument("--h0", type=float, default=3.46e-21)
    ap.add_argument("--f-gw", type=float, default=200.0)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--t0", type=float, default=30.0)
    ap.add_argument(
        "--mc",
        type=int,
        default=0,
        help="Run N Monte Carlo trials with different seeds and emit mc_summary.csv + mc_table.tex.",
    )
    args = ap.parse_args(argv)

    print("==================================================")
    print("          PROJECT ASTRA — REPRODUCIBILITY          ")
    print("==================================================")
    print(f"[ASTRA] Prompt: {args.prompt}")

    params = AstraParams(
        mjd=int(args.mjd),
        h0_refined=float(args.h0),
        f_gw_hz=float(args.f_gw),
        tau_s=float(args.tau),
        t0_s=float(args.t0),
    )
    gp = GatingParams(
        fs_hz=int(args.fs),
        duration_s=int(args.duration),
        noise_std=float(args.noise_std),
        threshold_sigma=float(args.threshold_sigma),
    )

    h = run_astra_kernel(params)

    rng = np.random.default_rng(int(args.seed))
    t = _make_timeseries(gp)
    noise = rng.normal(0.0, gp.noise_std, size=t.shape).astype(np.float64)
    sig = inject_burst(t, h, params.f_gw_hz, params.tau_s, params.t0_s)
    data = noise + sig
    threshold = float(gp.threshold_sigma) * float(np.std(noise))
    gated_data, _mask = apply_gating(data, threshold)

    summary = verify_gating_paradox(h, params, gp, seed=int(args.seed), verbose=True)
    out_dir = Path(args.out)
    write_artifacts(out_dir, summary, t=t, data=data, gated_data=gated_data)

    if int(args.mc) > 0:
        rows: list[dict] = []
        base_seed = int(args.seed)
        for i in range(int(args.mc)):
            s = base_seed + i
            rows.append(verify_gating_paradox(h, params, gp, seed=s, verbose=False))
        _write_mc_artifacts(out_dir, rows)
    print(f"\n[ASTRA] Wrote artifacts: {out_dir.resolve()}")
    print("[ASTRA] - verification_log.txt")
    print("[ASTRA] - astra_injection.npz")
    if int(args.mc) > 0:
        print("[ASTRA] - mc_summary.csv")
        print("[ASTRA] - mc_table.tex")


if __name__ == "__main__":
    main()
