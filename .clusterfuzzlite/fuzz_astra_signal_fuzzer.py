#!/usr/bin/env python3

from __future__ import annotations

import math
import sys

import atheris
import numpy as np

with atheris.instrument_imports():
    from harmonic_matter_engine_v6.astra.astra_proof import (
        AstraParams,
        GatingParams,
        _make_timeseries,
        apply_gating,
        inject_burst,
        verify_gating_paradox,
    )


def _bounded_float(value: float, lower: float, upper: float, fallback: float) -> float:
    if not math.isfinite(value):
        return fallback
    return min(max(value, lower), upper)


def TestOneInput(data: bytes) -> None:
    provider = atheris.FuzzedDataProvider(data)

    fs_hz = provider.ConsumeIntInRange(8, 1024)
    duration_s = provider.ConsumeIntInRange(1, 8)
    noise_std = _bounded_float(provider.ConsumeFloat(), 0.0, 1.0e-18, 5.0e-23)
    threshold_sigma = _bounded_float(provider.ConsumeFloat(), 0.0, 25.0, 8.0)

    gp = GatingParams(
        fs_hz=fs_hz,
        duration_s=duration_s,
        noise_std=noise_std,
        threshold_sigma=threshold_sigma,
    )
    t = _make_timeseries(gp)

    h0 = _bounded_float(provider.ConsumeFloat(), 0.0, 1.0e-18, 1.0e-21)
    f_hz = _bounded_float(provider.ConsumeFloat(), 0.0, fs_hz / 2.0, 200.0)
    tau_s = _bounded_float(provider.ConsumeFloat(), 1.0e-4, 5.0, 0.3)
    t0_s = _bounded_float(provider.ConsumeFloat(), 0.0, duration_s, duration_s / 2.0)

    signal = inject_burst(t, h0=h0, f_hz=f_hz, tau_s=tau_s, t0_s=t0_s)
    threshold = _bounded_float(provider.ConsumeFloat(), 0.0, 10.0, 1.0)
    gated_signal, mask = apply_gating(signal, threshold=threshold)

    if signal.shape != gated_signal.shape or mask.shape != signal.shape:
        raise RuntimeError("Gating changed array shapes")

    params = AstraParams(
        mjd=provider.ConsumeIntInRange(50000, 70000),
        f_spin_hz=_bounded_float(provider.ConsumeFloat(), 1.0, 1000.0, 100.0),
        glitch_mag=_bounded_float(provider.ConsumeFloat(), 0.0, 1.0e-8, 1.15e-12),
        gv=_bounded_float(provider.ConsumeFloat(), 0.0, 2.0, 0.35),
        E_vac_erg=_bounded_float(provider.ConsumeFloat(), 1.0, 1.0e40, 4.0e33),
        dist_cm=_bounded_float(provider.ConsumeFloat(), 1.0, 1.0e30, 1.3 * 3.086e21),
        h0_refined=h0,
        f_gw_hz=f_hz,
        tau_s=tau_s,
        t0_s=t0_s,
    )
    summary = verify_gating_paradox(
        h_signal=h0,
        params=params,
        gp=gp,
        seed=provider.ConsumeIntInRange(0, 2**31 - 1),
        verbose=False,
    )

    if not np.isfinite(summary["snr_before"]):
        raise RuntimeError("Non-finite snr_before")
    if not np.isfinite(summary["snr_after"]):
        raise RuntimeError("Non-finite snr_after")
    if not 0.0 <= summary["gated_fraction"] <= 1.0:
        raise RuntimeError("gated_fraction out of range")


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()