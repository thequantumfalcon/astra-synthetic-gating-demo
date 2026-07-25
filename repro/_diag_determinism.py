"""Temporary diagnostic: print exact bit patterns of every intermediate.

Run on each CI platform to find which step of the pipeline is not
bit-reproducible across architectures. Not part of the shipped tooling.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from astra.astra_proof import AstraParams, GatingParams, inject_burst  # noqa: E402


def h(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def main() -> None:
    p, gp = AstraParams(), GatingParams()
    print(f"platform : {platform.platform()}")
    print(f"machine  : {platform.machine()}")
    print(f"numpy    : {np.__version__}")
    print(f"python   : {platform.python_version()}")

    n = int(gp.fs_hz * gp.duration_s)
    t = np.arange(n, dtype=np.float64) / float(gp.fs_hz)
    print(f"t        : {h(t)}")

    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, gp.noise_std, size=t.shape).astype(np.float64)
    print(f"noise    : {h(noise)}")

    sig = inject_burst(t, p.h0, p.f_gw_hz, p.tau_s, p.t0_s)
    print(f"sig      : {h(sig)}   <- uses np.exp and np.sin")

    data = noise + sig
    print(f"data     : {h(data)}")

    std = float(np.std(noise))
    print(f"std      : {std.hex()}")

    threshold = float(gp.threshold_sigma) * std
    print(f"threshold: {threshold.hex()}")

    gated, mask = data.copy(), np.abs(data) > threshold
    gated[mask] = 0.0
    print(f"gated    : {h(gated)}")
    print(f"n_gated  : {int(mask.sum())}")

    snr_before = float(np.max(np.abs(data)) / (std + 1e-30))
    snr_after = float(np.max(np.abs(gated)) / (std + 1e-30))
    print(f"snr_before: {snr_before.hex()}  {snr_before!r}")
    print(f"snr_after : {snr_after.hex()}  {snr_after!r}")


if __name__ == "__main__":
    main()
