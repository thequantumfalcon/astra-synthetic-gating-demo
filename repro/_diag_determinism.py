"""Temporary diagnostic: is the divergence a permutation, or different values?"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from astra.astra_proof import AstraParams, GatingParams, inject_burst, population_std  # noqa: E402


def raw(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def sortedh(a: np.ndarray) -> str:
    return hashlib.sha256(np.sort(a).tobytes()).hexdigest()[:16]


def main() -> None:
    p, gp = AstraParams(), GatingParams()
    print(f"machine  : {platform.machine()}  numpy {np.__version__}")

    n = int(gp.fs_hz * gp.duration_s)
    t = np.arange(n, dtype=np.float64) / float(gp.fs_hz)
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, gp.noise_std, size=t.shape).astype(np.float64)
    sig = inject_burst(t, p.h0, p.f_gw_hz, p.tau_s, p.t0_s)

    print(f"noise raw   : {raw(noise)}")
    print(f"noise sorted: {sortedh(noise)}   <- same => permutation only")
    print(f"sig   raw   : {raw(sig)}")
    print(f"sig   sorted: {sortedh(sig)}")
    print(f"noise[0]    : {float(noise[0]).hex()}")
    print(f"noise[1]    : {float(noise[1]).hex()}")
    print(f"sig[122880] : {float(sig[122880]).hex()}")
    print(f"std(fsum)   : {population_std(noise).hex()}")


if __name__ == "__main__":
    main()
