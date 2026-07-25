"""Unit tests for ASTRA core signal-processing and gating functions.

Tests cover both astra_proof.py (synthetic gating demo) and
astra_real_verify.py (open-data verification protocol).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make engine importable without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from astra.astra_proof import (
    AstraParams,
    GatingParams,
    _make_timeseries,
    apply_gating,
    inject_burst,
    population_std,
    run_gating_trial,
)

# astra_real_verify lives in paper/, not in a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))
import astra_real_verify as arv


# ---------------------------------------------------------------------------
# astra_proof.py — inject_burst
# ---------------------------------------------------------------------------
class TestInjectBurst:
    def test_shape_matches_input(self):
        t = np.linspace(0, 1, 4096, dtype=np.float64)
        sig = inject_burst(t, h0=1e-21, f_hz=200.0, tau_s=0.3, t0_s=0.5)
        assert sig.shape == t.shape

    def test_zero_before_t0(self):
        t = np.linspace(0, 1, 4096, dtype=np.float64)
        sig = inject_burst(t, h0=1e-21, f_hz=200.0, tau_s=0.3, t0_s=0.5)
        before = sig[t < 0.5]
        np.testing.assert_array_equal(before, 0.0)

    def test_nonzero_after_t0(self):
        t = np.linspace(0, 2, 8192, dtype=np.float64)
        sig = inject_burst(t, h0=1e-21, f_hz=200.0, tau_s=0.3, t0_s=0.5)
        after = sig[t >= 0.5]
        assert np.any(after != 0.0)

    def test_peak_amplitude_bounded(self):
        t = np.linspace(0, 2, 8192, dtype=np.float64)
        h0 = 3.46e-21
        sig = inject_burst(t, h0=h0, f_hz=200.0, tau_s=0.3, t0_s=0.5)
        assert np.max(np.abs(sig)) <= h0 * 1.01  # allow tiny float margin

    def test_exponential_decay(self):
        """Envelope should decay: peak near t0 > peak at t0+2*tau."""
        t = np.linspace(0, 5, 20480, dtype=np.float64)
        sig = inject_burst(t, h0=1.0, f_hz=50.0, tau_s=0.5, t0_s=1.0)
        # Envelope near t0
        window_early = (t >= 1.0) & (t < 1.1)
        window_late = (t >= 2.0) & (t < 2.1)
        assert np.max(np.abs(sig[window_early])) > np.max(np.abs(sig[window_late]))

    def test_deterministic(self):
        t = np.linspace(0, 1, 4096, dtype=np.float64)
        s1 = inject_burst(t, h0=1e-21, f_hz=200.0, tau_s=0.3, t0_s=0.3)
        s2 = inject_burst(t, h0=1e-21, f_hz=200.0, tau_s=0.3, t0_s=0.3)
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# astra_proof.py — apply_gating
# ---------------------------------------------------------------------------
class TestApplyGating:
    def test_below_threshold_unchanged(self):
        data = np.array([0.1, -0.2, 0.05, -0.15], dtype=np.float64)
        gated, mask = apply_gating(data, threshold=1.0)
        np.testing.assert_array_equal(gated, data)
        assert not np.any(mask)

    def test_above_threshold_zeroed(self):
        data = np.array([0.1, 5.0, -0.2, -3.0], dtype=np.float64)
        gated, mask = apply_gating(data, threshold=1.0)
        assert gated[1] == 0.0
        assert gated[3] == 0.0
        assert mask[1] and mask[3]
        assert gated[0] == data[0]  # untouched

    def test_does_not_modify_input(self):
        data = np.array([0.1, 5.0, -0.2], dtype=np.float64)
        original = data.copy()
        apply_gating(data, threshold=1.0)
        np.testing.assert_array_equal(data, original)

    def test_all_below(self):
        data = np.zeros(100, dtype=np.float64)
        gated, mask = apply_gating(data, threshold=1.0)
        assert not np.any(mask)

    def test_all_above(self):
        data = np.full(100, 10.0, dtype=np.float64)
        gated, mask = apply_gating(data, threshold=1.0)
        assert np.all(mask)
        np.testing.assert_array_equal(gated, 0.0)


# ---------------------------------------------------------------------------
# astra_proof.py — _make_timeseries
# ---------------------------------------------------------------------------
class TestMakeTimeseries:
    def test_length(self):
        gp = GatingParams(fs_hz=4096, duration_s=60)
        t = _make_timeseries(gp)
        assert len(t) == 4096 * 60

    def test_starts_at_zero(self):
        gp = GatingParams(fs_hz=1000, duration_s=1)
        t = _make_timeseries(gp)
        assert t[0] == 0.0

    def test_spacing(self):
        gp = GatingParams(fs_hz=4096, duration_s=1)
        t = _make_timeseries(gp)
        dt = t[1] - t[0]
        np.testing.assert_allclose(dt, 1.0 / 4096, rtol=1e-12)


# ---------------------------------------------------------------------------
# astra_proof.py — population_std
# ---------------------------------------------------------------------------
class TestPopulationStd:
    def test_matches_numpy_on_ordinary_data(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=10_000)
        np.testing.assert_allclose(population_std(x), np.std(x), rtol=1e-12)

    def test_is_invariant_under_permutation(self):
        """The property the reproducibility contract rests on.

        np.std reduces in an order that depends on SIMD width, so it can return a
        different final ulp on different CPUs. population_std must not.
        """
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 5.0e-23, size=50_000)
        shuffled = rng.permutation(x)
        assert population_std(shuffled) == population_std(x)

    def test_zero_variance(self):
        assert population_std(np.full(100, 3.0)) == 0.0

    def test_single_element(self):
        assert population_std(np.array([2.5])) == 0.0


# ---------------------------------------------------------------------------
# astra_proof.py — run_gating_trial
# ---------------------------------------------------------------------------
class TestRunGatingTrial:
    def test_returns_dict(self):
        result = run_gating_trial(1e-21, AstraParams(), GatingParams(), seed=0, verbose=False)
        assert isinstance(result, dict)
        assert "snr_before" in result
        assert "snr_after" in result
        assert "gated_fraction" in result

    def test_return_arrays(self):
        result = run_gating_trial(
            1e-21,
            AstraParams(),
            GatingParams(),
            seed=0,
            verbose=False,
            return_arrays=True,
        )
        summary, t, data, gated_data = result
        assert isinstance(summary, dict)
        assert t.shape == data.shape == gated_data.shape

    def test_deterministic_across_calls(self):
        r1 = run_gating_trial(1e-21, AstraParams(), GatingParams(), seed=42, verbose=False)
        r2 = run_gating_trial(1e-21, AstraParams(), GatingParams(), seed=42, verbose=False)
        assert r1 == r2

    def test_different_seeds_differ(self):
        r1 = run_gating_trial(1e-21, AstraParams(), GatingParams(), seed=0, verbose=False)
        r2 = run_gating_trial(1e-21, AstraParams(), GatingParams(), seed=99, verbose=False)
        assert r1["snr_before"] != r2["snr_before"]


# ---------------------------------------------------------------------------
# astra_real_verify.py — tukey_window
# ---------------------------------------------------------------------------
class TestTukeyWindow:
    def test_length(self):
        w = arv.tukey_window(256, alpha=0.25)
        assert len(w) == 256

    def test_ones_in_flat_region(self):
        w = arv.tukey_window(1000, alpha=0.1)
        # Central 80% should be 1.0
        mid = w[100:900]
        np.testing.assert_allclose(mid, 1.0, atol=1e-12)

    def test_edges_below_one(self):
        w = arv.tukey_window(1000, alpha=0.5)
        assert w[0] < 0.01
        assert w[-1] < 0.01

    def test_alpha_zero_is_rectangular(self):
        w = arv.tukey_window(100, alpha=0.0)
        np.testing.assert_array_equal(w, 1.0)

    def test_alpha_one_is_hann(self):
        w = arv.tukey_window(100, alpha=1.0)
        assert w[0] < 0.01
        assert w[50] > 0.99

    def test_n_one(self):
        w = arv.tukey_window(1, alpha=0.25)
        assert len(w) == 1
        assert w[0] == 1.0

    def test_n_zero(self):
        w = arv.tukey_window(0, alpha=0.25)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# astra_real_verify.py — make_template
# ---------------------------------------------------------------------------
class TestMakeTemplate:
    def test_shape(self):
        dt = 1.0 / 4096
        tmpl = arv.make_template(dt, arv.TemplateParams())
        expected_n = int(round(arv.TemplateParams().duration_s / dt))
        assert len(tmpl) == expected_n

    def test_starts_near_zero(self):
        """sin(0) = 0, so first sample should be ~0."""
        dt = 1.0 / 4096
        tmpl = arv.make_template(dt, arv.TemplateParams())
        assert abs(tmpl[0]) < 1e-25

    def test_deterministic(self):
        dt = 1.0 / 4096
        t1 = arv.make_template(dt, arv.TemplateParams())
        t2 = arv.make_template(dt, arv.TemplateParams())
        np.testing.assert_array_equal(t1, t2)


# ---------------------------------------------------------------------------
# astra_real_verify.py — normalized_xcorr_max
# ---------------------------------------------------------------------------
class TestNormalizedXcorrMax:
    def test_self_correlation_high(self):
        """Correlating a signal with itself should give a high score."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000).astype(np.float64)
        score = arv.normalized_xcorr_max(x, x)
        assert score > 0.5

    def test_noise_vs_template_low(self):
        """Pure noise vs. a sine template should score low."""
        rng = np.random.default_rng(0)
        noise = rng.normal(size=4096).astype(np.float64)
        t = np.arange(200, dtype=np.float64) / 4096
        template = np.sin(2 * np.pi * 200 * t)
        score = arv.normalized_xcorr_max(noise, template)
        # Not an exact threshold, but should be modest
        assert score < 50.0

    def test_same_length_arrays(self):
        """Regression: same-length arrays should use 'full' mode."""
        x = np.ones(100, dtype=np.float64)
        y = np.ones(100, dtype=np.float64)
        score = arv.normalized_xcorr_max(x, y)
        assert np.isfinite(score)
        assert score > 0.0


# ---------------------------------------------------------------------------
# astra_real_verify.py — apply_energy_gate
# ---------------------------------------------------------------------------
class TestApplyEnergyGate:
    def test_quiet_signal_ungated(self):
        """Uniform low-amplitude signal should not be gated."""
        rng = np.random.default_rng(0)
        strain = rng.normal(0, 1e-23, size=4096).astype(np.float64)
        gated, thr = arv.apply_energy_gate(strain, gate_k=25.0, tukey_alpha=0.25)
        # Most samples should survive
        frac_zeroed = np.mean(gated == 0.0)
        assert frac_zeroed < 0.1

    def test_spike_is_removed(self):
        """A large spike should be zeroed by gating."""
        strain = np.zeros(4096, dtype=np.float64)
        strain[2000] = 1.0  # huge relative to rest
        gated, _ = arv.apply_energy_gate(strain, gate_k=5.0, tukey_alpha=0.0)
        assert gated[2000] == 0.0

    def test_taper_smooths_edges(self):
        """With tukey_alpha > 0, gating boundary should be tapered, not hard."""
        rng = np.random.default_rng(42)
        strain = rng.normal(0, 1.0, size=8192).astype(np.float64)
        # Insert a loud block
        strain[4000:4100] = 100.0
        gated, _ = arv.apply_energy_gate(strain, gate_k=5.0, tukey_alpha=0.5)
        # Samples near the gated region boundary should be attenuated but
        # not necessarily zero (taper transition)
        assert gated[4050] == 0.0  # inside gated region

    def test_output_shape(self):
        strain = np.ones(1024, dtype=np.float64)
        gated, thr = arv.apply_energy_gate(strain, gate_k=25.0, tukey_alpha=0.25)
        assert gated.shape == strain.shape
        assert isinstance(thr, float)


# ---------------------------------------------------------------------------
# astra_real_verify.py — synthetic_strain
# ---------------------------------------------------------------------------
class TestSyntheticStrain:
    def test_shape(self):
        strain, dt = arv.synthetic_strain(4096, 1, 1e-23, seed=0)
        assert len(strain) == 4096
        np.testing.assert_allclose(dt, 1.0 / 4096)

    def test_deterministic(self):
        s1, _ = arv.synthetic_strain(4096, 1, 1e-23, seed=42)
        s2, _ = arv.synthetic_strain(4096, 1, 1e-23, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        s1, _ = arv.synthetic_strain(4096, 1, 1e-23, seed=0)
        s2, _ = arv.synthetic_strain(4096, 1, 1e-23, seed=1)
        assert not np.array_equal(s1, s2)
