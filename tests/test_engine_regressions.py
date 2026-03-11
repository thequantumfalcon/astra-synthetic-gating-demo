from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT))

from harmonic_matter_engine_v6.agents.architect import GenerativeArchitect
from harmonic_matter_engine_v6.core.walrus import WalrusSurrogate
import repro.run_astra as run_astra


def _demo_config() -> dict[str, dict[str, float]]:
    return {
        "physics": {
            "smoothing_length": 0.3,
            "speed_of_sound": 10.0,
            "rest_density": 1.0,
            "viscosity": 0.01,
            "background_pressure": 0.0,
        }
    }


class TestWalrusSurrogate:
    def test_generate_initial_state_is_deterministic_for_same_seed(self, monkeypatch):
        monkeypatch.setattr("harmonic_matter_engine_v6.core.walrus.time.sleep", lambda _: None)

        walrus_a = WalrusSurrogate(seed=7)
        walrus_b = WalrusSurrogate(seed=7)

        pos_a, vel_a = walrus_a.generate_initial_state("demo", particle_count=8)
        pos_b, vel_b = walrus_b.generate_initial_state("demo", particle_count=8)

        np.testing.assert_array_equal(pos_a, pos_b)
        np.testing.assert_array_equal(vel_a, vel_b)

    def test_explicit_seed_overrides_instance_seed(self, monkeypatch):
        monkeypatch.setattr("harmonic_matter_engine_v6.core.walrus.time.sleep", lambda _: None)

        walrus = WalrusSurrogate(seed=1)

        pos_a, vel_a = walrus.generate_initial_state("demo", particle_count=4, seed=9)
        pos_b, vel_b = walrus.generate_initial_state("demo", particle_count=4, seed=9)

        np.testing.assert_array_equal(pos_a, pos_b)
        np.testing.assert_array_equal(vel_a, vel_b)


class TestArchitectAndLuthier:
    def test_architect_prompt_heuristics(self):
        architect = GenerativeArchitect()

        diamond = architect.design_simulation("Liquid diamond concept")
        default = architect.design_simulation("water")

        assert diamond["stiffness"] == 200.0
        assert diamond["surface_tension"] == 0.9
        assert default == {
            "viscosity": 0.015,
            "stiffness": 100.0,
            "surface_tension": 0.6,
        }

    def test_luthier_bakes_expected_scene_shape(self):
        pytest.importorskip("jax")
        MaterialLuthier = importlib.import_module(
            "harmonic_matter_engine_v6.agents.luthier"
        ).MaterialLuthier

        luthier = MaterialLuthier(num_splats=16)
        scene = luthier.bake_scene()

        assert scene["xyz"].shape == (16, 3)
        assert scene["rotation"].shape == (16, 4)
        assert scene["impedance"].shape == (16, 1)


class TestAudioVisualGaussianSplatting:
    def test_init_scene_is_reproducible_across_instances(self):
        pytest.importorskip("jax")
        AudioVisualGaussianSplatting = importlib.import_module(
            "harmonic_matter_engine_v6.core.av_gs"
        ).AudioVisualGaussianSplatting

        scene_a = AudioVisualGaussianSplatting(num_splats=8, seed=3).init_scene()
        scene_b = AudioVisualGaussianSplatting(num_splats=8, seed=3).init_scene()

        for key in scene_a:
            np.testing.assert_array_equal(scene_a[key], scene_b[key])

    def test_query_acoustic_field_returns_expected_shape(self):
        pytest.importorskip("jax")
        jnp = importlib.import_module("jax.numpy")
        AudioVisualGaussianSplatting = importlib.import_module(
            "harmonic_matter_engine_v6.core.av_gs"
        ).AudioVisualGaussianSplatting

        splats = AudioVisualGaussianSplatting(num_splats=6, seed=0).init_scene()
        query_points = jnp.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.1]], dtype=jnp.float32)

        impedance, absorption = AudioVisualGaussianSplatting(num_splats=6, seed=0).query_acoustic_field(
            splats, query_points
        )

        assert impedance.shape == (2, 1)
        assert absorption.shape == (2, 1)
        assert np.all(np.isfinite(np.asarray(impedance)))
        assert np.all(np.isfinite(np.asarray(absorption)))


class TestLiquidPhysics:
    def test_export_step_fn_preserves_shapes(self):
        pytest.importorskip("jax")
        jnp = importlib.import_module("jax.numpy")
        LiquidPhysics = importlib.import_module(
            "harmonic_matter_engine_v6.core.jax_sph"
        ).LiquidPhysics

        solver = LiquidPhysics(_demo_config())
        step_fn = solver.export_step_fn(dt=0.01)

        pos = jnp.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=jnp.float32)
        vel = jnp.zeros_like(pos)
        mass = jnp.ones((2,), dtype=jnp.float32)

        new_pos, new_vel, rho = step_fn(pos, vel, mass)

        assert new_pos.shape == pos.shape
        assert new_vel.shape == vel.shape
        assert rho.shape == (2,)


class TestRunAstraHelpers:
    def test_deterministic_env_adds_engine_path(self, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "existing-path")
        monkeypatch.setattr(run_astra, "ENGINE_DIR", Path("C:/tmp/engine"))

        env = run_astra._deterministic_env()

        assert env["PYTHONHASHSEED"] == "0"
        assert env["OMP_NUM_THREADS"] == "1"
        python_path_entries = env["PYTHONPATH"].split(run_astra.os.pathsep)
        assert Path(python_path_entries[0]) == Path("C:/tmp/engine")

    def test_write_manifest_creates_expected_payload(self, tmp_path):
        run_astra._write_manifest(tmp_path)

        manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))

        assert manifest["seed"] == run_astra.SEED
        assert manifest["mc_trials"] == run_astra.MC_TRIALS
        assert "python_version" in manifest

    def test_main_copies_engine_proof_into_bundle(self, tmp_path, monkeypatch):
        paper_dir = tmp_path / "paper"
        engine_dir = tmp_path / "engine"
        astra_output_dir = tmp_path / "astra_output"
        bundle_dir = tmp_path / "bundle"

        paper_dir.mkdir()
        (paper_dir / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")

        proof_source = engine_dir / "harmonic_matter_engine_v6" / "astra"
        proof_source.mkdir(parents=True)
        (proof_source / "astra_proof.py").write_text("print('proof')\n", encoding="utf-8")

        calls: list[list[str]] = []

        def fake_run(cmd: list[str], cwd: Path | None = None) -> None:
            _ = cwd
            calls.append(cmd)

        monkeypatch.setattr(run_astra, "PAPER_DIR", paper_dir)
        monkeypatch.setattr(run_astra, "ENGINE_DIR", engine_dir)
        monkeypatch.setattr(run_astra, "ASTRA_OUTPUT_DIR", astra_output_dir)
        monkeypatch.setattr(run_astra, "BUNDLE_DIR", bundle_dir)
        monkeypatch.setattr(run_astra, "_run", fake_run)

        run_astra.main()

        copied_proof = bundle_dir / "engine" / "harmonic_matter_engine_v6" / "astra" / "astra_proof.py"
        assert copied_proof.read_text(encoding="utf-8") == "print('proof')\n"
        assert len(calls) == 2
        assert calls[0][2] == "harmonic_matter_engine_v6.astra"
        assert calls[1][-2:] == ["--mc", str(run_astra.MC_TRIALS)]
