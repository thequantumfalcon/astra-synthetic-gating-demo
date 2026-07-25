"""Regression tests for the reproduction driver in repro/run_astra.py."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT))

run_astra = importlib.import_module("repro.run_astra")


class TestRunAstraHelpers:
    def test_deterministic_env_adds_engine_path(self, monkeypatch, tmp_path):
        engine_dir = tmp_path / "engine"
        monkeypatch.setenv("PYTHONPATH", "existing-path")
        monkeypatch.setattr(run_astra, "ENGINE_DIR", engine_dir)

        env = run_astra._deterministic_env()

        assert env["PYTHONHASHSEED"] == "0"
        assert env["OMP_NUM_THREADS"] == "1"
        python_path_entries = env["PYTHONPATH"].split(run_astra.os.pathsep)
        assert Path(python_path_entries[0]) == engine_dir

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

        proof_source = engine_dir / "astra"
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

        copied_proof = bundle_dir / "engine" / "astra" / "astra_proof.py"
        assert copied_proof.read_text(encoding="utf-8") == "print('proof')\n"
        assert len(calls) == 2
        assert calls[0][2] == "astra"
        assert calls[1][-2:] == ["--mc", str(run_astra.MC_TRIALS)]
