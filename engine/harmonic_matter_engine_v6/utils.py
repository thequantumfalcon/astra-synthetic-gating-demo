from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def package_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(explicit_path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.json.

    Order:
    1) explicit_path (if provided)
    2) env var HME_CONFIG
    3) bundled harmonic_matter_engine_v6/config.json
    """
    if explicit_path:
        path = Path(explicit_path)
    elif os.environ.get("HME_CONFIG"):
        path = Path(os.environ["HME_CONFIG"])
    else:
        path = package_root() / "config.json"

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def clamp_demo_particle_count(particle_count: int, cap: int = 1024) -> int:
    """All-to-all SPH is O(N^2). Keep demos runnable by clamping.

    This doesn't change the stored config; it's only a runtime safety for the demo orchestrator.
    """
    if particle_count <= cap:
        return particle_count
    return cap
