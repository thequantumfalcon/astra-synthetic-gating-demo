from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(explicit_path: str | None = None) -> dict[str, Any]:
    """Load a JSON config file.

    Order:
    1) explicit_path (if provided)
    2) env var ASTRA_CONFIG
    3) config.json next to this package
    """
    if explicit_path:
        path = Path(explicit_path)
    elif os.environ.get("ASTRA_CONFIG"):
        path = Path(os.environ["ASTRA_CONFIG"])
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
