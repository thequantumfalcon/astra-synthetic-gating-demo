#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import atheris

with atheris.instrument_imports():
    from astra.utils import load_config, save_json


def _consume_text(provider: atheris.FuzzedDataProvider, max_len: int = 32) -> str:
    size = provider.ConsumeIntInRange(0, max_len)
    return provider.ConsumeBytes(size).decode("utf-8", errors="ignore")


def _consume_json_scalar(provider: atheris.FuzzedDataProvider) -> Any:
    kind = provider.ConsumeIntInRange(0, 5)
    if kind == 0:
        return None
    if kind == 1:
        return bool(provider.ConsumeIntInRange(0, 1))
    if kind == 2:
        return provider.ConsumeIntInRange(-10_000, 10_000)
    if kind == 3:
        value = provider.ConsumeFloat()
        if not math.isfinite(value):
            return 0.0
        return max(min(value, 1.0e6), -1.0e6)
    if kind == 4:
        return _consume_text(provider)
    return provider.ConsumeIntInRange(-1_000_000, 1_000_000)


def _consume_json_value(provider: atheris.FuzzedDataProvider, depth: int = 0) -> Any:
    if depth >= 2:
        return _consume_json_scalar(provider)

    kind = provider.ConsumeIntInRange(0, 2)
    if kind == 0:
        return _consume_json_scalar(provider)
    if kind == 1:
        length = provider.ConsumeIntInRange(0, 4)
        return [_consume_json_value(provider, depth + 1) for _ in range(length)]

    length = provider.ConsumeIntInRange(0, 4)
    out: dict[str, Any] = {}
    for _ in range(length):
        out[_consume_text(provider, 16)] = _consume_json_value(provider, depth + 1)
    return out


def TestOneInput(data: bytes) -> None:
    provider = atheris.FuzzedDataProvider(data)

    payload = _consume_json_value(provider)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config_path = tmp_path / "config.json"
        save_json(config_path, payload)
        loaded = load_config(str(config_path))

        if json.dumps(loaded, sort_keys=True) != json.dumps(payload, sort_keys=True):
            raise RuntimeError("load_config/save_json round-trip changed payload")

        malformed_path = tmp_path / "malformed.json"
        malformed_path.write_text(_consume_text(provider, 128), encoding="utf-8")
        try:
            load_config(str(malformed_path))
        except (json.JSONDecodeError, OSError):
            pass


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
