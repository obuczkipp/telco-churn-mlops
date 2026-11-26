"""Load the YAML configuration into a plain dictionary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the config file. Relative paths inside it are resolved against the project root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str) -> Path:
    """Turn a path from the config into an absolute path under the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
