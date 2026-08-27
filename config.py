"""YAML config loading with strict-by-default lookups."""
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(path) -> dict:
    """Load a YAML config. Returns {} when the file does not exist."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_named(name: str) -> dict:
    """Load one of the project configs by bare name, e.g. load_named('risk')."""
    return load_config(CONFIG_DIR / f"{name}.yaml")


def require(cfg: dict, key: str) -> Any:
    """Fetch a config key or raise. Use for values that must never silently default.

    Silent defaults are how a backtest ends up quietly using different costs than
    the one you reported last week.
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"missing required config key: {key!r}")
        cur = cur[part]
    return cur
