"""Configuration management utilities."""

from pathlib import Path
from typing import Any
import yaml
from loguru import logger


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return config


def merge_configs(*configs: dict) -> dict[str, Any]:
    """Merge multiple configurations (later configs override earlier)."""
    result = {}
    for config in configs:
        _deep_merge(result, config)
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Deep merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_config(config: dict, path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Saved config to {path}")
