"""Load experiment configs from YAML/JSON, with dotted command-line overrides."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from alphaforge.config.schema import ExperimentConfig


def _read_raw(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format {path.suffix!r} (use .yaml/.yml/.json)")
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level")
    return raw


def apply_overrides(raw: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``key.sub=value`` overrides (values parsed as YAML) to a raw config mapping.

    Example:
        ``apply_overrides(raw, ["portfolio.target_vol=0.15", "data.timeframe=4h"])``
    """
    result = copy.deepcopy(dict(raw))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key.path=value")
        dotted, value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node: dict[str, Any] = result
        for key in keys[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override {dotted!r}: {key!r} is not a mapping")
            node = child
        node[keys[-1]] = yaml.safe_load(value)
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a YAML or JSON file."""
    raw = apply_overrides(_read_raw(Path(path)), overrides)
    return ExperimentConfig.model_validate(raw)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """JSON-safe dictionary representation (used for hashing and run manifests)."""
    return config.model_dump(mode="json")
