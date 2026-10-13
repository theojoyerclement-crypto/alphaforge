"""Load experiment configs from YAML/JSON, with dotted command-line overrides."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from alphaforge.config.schema import ExperimentConfig


class ConfigError(ValueError):
    """A config file that cannot be read or validated; the message is meant for the user."""


def _read_raw(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(text)
        elif path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raise ConfigError(f"Unsupported config format {path.suffix!r} (use .yaml/.yml/.json)")
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        kind = path.suffix.lstrip(".").upper()
        raise ConfigError(f"{path} is not valid {kind}:\n{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config {path} must contain a mapping at the top level")
    return raw


def _allowed_keys(loc: tuple[int | str, ...]) -> list[str] | None:
    """Field names accepted at the parent of ``loc`` (to suggest fixes for typos)."""
    model: type[BaseModel] = ExperimentConfig
    for part in loc[:-1]:
        if isinstance(part, int):
            continue
        field = model.model_fields.get(part)
        annotation = getattr(field, "annotation", None)
        args = getattr(annotation, "__args__", ())
        candidates = [annotation, *args]
        nested = [c for c in candidates if isinstance(c, type) and issubclass(c, BaseModel)]
        if not nested:
            return None
        model = nested[0]
    return list(model.model_fields)


def format_validation_error(path: str | Path, error: ValidationError) -> str:
    """One readable line per problem, e.g. ``portfolio.target_volatility: unknown key``."""
    lines = [f"{path}: {error.error_count()} invalid setting(s)"]
    for item in error.errors():
        where = ".".join(str(p) for p in item["loc"]) or "<root>"
        if item["type"] == "extra_forbidden":
            allowed = _allowed_keys(item["loc"])
            hint = f" (allowed: {', '.join(allowed)})" if allowed else ""
            lines.append(f"  - {where}: unknown key{hint}")
        else:
            lines.append(f"  - {where}: {item['msg']} (got {item.get('input')!r})")
    return "\n".join(lines)


def apply_overrides(raw: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``key.sub=value`` overrides (values parsed as YAML) to a raw config mapping.

    Example:
        ``apply_overrides(raw, ["portfolio.target_vol=0.15", "data.timeframe=4h"])``
    """
    result = copy.deepcopy(dict(raw))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override {item!r} must look like key.path=value")
        dotted, value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node: dict[str, Any] = result
        for key in keys[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise ConfigError(f"Cannot override {dotted!r}: {key!r} is not a mapping")
            node = child
        node[keys[-1]] = yaml.safe_load(value)
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a YAML or JSON file.

    Raises:
        ConfigError: unreadable file, bad syntax, or settings that fail validation.
    """
    raw = apply_overrides(_read_raw(Path(path)), overrides)
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """JSON-safe dictionary representation (used for hashing and run manifests)."""
    return config.model_dump(mode="json")
