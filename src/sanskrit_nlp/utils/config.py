"""Simple YAML/JSON configuration loader with sane defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Thin dict wrapper with dotted-path access.

    Example::

        cfg = load_config("configs/base.yaml")
        cfg["metric"]["comet_model"]
    """

    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Dotted-path get: ``cfg.get("ocr.metrics")``."""
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """Dotted-path set."""
        node = self.data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        return self.data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(data=data or {})


def load_config(path: str | Path) -> Config:
    """Load a YAML (or JSON) config file.

    YAML is the canonical format; JSON works too. Raises ``ValueError``
    on unsupported extensions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        raise ValueError(f"Unsupported config format: {path.name}")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return Config.from_dict(data)


def dump_config(config: Config | dict[str, Any], path: str | Path) -> None:
    """Serialize a config to YAML."""
    path = Path(path)
    data = config.to_dict() if isinstance(config, Config) else config
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
