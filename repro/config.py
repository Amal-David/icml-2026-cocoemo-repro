from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ReproConfig:
    source: Path
    raw: dict[str, Any]
    digest: str

    @property
    def run_name(self) -> str:
        return str(self.raw["run"]["name"])

    @property
    def seed(self) -> int:
        return int(self.raw["run"]["seed"])

    @property
    def stage(self) -> str:
        return str(self.raw["run"]["stage"])

    @property
    def claims(self) -> tuple[int, ...]:
        return tuple(int(claim) for claim in self.raw.get("claims", []))


def load_config(path: str | Path) -> ReproConfig:
    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigError(f"config does not exist: {source}")

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    run = raw.get("run")
    if not isinstance(run, dict):
        raise ConfigError("config.run must be a mapping")
    for key in ("name", "seed", "stage"):
        if key not in run:
            raise ConfigError(f"config.run.{key} is required")
    if not str(run["name"]).strip():
        raise ConfigError("config.run.name must not be empty")
    if not isinstance(run["seed"], int):
        raise ConfigError("config.run.seed must be an integer")
    if run["stage"] not in {"preflight", "baseline", "claim1", "claim2", "claim3", "claim4", "claim5"}:
        raise ConfigError(f"unsupported stage: {run['stage']}")

    claims = raw.get("claims", [])
    if not isinstance(claims, list) or any(not isinstance(value, int) for value in claims):
        raise ConfigError("config.claims must be a list of integers")
    if any(value not in {1, 2, 3, 4, 5} for value in claims):
        raise ConfigError("config.claims may only contain 1 through 5")

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ReproConfig(source=source, raw=raw, digest=digest)
