from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class ContractError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_complete_counts(*, requested: int, generated: int, evaluated: int) -> None:
    if requested < 1:
        raise ContractError("requested sample count must be positive")
    if requested != generated or requested != evaluated:
        raise ContractError(
            f"incomplete run: requested={requested}, generated={generated}, evaluated={evaluated}"
        )


def require_metrics(records: Iterable[dict[str, Any]], required: Iterable[str]) -> None:
    required_set = set(required)
    for index, record in enumerate(records):
        missing = sorted(key for key in required_set if record.get(key) is None)
        if missing:
            raise ContractError(f"record {index} is missing required metrics: {', '.join(missing)}")


@dataclass
class ArtifactLedger:
    root: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, path: str | Path, *, kind: str) -> None:
        artifact = Path(path).resolve()
        root = self.root.resolve()
        if not artifact.is_file():
            raise ContractError(f"artifact does not exist: {artifact}")
        try:
            relative = artifact.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"artifact is outside run root: {artifact}") from exc
        self.entries.append(
            {
                "path": relative.as_posix(),
                "kind": kind,
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )

    def write(self) -> Path:
        output = self.root / "artifact_manifest.json"
        output.write_text(
            json.dumps({"artifacts": sorted(self.entries, key=lambda item: item["path"])}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return output
