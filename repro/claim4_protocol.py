"""Frozen, offline-testable parts of the Claim 4 public-data protocol.

The paper's 772-item CREMA-D manifest was not released.  This module therefore
defines a smaller public proxy before any audio is fetched or model is loaded:
one multi-rater disagreement item for each of 60 deterministically selected
actors.  It also owns the Git-LFS byte-integrity contract for those items.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from repro.contracts import ContractError, sha256_file
from repro.cremad_selection import VoteRow, is_supported_disagreement, target_distribution


CLAIM4_ARM_NAMES = (
    "released_four_way",
    "dominant_non_neutral",
    "shuffled_distribution",
    "random_norm_matched",
    "renormalized_non_neutral_diagnostic",
    "neutral_zero_vector_diagnostic",
    "identity_neutral_control",
)
REQUIRED_CLAIM4_ARMS = len(CLAIM4_ARM_NAMES)
_LFS_OID = re.compile(r"^oid sha256:([0-9a-f]{64})$", re.MULTILINE)
_LFS_SIZE = re.compile(r"^size ([0-9]+)$", re.MULTILINE)


@dataclass(frozen=True)
class Claim4ManifestItem:
    actor_id: str
    row_id: int
    file_name: str
    provided_label: str
    distribution: dict[str, float]
    reference_row_id: int
    reference_file_name: str
    reference_provided_label: str
    reference_majority_labels: tuple[str, ...]

    @property
    def wav_name(self) -> str:
        return f"{self.file_name}.wav"

    @property
    def reference_wav_name(self) -> str:
        return f"{self.reference_file_name}.wav"


@dataclass(frozen=True)
class LfsPointer:
    oid_sha256: str
    size_bytes: int


def _stable_rank(*parts: object) -> str:
    value = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def cremad_actor_id(file_name: str) -> str:
    parts = file_name.split("_")
    if len(parts) != 4 or not parts[0].isdigit() or len(parts[0]) != 4:
        raise ContractError(f"invalid CREMA-D filename for actor selection: {file_name}")
    return parts[0]


def is_claim4_eligible(row: VoteRow) -> bool:
    """Public-proxy stratification for a neutral-plus-multiple-emotion target.

    This is deliberately not attributed to the paper. It makes the bounded
    public selection unambiguous: supported disagreement, positive neutral
    mass, two or more active non-neutral labels, and one unique dominant
    non-neutral label.
    """

    if not is_supported_disagreement(row):
        return False
    if row.counts["N"] <= 0:
        return False
    non_neutral_counts = [row.counts[label] for label in ("A", "H", "S")]
    if sum(count > 0 for count in non_neutral_counts) < 2:
        return False
    return non_neutral_counts.count(max(non_neutral_counts)) == 1


def build_claim4_manifest(
    rows: Sequence[VoteRow], *, actor_count: int, seed: int
) -> list[Claim4ManifestItem]:
    """Select exactly one eligible row for each of ``actor_count`` actors.

    Actor and within-actor item choice use SHA-256 ranks rather than input file
    order, making selection deterministic across CSV parsers and platforms.
    """

    if actor_count < 1:
        raise ContractError("claim4 actor_count must be positive")
    if not isinstance(seed, int):
        raise ContractError("claim4 seed must be an integer")
    grouped: dict[str, list[VoteRow]] = {}
    neutral_references: dict[str, list[VoteRow]] = {}
    for row in rows:
        actor = cremad_actor_id(row.file_name)
        if is_claim4_eligible(row):
            grouped.setdefault(actor, []).append(row)
        if row.provided_label == "N" and row.majority_labels == ("N",):
            neutral_references.setdefault(actor, []).append(row)
    if len(grouped) < actor_count:
        raise ContractError(
            "insufficient eligible CREMA-D actors: "
            f"required={actor_count}, available={len(grouped)}"
        )

    selected_actors = sorted(grouped, key=lambda actor: (_stable_rank(seed, "actor", actor), actor))[ :actor_count]
    manifest: list[Claim4ManifestItem] = []
    for actor in selected_actors:
        row = min(
            grouped[actor],
            key=lambda candidate: (
                _stable_rank(seed, "sample", actor, candidate.row_id, candidate.file_name),
                candidate.row_id,
                candidate.file_name,
            ),
        )
        references = [
            candidate
            for candidate in neutral_references.get(actor, [])
            if candidate.file_name != row.file_name
        ]
        if not references:
            raise ContractError(
                f"Claim 4 selected actor has no distinct neutral reference: actor={actor}"
            )
        reference = min(
            references,
            key=lambda candidate: (
                _stable_rank(seed, "neutral_reference", actor, row.row_id, candidate.row_id, candidate.file_name),
                candidate.row_id,
                candidate.file_name,
            ),
        )
        if (
            reference.provided_label != "N"
            or reference.majority_labels != ("N",)
            or reference.file_name == row.file_name
        ):
            raise ContractError("Claim 4 neutral reference contract failed")
        manifest.append(
            Claim4ManifestItem(
                actor_id=actor,
                row_id=row.row_id,
                file_name=row.file_name,
                provided_label=row.provided_label,
                distribution=target_distribution(row),
                reference_row_id=reference.row_id,
                reference_file_name=reference.file_name,
                reference_provided_label=reference.provided_label,
                reference_majority_labels=reference.majority_labels,
            )
        )
    manifest.sort(key=lambda item: item.actor_id)
    if len(manifest) != actor_count or len({item.actor_id for item in manifest}) != actor_count:
        raise ContractError("Claim 4 manifest must contain exactly one row per selected actor")
    return manifest


def parse_lfs_pointer(payload: bytes | str) -> LfsPointer:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    if not text.startswith("version https://git-lfs.github.com/spec/v1\n"):
        raise ContractError("Git-LFS pointer has an unexpected version header")
    oid = _LFS_OID.search(text)
    size = _LFS_SIZE.search(text)
    if oid is None or size is None:
        raise ContractError("Git-LFS pointer must include sha256 oid and size")
    return LfsPointer(oid_sha256=oid.group(1), size_bytes=int(size.group(1)))


def _safe_cache_path(cache_root: Path, item: Claim4ManifestItem, *, filename: str | None = None) -> Path:
    filename = filename or item.wav_name
    if Path(filename).name != filename or not filename.endswith(".wav"):
        raise ContractError(f"unsafe CREMA-D audio filename: {filename}")
    return cache_root / item.actor_id / filename


def ensure_lfs_wav(
    *,
    cache_root: Path,
    item: Claim4ManifestItem,
    pointer: LfsPointer,
    fetch_content: Callable[[], bytes],
    filename: str | None = None,
) -> Path:
    """Return a cache entry only after its LFS object hash and size are verified.

    ``fetch_content`` is injected so offline tests exercise every failure mode
    without network access.  Promotion is atomic, so a failed download cannot
    become a future cache hit.
    """

    target = _safe_cache_path(cache_root, item, filename=filename)
    if target.exists():
        if target.stat().st_size != pointer.size_bytes or sha256_file(target) != pointer.oid_sha256:
            raise ContractError(f"cached CREMA-D WAV failed LFS verification: {target}")
        return target

    payload = fetch_content()
    if not isinstance(payload, bytes):
        raise ContractError("Git-LFS content fetcher must return bytes")
    if len(payload) != pointer.size_bytes:
        raise ContractError(
            f"Git-LFS WAV size mismatch: expected={pointer.size_bytes}, actual={len(payload)}"
        )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != pointer.oid_sha256:
        raise ContractError(f"Git-LFS WAV sha256 mismatch: expected={pointer.oid_sha256}, actual={actual}")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def fetch_url_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def write_manifest(path: Path, manifest: Sequence[Claim4ManifestItem]) -> None:
    path.write_text(
        json.dumps([asdict(item) for item in manifest], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
