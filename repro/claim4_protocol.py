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
_GIT_BLOB_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_CREMAD_WAV_NAME = re.compile(r"^[0-9]{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.wav$")
_CLAIM4_ASSET_SCHEMA = "claim4_cremad_lfs_assets_v1"
_CLAIM4_REPOSITORY = "CheyneyComputerScience/CREMA-D"


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


@dataclass(frozen=True)
class FrozenClaim4Asset:
    """One immutable CREMA-D LFS object selected by the public proxy."""

    filename: str
    repository_path: str
    git_blob_sha1: str
    lfs_oid_sha256: str
    size_bytes: int
    download_url: str


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


def git_blob_sha1(payload: bytes) -> str:
    """Return Git's SHA-1 object identifier for an exact blob payload."""

    if not isinstance(payload, bytes):
        raise ContractError("Git blob payload must be bytes")
    return hashlib.sha1(f"blob {len(payload)}\\0".encode("ascii") + payload).hexdigest()


def _expected_claim4_asset_url(*, revision: str, filename: str) -> str:
    return (
        "https://media.githubusercontent.com/media/"
        f"{_CLAIM4_REPOSITORY}/{revision}/AudioWAV/{filename}"
    )


def _expected_claim4_pointer_url(*, revision: str, filename: str) -> str:
    """Return the immutable raw Git blob URL used only to freeze an LFS pointer."""

    return (
        "https://raw.githubusercontent.com/"
        f"{_CLAIM4_REPOSITORY}/{revision}/AudioWAV/{filename}"
    )


def _require_claim4_asset_filename(filename: object) -> str:
    if not isinstance(filename, str) or _CREMAD_WAV_NAME.fullmatch(filename) is None:
        raise ContractError(f"invalid frozen Claim 4 CREMA-D WAV filename: {filename!r}")
    return filename


def load_frozen_claim4_asset_manifest(
    path: Path, *, expected_sha256: str, revision: str
) -> dict[str, FrozenClaim4Asset]:
    """Load the committed asset manifest without any GitHub metadata requests.

    The manifest is a reproducibility boundary: runtime callers get only pinned
    media URLs and independently checked Git-LFS object identities.  No mutable
    branch URL, Contents API request, or Blob API request is accepted.
    """

    if not path.is_file():
        raise ContractError(f"frozen Claim 4 asset manifest is missing: {path}")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ContractError("Claim 4 frozen asset-manifest SHA-256 must be lowercase hexadecimal")
    actual_hash = sha256_file(path)
    if actual_hash != expected_sha256:
        raise ContractError(
            "frozen Claim 4 asset-manifest SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_hash}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"frozen Claim 4 asset manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "source", "assets"}:
        raise ContractError("frozen Claim 4 asset manifest has an unexpected top-level schema")
    if payload["schema_version"] != _CLAIM4_ASSET_SCHEMA:
        raise ContractError("frozen Claim 4 asset manifest schema version changed")
    source = payload["source"]
    if not isinstance(source, dict) or set(source) != {"repository", "revision"}:
        raise ContractError("frozen Claim 4 asset manifest source schema changed")
    if source != {"repository": _CLAIM4_REPOSITORY, "revision": revision}:
        raise ContractError("frozen Claim 4 asset manifest repository or revision mismatch")
    raw_assets = payload["assets"]
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ContractError("frozen Claim 4 asset manifest must contain assets")

    assets: dict[str, FrozenClaim4Asset] = {}
    last_filename = ""
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, dict) or set(raw_asset) != {
            "filename", "repository_path", "git_blob_sha1", "lfs_oid_sha256", "size_bytes", "download_url"
        }:
            raise ContractError("frozen Claim 4 asset entry schema changed")
        filename = _require_claim4_asset_filename(raw_asset["filename"])
        if filename <= last_filename:
            raise ContractError("frozen Claim 4 assets must be uniquely sorted by filename")
        last_filename = filename
        repository_path = raw_asset["repository_path"]
        if repository_path != f"AudioWAV/{filename}":
            raise ContractError(f"frozen Claim 4 asset path mismatch for {filename}")
        git_sha1 = raw_asset["git_blob_sha1"]
        lfs_oid = raw_asset["lfs_oid_sha256"]
        size_bytes = raw_asset["size_bytes"]
        download_url = raw_asset["download_url"]
        if not isinstance(git_sha1, str) or _GIT_BLOB_SHA1.fullmatch(git_sha1) is None:
            raise ContractError(f"frozen Claim 4 Git blob SHA-1 is invalid for {filename}")
        if not isinstance(lfs_oid, str) or re.fullmatch(r"[0-9a-f]{64}", lfs_oid) is None:
            raise ContractError(f"frozen Claim 4 LFS SHA-256 is invalid for {filename}")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 1:
            raise ContractError(f"frozen Claim 4 LFS size is invalid for {filename}")
        expected_url = _expected_claim4_asset_url(revision=revision, filename=filename)
        if download_url != expected_url:
            raise ContractError(f"frozen Claim 4 raw download URL mismatch for {filename}")
        assets[filename] = FrozenClaim4Asset(
            filename=filename,
            repository_path=repository_path,
            git_blob_sha1=git_sha1,
            lfs_oid_sha256=lfs_oid,
            size_bytes=size_bytes,
            download_url=download_url,
        )
    return assets


def require_claim4_asset_coverage(
    manifest: Sequence[Claim4ManifestItem], assets: Mapping[str, FrozenClaim4Asset]
) -> None:
    """Require exact target/reference coverage for the frozen 60-actor selection."""

    required = {
        f"{file_name}.wav"
        for item in manifest
        for file_name in (item.file_name, item.reference_file_name)
    }
    available = set(assets)
    if required != available:
        raise ContractError(
            "frozen Claim 4 asset coverage mismatch: "
            f"missing={sorted(required - available)}, unexpected={sorted(available - required)}"
        )


def claim4_asset_for_filename(
    assets: Mapping[str, FrozenClaim4Asset], file_name: str
) -> FrozenClaim4Asset:
    filename = _require_claim4_asset_filename(f"{file_name}.wav")
    asset = assets.get(filename)
    if asset is None:
        raise ContractError(f"frozen Claim 4 asset is missing for selected filename: {filename}")
    return asset


def _safe_cache_path(cache_root: Path, item: Claim4ManifestItem, *, filename: str | None = None) -> Path:
    if not cache_root.is_absolute() or ".." in cache_root.parts:
        raise ContractError(f"unsafe CREMA-D cache root: {cache_root}")
    # Do not follow a user-controlled cache through a symlink.  A verified
    # digest is not sufficient when the cache path can escape its intended
    # private workspace.
    for parent in (cache_root, *cache_root.parents):
        if parent.is_symlink():
            raise ContractError(f"unsafe CREMA-D cache root symlink: {parent}")
    if cache_root.exists() and not cache_root.is_dir():
        raise ContractError(f"CREMA-D cache root is not a directory: {cache_root}")
    if not isinstance(item.actor_id, str) or re.fullmatch(r"[0-9]{4}", item.actor_id) is None:
        raise ContractError(f"unsafe CREMA-D actor cache component: {item.actor_id!r}")
    filename = filename or item.wav_name
    if Path(filename).name != filename or not filename.endswith(".wav"):
        raise ContractError(f"unsafe CREMA-D audio filename: {filename}")
    target = cache_root / item.actor_id / filename
    try:
        target.relative_to(cache_root)
    except ValueError as exc:
        raise ContractError(f"unsafe CREMA-D cache target: {target}") from exc
    if target.is_symlink() or target.parent.is_symlink():
        raise ContractError(f"unsafe CREMA-D cache target symlink: {target}")
    return target


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
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise ContractError(f"unsafe CREMA-D cached WAV path: {target}")
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
    # Re-check after mkdir: an existing cache parent must remain a real
    # directory before a verified object is promoted into it.
    if target.parent.is_symlink() or not target.parent.is_dir() or target.is_symlink():
        raise ContractError(f"unsafe CREMA-D cache target after directory creation: {target}")
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
