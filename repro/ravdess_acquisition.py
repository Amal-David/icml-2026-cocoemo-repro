from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from repro.contracts import ContractError, sha256_file
from repro.ravdess_manifest import build_claim1_groups, require_full_public_scope


ZENODO_RECORD_URL = "https://zenodo.org/records/1188976"
ZENODO_ARCHIVE_URL = (
    "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content"
)
ZENODO_ARCHIVE_FILENAME = "Audio_Speech_Actors_01-24.zip"
ZENODO_ARCHIVE_SIZE_BYTES = 208468073
ZENODO_ARCHIVE_MD5 = "bc696df654c87fed845eb13823edef8a"
RAVDESS_LICENSE = "CC BY-NC-SA 4.0"
RAVDESS_RELEASE = "RAVDESS-v1.0-audio-speech"
EXPECTED_ARCHIVE_WAVS = 1440
EXPECTED_CLAIM1_GROUPS = 96
RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RavdessAcquisitionSettings:
    source_url: str
    release: str
    root: str
    acquire: bool
    license: str
    license_acknowledged: bool
    archive_url: str
    archive_filename: str
    archive_size_bytes: int
    archive_md5: str
    expected_archive_wavs: int
    expected_total_groups: int


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"claim1.ravdess.{key} must be a non-empty string")
    return value


def _required_positive_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or value < 1:
        raise ContractError(f"claim1.ravdess.{key} must be a positive integer")
    return value


def parse_ravdess_acquisition_settings(values: dict[str, Any]) -> RavdessAcquisitionSettings:
    acquire = values.get("acquire")
    if not isinstance(acquire, bool):
        raise ContractError("claim1.ravdess.acquire must be an explicit boolean")
    license_acknowledged = values.get("license_acknowledged")
    if not isinstance(license_acknowledged, bool):
        raise ContractError("claim1.ravdess.license_acknowledged must be an explicit boolean")

    settings = RavdessAcquisitionSettings(
        source_url=_required_string(values, "source_url"),
        release=_required_string(values, "release"),
        root=_required_string(values, "root"),
        acquire=acquire,
        license=_required_string(values, "license"),
        license_acknowledged=license_acknowledged,
        archive_url=_required_string(values, "archive_url"),
        archive_filename=_required_string(values, "archive_filename"),
        archive_size_bytes=_required_positive_int(values, "archive_size_bytes"),
        archive_md5=_required_string(values, "archive_md5").lower(),
        expected_archive_wavs=_required_positive_int(values, "expected_archive_wavs"),
        expected_total_groups=_required_positive_int(values, "expected_total_groups"),
    )
    expected = {
        "source_url": ZENODO_RECORD_URL,
        "release": RAVDESS_RELEASE,
        "license": RAVDESS_LICENSE,
        "archive_url": ZENODO_ARCHIVE_URL,
        "archive_filename": ZENODO_ARCHIVE_FILENAME,
        "archive_size_bytes": ZENODO_ARCHIVE_SIZE_BYTES,
        "archive_md5": ZENODO_ARCHIVE_MD5,
        "expected_archive_wavs": EXPECTED_ARCHIVE_WAVS,
        "expected_total_groups": EXPECTED_CLAIM1_GROUPS,
    }
    for key, required in expected.items():
        if getattr(settings, key) != required:
            raise ContractError(
                f"claim1.ravdess.{key} must pin the official Zenodo 1188976 archive"
            )
    if not settings.license_acknowledged:
        raise ContractError(
            "claim1.ravdess.license_acknowledged must be true for CC BY-NC-SA 4.0 acquisition"
        )
    relative_root = Path(settings.root)
    if relative_root.is_absolute() or ".." in relative_root.parts:
        raise ContractError("claim1.ravdess.root must be a relative cache path without '..'")
    return settings


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324: matching the upstream Zenodo checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, settings: RavdessAcquisitionSettings) -> None:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"RAVDESS archive is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != settings.archive_size_bytes:
        raise ContractError(
            "RAVDESS archive size mismatch: "
            f"expected={settings.archive_size_bytes}, actual={actual_size}"
        )
    actual_md5 = _md5_file(path)
    if actual_md5 != settings.archive_md5:
        raise ContractError(
            "RAVDESS archive MD5 mismatch: "
            f"expected={settings.archive_md5}, actual={actual_md5}"
        )

    try:
        with zipfile.ZipFile(path) as archive:
            wav_count = sum(
                not member.is_dir() and PurePosixPath(member.filename).suffix.lower() == ".wav"
                for member in archive.infolist()
            )
    except zipfile.BadZipFile as exc:
        raise ContractError(f"RAVDESS archive is not a valid ZIP: {path}") from exc
    if wav_count != settings.expected_archive_wavs:
        raise ContractError(
            "RAVDESS archive WAV count mismatch: "
            f"expected={settings.expected_archive_wavs}, actual={wav_count}"
        )


def _validate_member(member: zipfile.ZipInfo) -> PurePosixPath:
    raw = member.filename
    relative = PurePosixPath(raw)
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ContractError(f"RAVDESS archive contains symlink member: {raw}")
    if "\\" in raw or relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"RAVDESS archive contains unsafe path: {raw}")
    if not relative.parts or relative.parts == (".",):
        raise ContractError(f"RAVDESS archive contains empty path: {raw}")
    return relative


def _verify_extracted_dataset(root: Path, settings: RavdessAcquisitionSettings) -> Path:
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"RAVDESS extraction root is not a directory: {root}")
    wav_paths = sorted(path for path in root.rglob("*.wav") if path.is_file())
    if len(wav_paths) != settings.expected_archive_wavs:
        raise ContractError(
            "RAVDESS extracted WAV count mismatch: "
            f"expected={settings.expected_archive_wavs}, actual={len(wav_paths)}"
        )
    groups = build_claim1_groups(wav_paths)
    require_full_public_scope(groups)
    if len(groups) != settings.expected_total_groups:
        raise ContractError(
            "RAVDESS extracted Claim 1 group count mismatch: "
            f"expected={settings.expected_total_groups}, actual={len(groups)}"
        )
    return root


def _receipt_path(dataset_root: Path) -> Path:
    return dataset_root.parent / f"{dataset_root.name}.receipt.json"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"RAVDESS extraction contains a symlink: {path}")
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ContractError(f"RAVDESS extraction contains no regular files: {root}")
    return entries


def _receipt_payload(
    *, dataset_root: Path, settings: RavdessAcquisitionSettings, tree_manifest: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest_sha256 = hashlib.sha256(_canonical_json(tree_manifest)).hexdigest()
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "dataset_directory": dataset_root.name,
        "source": {
            "record_url": settings.source_url,
            "release": settings.release,
            "license": settings.license,
        },
        "archive": {
            "url": settings.archive_url,
            "filename": settings.archive_filename,
            "size_bytes": settings.archive_size_bytes,
            "md5": settings.archive_md5,
        },
        "tree": {
            "file_count": len(tree_manifest),
            "manifest_sha256": manifest_sha256,
            "manifest": tree_manifest,
        },
    }


def _write_receipt_atomic(receipt_path: Path, payload: dict[str, Any]) -> None:
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ContractError(
            f"RAVDESS extraction receipt already exists and will not be overwritten: {receipt_path}"
        )
    temporary = receipt_path.with_name(f".{receipt_path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, receipt_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verify_receipt(
    *, dataset_root: Path, settings: RavdessAcquisitionSettings
) -> None:
    receipt_path = _receipt_path(dataset_root)
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ContractError(
            "RAVDESS cache has no verified extraction receipt; do not use this pre-existing "
            "tree as claim evidence. Remove the cache manually, then rerun with "
            "claim1.ravdess.acquire: true."
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"RAVDESS extraction receipt is unreadable: {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise ContractError(f"RAVDESS extraction receipt is not a mapping: {receipt_path}")

    actual_manifest = _tree_manifest(dataset_root)
    expected = _receipt_payload(
        dataset_root=dataset_root, settings=settings, tree_manifest=actual_manifest
    )
    if receipt != expected:
        raise ContractError(
            "RAVDESS extraction receipt does not match the pinned archive identity and "
            "deterministic extracted-tree manifest; do not use this cache as claim evidence. "
            "Remove the cache manually, then rerun with claim1.ravdess.acquire: true."
        )


def _verify_existing_dataset(root: Path, settings: RavdessAcquisitionSettings) -> Path:
    _verify_extracted_dataset(root, settings)
    _verify_receipt(dataset_root=root, settings=settings)
    return root


def _download_archive(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=600) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _download_immutable_archive(archive_path: Path, settings: RavdessAcquisitionSettings) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() or archive_path.is_symlink():
        verify_archive(archive_path, settings)
        return archive_path

    temporary = archive_path.with_name(f".{archive_path.name}.{uuid.uuid4().hex}.partial")
    try:
        _download_archive(settings.archive_url, temporary)
        verify_archive(temporary, settings)
        os.replace(temporary, archive_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return archive_path


def _extract_immutable_archive(
    archive_path: Path, dataset_root: Path, settings: RavdessAcquisitionSettings
) -> Path:
    if dataset_root.exists() or dataset_root.is_symlink():
        return _verify_existing_dataset(dataset_root, settings)

    receipt_path = _receipt_path(dataset_root)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ContractError(
            "RAVDESS cache has an orphan extraction receipt without its dataset tree; "
            f"it will not be overwritten: {receipt_path}"
        )

    staging_parent = dataset_root.parent / f".{dataset_root.name}.{uuid.uuid4().hex}.extracting"
    staging_parent.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = _validate_member(member)
                destination = staging_parent.joinpath(*relative.parts)
                destination_parent = destination.parent
                destination_parent.mkdir(parents=True, exist_ok=True)
                if member.is_dir():
                    destination.mkdir(exist_ok=True)
                    continue
                with archive.open(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)

        staged_root = staging_parent / dataset_root.name
        _verify_extracted_dataset(staged_root, settings)
        receipt = _receipt_payload(
            dataset_root=dataset_root,
            settings=settings,
            tree_manifest=_tree_manifest(staged_root),
        )
        os.replace(staged_root, dataset_root)
        _write_receipt_atomic(receipt_path, receipt)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    shutil.rmtree(staging_parent, ignore_errors=True)
    return dataset_root


def ensure_ravdess_dataset(*, repo: Path, settings: RavdessAcquisitionSettings) -> Path:
    """Return a verified immutable RAVDESS cache, acquiring it only with config opt-in."""
    dataset_root = repo / settings.root
    if dataset_root.exists() or dataset_root.is_symlink():
        return _verify_existing_dataset(dataset_root, settings)
    if not settings.acquire:
        raise ContractError(
            "RAVDESS cache is absent; set claim1.ravdess.acquire: true after acknowledging "
            "CC BY-NC-SA 4.0 to download the pinned official archive"
        )
    archive_path = dataset_root.parent / "archives" / settings.archive_filename
    archive = _download_immutable_archive(archive_path, settings)
    return _extract_immutable_archive(archive, dataset_root, settings)
