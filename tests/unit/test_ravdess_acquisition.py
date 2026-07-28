from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from repro.contracts import ContractError
from repro.ravdess_acquisition import (
    RavdessAcquisitionSettings,
    ensure_ravdess_dataset,
    parse_ravdess_acquisition_settings,
)


def _settings(*, acquire: bool, archive: Path, size: int, md5: str) -> RavdessAcquisitionSettings:
    return RavdessAcquisitionSettings(
        source_url="test-record",
        release="test-release",
        root=".cache/ravdess/Audio_Speech_Actors_01-24",
        acquire=acquire,
        license="test-license",
        license_acknowledged=True,
        archive_url="https://example.invalid/ravdess.zip",
        archive_filename=archive.name,
        archive_size_bytes=size,
        archive_md5=md5,
        expected_archive_wavs=1440,
        expected_total_groups=96,
    )


def _write_complete_archive(path: Path, *, include_unsafe_path: bool = False) -> None:
    emotions = ("01", "02", "03", "04", "05", "06", "07", "08")
    wrote_unsafe_path = False
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for actor in range(1, 25):
            for statement in (1, 2):
                for repetition in (1, 2):
                    for emotion in emotions:
                        intensities = ("01",) if emotion == "01" else ("01", "02")
                        for intensity in intensities:
                            if include_unsafe_path and not wrote_unsafe_path:
                                filename = "../escape.wav"
                                wrote_unsafe_path = True
                            else:
                                filename = (
                                    "Audio_Speech_Actors_01-24/"
                                    f"Actor_{actor:02d}/03-01-{emotion}-{intensity}-"
                                    f"{statement:02d}-{repetition:02d}-{actor:02d}.wav"
                                )
                            archive.writestr(filename, b"")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # nosec B324: test fixture checksum


def test_missing_cache_requires_explicit_acquisition_opt_in(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(b"not-used")
    settings = _settings(
        acquire=False,
        archive=archive,
        size=archive.stat().st_size,
        md5=_md5(archive),
    )

    with pytest.raises(ContractError, match="set claim1.ravdess.acquire: true"):
        ensure_ravdess_dataset(repo=tmp_path, settings=settings)


def test_acquisition_extracts_only_verified_complete_archive(tmp_path: Path) -> None:
    source_archive = tmp_path / "fixture.zip"
    _write_complete_archive(source_archive)
    settings = _settings(
        acquire=True,
        archive=source_archive,
        size=source_archive.stat().st_size,
        md5=_md5(source_archive),
    )
    cache_archive = (
        tmp_path / ".cache" / "ravdess" / "archives" / source_archive.name
    )
    cache_archive.parent.mkdir(parents=True)
    source_archive.replace(cache_archive)

    root = ensure_ravdess_dataset(repo=tmp_path, settings=settings)

    assert root == tmp_path / ".cache" / "ravdess" / "Audio_Speech_Actors_01-24"
    assert len(list(root.rglob("*.wav"))) == 1440
    receipt = root.parent / f"{root.name}.receipt.json"
    assert receipt.is_file()
    # A second invocation verifies the existing immutable extraction and receipt without downloading.
    assert ensure_ravdess_dataset(repo=tmp_path, settings=settings) == root


def test_extraction_rejects_path_traversal_before_writing(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    _write_complete_archive(archive, include_unsafe_path=True)
    settings = _settings(
        acquire=True,
        archive=archive,
        size=archive.stat().st_size,
        md5=_md5(archive),
    )
    cache_archive = tmp_path / ".cache" / "ravdess" / "archives" / archive.name
    cache_archive.parent.mkdir(parents=True)
    archive.replace(cache_archive)

    with pytest.raises(ContractError, match="unsafe path"):
        ensure_ravdess_dataset(repo=tmp_path, settings=settings)
    assert not (tmp_path / "escape.wav").exists()


def _acquired_cache(tmp_path: Path) -> tuple[Path, RavdessAcquisitionSettings]:
    source_archive = tmp_path / "fixture.zip"
    _write_complete_archive(source_archive)
    settings = _settings(
        acquire=True,
        archive=source_archive,
        size=source_archive.stat().st_size,
        md5=_md5(source_archive),
    )
    cache_archive = tmp_path / ".cache" / "ravdess" / "archives" / source_archive.name
    cache_archive.parent.mkdir(parents=True)
    source_archive.replace(cache_archive)
    return ensure_ravdess_dataset(repo=tmp_path, settings=settings), settings


def test_structurally_valid_substituted_tree_is_rejected_by_receipt(tmp_path: Path) -> None:
    root, settings = _acquired_cache(tmp_path)
    substituted = next(root.rglob("*.wav"))
    substituted.write_bytes(b"different-but-structurally-valid")

    with pytest.raises(ContractError, match="does not match the pinned archive identity"):
        ensure_ravdess_dataset(repo=tmp_path, settings=settings)


def test_missing_or_tampered_receipt_rejects_existing_cache(tmp_path: Path) -> None:
    root, settings = _acquired_cache(tmp_path)
    receipt_path = root.parent / f"{root.name}.receipt.json"
    receipt_path.unlink()
    with pytest.raises(ContractError, match="no verified extraction receipt"):
        ensure_ravdess_dataset(repo=tmp_path, settings=settings)

    # Restore a syntactically valid but false receipt: structural checks alone must not accept it.
    receipt_path.write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="does not match the pinned archive identity"):
        ensure_ravdess_dataset(repo=tmp_path, settings=settings)


def test_official_config_rejects_altered_license_or_archive_pin() -> None:
    values = {
        "source_url": "https://zenodo.org/records/1188976",
        "release": "RAVDESS-v1.0-audio-speech",
        "root": ".cache/ravdess/Audio_Speech_Actors_01-24",
        "acquire": True,
        "license": "CC BY-NC-SA 4.0",
        "license_acknowledged": True,
        "archive_url": "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content",
        "archive_filename": "Audio_Speech_Actors_01-24.zip",
        "archive_size_bytes": 208468073,
        "archive_md5": "bc696df654c87fed845eb13823edef8a",
        "expected_archive_wavs": 1440,
        "expected_total_groups": 96,
    }
    parsed = parse_ravdess_acquisition_settings(values)
    assert parsed.acquire is True
    values["license"] = "CC BY 4.0"
    with pytest.raises(ContractError, match="official Zenodo"):
        parse_ravdess_acquisition_settings(values)
    values["license"] = "CC BY-NC-SA 4.0"
    values["release"] = "RAVDESS-v2"
    with pytest.raises(ContractError, match="official Zenodo"):
        parse_ravdess_acquisition_settings(values)
