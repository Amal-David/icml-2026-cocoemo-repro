"""Immutable private submission persistence for the listening study."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from study.protocol import ProtocolError, assert_private_directory, canonical_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def submission_path(storage_root: Path, study_id_hmac: str) -> Path:
    if len(study_id_hmac) != 64 or any(char not in "0123456789abcdef" for char in study_id_hmac):
        raise ProtocolError("study ID must be a SHA-256 HMAC")
    return storage_root / "submissions" / f"{study_id_hmac}.json"


def write_new_private_file(path: Path, payload: bytes) -> Path:
    """Create a new 0600 regular file without following a target symlink."""

    parent = path.parent
    assert_private_directory(parent)
    if path.name in {"", ".", ".."}:
        raise ProtocolError("private output file name is invalid")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ProtocolError("platform lacks required no-follow private-file support")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as exc:
        raise ProtocolError("private output parent cannot be opened safely") from exc
    descriptor: int | None = None
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
        created = True
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_mode & 0o077:
            raise ProtocolError("private output file permissions are unsafe")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolError("private output already exists") from exc
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(path.name, dir_fd=directory_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(directory_fd)
    return path


def write_immutable_submission(storage_root: Path, submission: Mapping[str, Any]) -> Path:
    """Create exactly one 0600 file, atomically; duplicate IDs are rejected."""

    assert_private_directory(storage_root)
    study_id_hmac = submission.get("study_id_hmac")
    if not isinstance(study_id_hmac, str):
        raise ProtocolError("submission has no HMAC study ID")
    output = submission_path(storage_root, study_id_hmac)
    output.parent.mkdir(mode=0o700, exist_ok=True)
    payload = dict(submission)
    payload["server_timestamp"] = utc_now()
    encoded = canonical_json(payload) + b"\n"
    try:
        return write_new_private_file(output, encoded)
    except ProtocolError as exc:
        if str(exc) == "private output already exists":
            raise ProtocolError("the earliest valid submission for this invitation is already stored") from exc
        raise


def load_submissions(storage_root: Path) -> list[dict[str, Any]]:
    assert_private_directory(storage_root)
    directory = storage_root / "submissions"
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ProtocolError("submission storage path is not a directory")
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.stat().st_mode & 0o077:
            raise ProtocolError(f"submission is not private: {path.name}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"submission cannot be read: {path.name}") from exc
        if not isinstance(value, dict):
            raise ProtocolError(f"submission is not an object: {path.name}")
        if submission_path(storage_root, str(value.get("study_id_hmac", ""))) != path:
            raise ProtocolError(f"submission file name does not match its HMAC: {path.name}")
        records.append(value)
    return records
