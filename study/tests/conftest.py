from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from study.create_manifest_template import template_manifest
from study.protocol import hmac_code
from study.service import StudyService


def _write_wav(path: Path) -> tuple[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)
    return hashlib.sha256(path.read_bytes()).hexdigest(), 0.1


@pytest.fixture()
def frozen_study(tmp_path: Path):
    audio_root = tmp_path / "audio"
    storage_root = tmp_path / "private"
    storage_root.mkdir(mode=0o700)
    manifest = template_manifest()
    for row in manifest["experimental_stimuli"]:
        relative = f"stimuli/{row['stimulus_id']}.wav"
        digest, duration = _write_wav(audio_root / relative)
        row["audio_path"] = relative
        row["audio_sha256"] = digest
        row["duration_s"] = duration
        if row["track"] == "mismatch":
            row["target_allocation"] = {"angry": 0, "happy": 100, "sad": 0, "surprised": 0, "neutral": 0}
    for row in manifest["quality_trials"]:
        if row["kind"] == "duplicate":
            continue
        relative = f"quality/{row['trial_id']}.wav"
        digest, duration = _write_wav(audio_root / relative)
        row["audio_path"] = relative
        row["audio_sha256"] = digest
        row["duration_s"] = duration
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    secret = "h" * 40
    order_secret = "o" * 40
    codes = []
    invitation_rows = []
    priority_commitment = hashlib.sha256(b"test-priority-seed").hexdigest()
    for group in range(3):
        for priority in range(12):
            code = f"group-{group}-priority-{priority}"
            codes.append(code)
            invitation_rows.append(
                {
                    "code_hmac": hmac_code(code, secret),
                    "group": group,
                    "priority": priority,
                    "priority_seed_commitment": priority_commitment,
                }
            )
    invitation_path = tmp_path / "invitations.json"
    invitation_path.write_text(json.dumps(invitation_rows), encoding="utf-8")
    service = StudyService(
        manifest=manifest,
        invitations={
            row["code_hmac"]: __import__("study.protocol", fromlist=["Invitation"]).Invitation(
                code_hmac=row["code_hmac"], group=row["group"], priority=row["priority"]
            )
            for row in invitation_rows
        },
        hmac_secret=secret,
        order_secret=order_secret,
        storage_root=storage_root,
    )
    return {
        "audio_root": audio_root,
        "storage_root": storage_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "invitation_path": invitation_path,
        "service": service,
        "codes": codes,
    }
