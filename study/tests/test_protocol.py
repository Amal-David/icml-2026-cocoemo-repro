from __future__ import annotations

import json
from pathlib import Path

import pytest

from study.protocol import ProtocolError, build_experimental_trials, build_quality_trials, load_manifest, manifest_hash


def test_frozen_media_manifest_validates_exactly(frozen_study) -> None:
    manifest = load_manifest(frozen_study["manifest_path"], audio_root=frozen_study["audio_root"])
    assert len(manifest["experimental_stimuli"]) == 72
    assert len(manifest["quality_trials"]) == 6
    for group in range(3):
        experimental = build_experimental_trials(manifest, group=group, study_id_hmac="a" * 64, order_secret="z" * 40)
        quality = build_quality_trials(manifest, group=group, study_id_hmac="a" * 64, order_secret="z" * 40)
        assert len(experimental) == 24
        assert len(quality) == 6
        assert all(trial.source_stimulus_id in {row.stimulus_id for row in experimental} for trial in quality if trial.kind == "duplicate")


def test_hidden_duplicate_is_always_after_its_source_for_every_invitation(frozen_study) -> None:
    from study.protocol import interleave_trials

    manifest = frozen_study["manifest"]
    for code_index, _ in enumerate(frozen_study["codes"]):
        group = code_index // 12
        study_id = f"{code_index:064x}"
        experimental = build_experimental_trials(manifest, group=group, study_id_hmac=study_id, order_secret="z" * 40)
        quality = build_quality_trials(manifest, group=group, study_id_hmac=study_id, order_secret="z" * 40)
        schedule = interleave_trials(experimental, quality)
        positions = {}
        for index, trial in enumerate(schedule):
            positions.setdefault(trial.stimulus_id, index)
        for index, trial in enumerate(schedule):
            if trial.kind == "duplicate":
                assert trial.source_stimulus_id is not None
                assert positions[trial.source_stimulus_id] < index


def test_changed_media_fails_closed(frozen_study) -> None:
    manifest = frozen_study["manifest"]
    path = frozen_study["audio_root"] / manifest["experimental_stimuli"][0]["audio_path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ProtocolError, match="hash"):
        load_manifest(frozen_study["manifest_path"], audio_root=frozen_study["audio_root"])


def test_template_has_full_structure_but_cannot_be_served(tmp_path: Path) -> None:
    from study.create_manifest_template import template_manifest
    from study.protocol import validate_manifest

    manifest = template_manifest()
    validate_manifest(manifest)
    with pytest.raises(ProtocolError, match="missing"):
        validate_manifest(manifest, audio_root=tmp_path)
    assert manifest_hash(manifest)


def test_manifest_rejects_prohibited_corpus_license(frozen_study) -> None:
    changed = json.loads(json.dumps(frozen_study["manifest"]))
    changed["experimental_stimuli"][0]["provenance"]["license_basis"] = "RAVDESS permission"
    with pytest.raises(ProtocolError, match="prohibited"):
        from study.protocol import validate_manifest

        validate_manifest(changed)
