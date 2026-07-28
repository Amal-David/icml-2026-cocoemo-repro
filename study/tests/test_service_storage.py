from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from study.protocol import ProtocolError
from study.storage import load_submissions, write_immutable_submission


def _response(service, trial):
    dominant = trial.expected_dominant_emotion or "happy"
    return {
        "protocol_version": service.manifest["protocol_version"],
        "manifest_hash": service.frozen_manifest_hash,
        "trial_id": trial.trial_id,
        "audio_sha256": trial.audio_sha256,
        "playback_duration_ms": round(trial.duration_s * 1000),
        "naturalness_score": 4,
        "dominant_emotion": dominant,
        "emotion_allocation": {"angry": 0, "happy": 100, "sad": 0, "surprised": 0, "neutral": 0},
    }


def _complete(service, code: str):
    state = service.start(invitation_code=code, consent=True, adult=True)
    for _ in range(30):
        state = service.record_trial(state, _response(service, service.current_trial(state)))
    return state


def test_service_records_one_immutable_complete_submission(frozen_study) -> None:
    service = frozen_study["service"]
    state = _complete(service, frozen_study["codes"][0])
    path = service.finish(state, now_unix_ms=state["started_unix_ms"] + 8 * 60 * 1000)
    assert path.exists()
    records = load_submissions(frozen_study["storage_root"])
    assert len(records) == 1
    assert len(records[0]["trials"]) == 30
    assert records[0]["valid_under_preregistered_exclusions"] is True
    with pytest.raises(ProtocolError, match="earliest"):
        service.finish(state, now_unix_ms=state["started_unix_ms"] + 8 * 60 * 1000)


def test_service_retains_short_playback_and_applies_aggregate_exclusion_at_finish(frozen_study) -> None:
    service = frozen_study["service"]
    state = service.start(invitation_code=frozen_study["codes"][0], consent=True, adult=True)
    response = _response(service, service.current_trial(state))
    response["playback_duration_ms"] = 1
    state = service.record_trial(state, response)
    for _ in range(29):
        response = _response(service, service.current_trial(state))
        response["playback_duration_ms"] = 1
        state = service.record_trial(state, response)
    service.finish(state, now_unix_ms=state["started_unix_ms"] + 8 * 60 * 1000)
    record = load_submissions(frozen_study["storage_root"])[0]
    assert record["valid_under_preregistered_exclusions"] is False
    assert "fewer_than_22_experimental_locked_playbacks" in record["exclusion_reasons"]


def test_service_rejects_tampered_assignment_state(frozen_study) -> None:
    service = frozen_study["service"]
    state = service.start(invitation_code=frozen_study["codes"][0], consent=True, adult=True)
    state["group"] = 2
    with pytest.raises(ProtocolError, match="assignment"):
        service.current_trial(state)


def test_service_rejects_forged_trial_pointer_and_response_history(frozen_study) -> None:
    service = frozen_study["service"]
    state = service.start(invitation_code=frozen_study["codes"][0], consent=True, adult=True)
    forged_pointer = dict(state)
    forged_pointer["index"] = 9
    with pytest.raises(ProtocolError, match="integrity"):
        service.current_trial(forged_pointer)
    forged_history = dict(state)
    forged_history["responses"] = [{"trial_id": "experiment-forged"}]
    with pytest.raises(ProtocolError, match="integrity"):
        service.current_trial(forged_history)


def test_service_rejects_missing_consent(frozen_study) -> None:
    with pytest.raises(ProtocolError, match="consent"):
        frozen_study["service"].start(invitation_code=frozen_study["codes"][0], consent=False, adult=True)


def test_concurrent_private_writes_do_not_race_on_storage_probe(frozen_study) -> None:
    root = frozen_study["storage_root"]

    def write(index: int):
        return write_immutable_submission(root, {"study_id_hmac": f"{index:064x}"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(write, range(8)))
    assert len({path.name for path in paths}) == 8
    assert len(load_submissions(root)) == 8
