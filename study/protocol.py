"""Frozen-protocol validation and deterministic trial construction.

The study app deliberately has no permissive mode.  A study can only open when
the manifest, its media, invitation-code hashes, and private storage pass the
same checks used by the offline tools.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from study.design import CONDITIONS, condition_for


PROTOCOL_VERSION = "cocoemo-listening-v1"
EMOTIONS = ("angry", "happy", "sad", "surprised", "neutral")
EXPERIMENTAL_COUNT = 72
QUALITY_COUNT = 6
PLAYBACK_MIN_FRACTION = 0.90
MIN_EXPERIMENTAL_PLAYED = 22
MIN_TOTAL_COMPLETION_SECONDS = 8 * 60
DUPLICATE_MAX_L1_DISTANCE = 40
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_HMAC_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_LICENSE_TERMS = ("esd", "iemocap", "crema", "ravdess")


class ProtocolError(ValueError):
    """Raised whenever a study input cannot meet the frozen protocol."""


@dataclass(frozen=True)
class Invitation:
    code_hmac: str
    group: int
    priority: int


@dataclass(frozen=True)
class Trial:
    trial_id: str
    kind: str
    item_id: str | None
    stimulus_id: str
    audio_path: str
    audio_sha256: str
    duration_s: float
    target_allocation: dict[str, int] | None
    expected_dominant_emotion: str | None
    source_stimulus_id: str | None = None


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(manifest))


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field} must be an object")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    value = _require_string(value, field)
    if not _SHA256_RE.fullmatch(value):
        raise ProtocolError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _validate_allocation(value: Any, field: str) -> dict[str, int]:
    allocation = _require_mapping(value, field)
    if set(allocation) != set(EMOTIONS):
        raise ProtocolError(f"{field} must contain exactly: {', '.join(EMOTIONS)}")
    normalized: dict[str, int] = {}
    for emotion in EMOTIONS:
        amount = allocation[emotion]
        if isinstance(amount, bool) or not isinstance(amount, int) or not 0 <= amount <= 100:
            raise ProtocolError(f"{field}.{emotion} must be an integer from 0 to 100")
        normalized[emotion] = amount
    if sum(normalized.values()) != 100:
        raise ProtocolError(f"{field} must sum to 100")
    return normalized


def _safe_audio_path(audio_root: Path, relative_path: str) -> Path:
    candidate = (audio_root / relative_path).resolve()
    try:
        candidate.relative_to(audio_root.resolve())
    except ValueError as exc:
        raise ProtocolError("audio_path must stay under the configured audio root") from exc
    return candidate


def _validate_audio_metadata(row: Mapping[str, Any], field: str) -> None:
    _require_string(row.get("audio_path"), f"{field}.audio_path")
    _require_sha256(row.get("audio_sha256"), f"{field}.audio_sha256")
    duration = row.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise ProtocolError(f"{field}.duration_s must be positive")


def _validate_provenance(row: Mapping[str, Any], field: str) -> None:
    provenance = _require_mapping(row.get("provenance"), f"{field}.provenance")
    for required in (
        "backbone",
        "model_revision",
        "vector_id",
        "vector_sha256",
        "reference_voice_id",
        "reference_voice_release_id",
        "generator_commit",
        "license_basis",
    ):
        _require_string(provenance.get(required), f"{field}.provenance.{required}")
    _require_sha256(provenance["vector_sha256"], f"{field}.provenance.vector_sha256")
    license_basis = provenance["license_basis"].casefold()
    if any(term in license_basis for term in _FORBIDDEN_LICENSE_TERMS):
        raise ProtocolError(f"{field}.provenance.license_basis names a prohibited evaluation corpus")
    if "consent" not in license_basis or "redistribution" not in license_basis:
        raise ProtocolError(f"{field}.provenance.license_basis must document consent and redistribution")


def validate_manifest(manifest: Mapping[str, Any], *, audio_root: Path | None = None) -> None:
    """Validate structure and, when supplied, hash/decode every frozen WAV.

    Any missing audio root, file, unsupported encoding, digest mismatch, or
    duration mismatch is fatal.  Media verification is intentionally not
    optional once a study is served.
    """

    _require_mapping(manifest, "manifest")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(f"manifest.protocol_version must equal {PROTOCOL_VERSION}")
    _require_string(manifest.get("manifest_version"), "manifest.manifest_version")
    _require_string(manifest.get("study_id"), "manifest.study_id")
    _require_string(manifest.get("generated_at"), "manifest.generated_at")
    experimental = manifest.get("experimental_stimuli")
    quality = manifest.get("quality_trials")
    if not isinstance(experimental, list) or len(experimental) != EXPERIMENTAL_COUNT:
        raise ProtocolError(f"manifest.experimental_stimuli must contain exactly {EXPERIMENTAL_COUNT} rows")
    if not isinstance(quality, list) or len(quality) != QUALITY_COUNT:
        raise ProtocolError(f"manifest.quality_trials must contain exactly {QUALITY_COUNT} rows")

    stimulus_ids: set[str] = set()
    rows_by_item: dict[str, list[Mapping[str, Any]]] = {}
    for index, row_value in enumerate(experimental):
        field = f"manifest.experimental_stimuli[{index}]"
        row = _require_mapping(row_value, field)
        stimulus_id = _require_string(row.get("stimulus_id"), f"{field}.stimulus_id")
        if stimulus_id in stimulus_ids:
            raise ProtocolError("experimental stimulus_id values must be unique")
        stimulus_ids.add(stimulus_id)
        item_id = _require_string(row.get("item_id"), f"{field}.item_id")
        track = row.get("track")
        if track not in {"mixed", "mismatch"}:
            raise ProtocolError(f"{field}.track must be mixed or mismatch")
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ProtocolError(f"{field}.condition must be one of {', '.join(CONDITIONS)}")
        _validate_audio_metadata(row, field)
        _validate_allocation(row.get("target_allocation"), f"{field}.target_allocation")
        _require_string(row.get("transcript"), f"{field}.transcript")
        alpha = row.get("alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha < 0:
            raise ProtocolError(f"{field}.alpha must be a non-negative number")
        _validate_provenance(row, field)
        rows_by_item.setdefault(item_id, []).append(row)

    if len(rows_by_item) != 24:
        raise ProtocolError("experimental stimuli must cover exactly 24 content families")
    if Counter(row["track"] for row in experimental) != Counter({"mixed": 36, "mismatch": 36}):
        raise ProtocolError("experimental stimuli must contain 36 mixed and 36 mismatch rows")
    for item_id, rows in rows_by_item.items():
        if len(rows) != 3 or {row["condition"] for row in rows} != set(CONDITIONS):
            raise ProtocolError(f"item {item_id} must have exactly one row per condition")
        if len({row["track"] for row in rows}) != 1:
            raise ProtocolError(f"item {item_id} must have one track")

    quality_ids: set[str] = set()
    quality_kinds = Counter()
    for index, row_value in enumerate(quality):
        field = f"manifest.quality_trials[{index}]"
        row = _require_mapping(row_value, field)
        trial_id = _require_string(row.get("trial_id"), f"{field}.trial_id")
        if trial_id in quality_ids or trial_id in stimulus_ids:
            raise ProtocolError("quality trial identifiers must be unique and disjoint from stimuli")
        quality_ids.add(trial_id)
        kind = row.get("kind")
        if kind not in {"instruction", "duplicate", "calibration"}:
            raise ProtocolError(f"{field}.kind must be instruction, duplicate, or calibration")
        quality_kinds[kind] += 1
        if kind == "duplicate":
            source_item = _require_string(row.get("source_item_id"), f"{field}.source_item_id")
            if source_item not in rows_by_item:
                raise ProtocolError(f"{field}.source_item_id must reference an experimental content family")
        else:
            _validate_audio_metadata(row, field)
            _validate_provenance(row, field)
            expected = row.get("expected_dominant_emotion")
            if expected not in EMOTIONS:
                raise ProtocolError(f"{field}.expected_dominant_emotion must be a locked emotion")
    if quality_kinds != Counter({"instruction": 2, "duplicate": 2, "calibration": 2}):
        raise ProtocolError("quality trials must contain two instruction, duplicate, and calibration rows")

    if audio_root is not None:
        for index, row in enumerate(experimental):
            _verify_audio(row, audio_root, f"manifest media row {index}")
        for index, row in enumerate(quality):
            if row["kind"] != "duplicate":
                _verify_audio(row, audio_root, f"manifest quality media row {index}")


def _verify_audio(row: Mapping[str, Any], audio_root: Path, field: str) -> None:
    path = _safe_audio_path(audio_root, str(row["audio_path"]))
    if not path.is_file():
        raise ProtocolError(f"{field} is missing: {row['audio_path']}")
    actual_hash = sha256_bytes(path.read_bytes())
    if not hmac.compare_digest(actual_hash, str(row["audio_sha256"])):
        raise ProtocolError(f"{field} hash does not match the frozen manifest")
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ProtocolError(f"{field} must be an uncompressed WAV")
            actual_duration = wav.getnframes() / wav.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ProtocolError(f"{field} is not a decodable WAV") from exc
    if abs(actual_duration - float(row["duration_s"])) > 0.010:
        raise ProtocolError(f"{field} duration does not match the frozen manifest")


def load_manifest(path: Path, *, audio_root: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("manifest cannot be read as JSON") from exc
    if not isinstance(data, dict):
        raise ProtocolError("manifest root must be an object")
    validate_manifest(data, audio_root=audio_root)
    return data


def hmac_code(code: str, secret: str) -> str:
    if not secret:
        raise ProtocolError("invitation-code HMAC secret is missing")
    normalized = code.strip()
    if not normalized:
        raise ProtocolError("invitation code is required")
    return hmac.new(secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def load_invitations(path: Path) -> dict[str, Invitation]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("invitation-code hash file cannot be read as JSON") from exc
    if not isinstance(data, list) or not data:
        raise ProtocolError("invitation-code hash file must be a non-empty JSON list")
    invitations: dict[str, Invitation] = {}
    by_group = Counter()
    priorities_by_group: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
    commitments: set[str] = set()
    commitment_count = 0
    for index, value in enumerate(data):
        row = _require_mapping(value, f"invitations[{index}]")
        digest = _require_string(row.get("code_hmac"), f"invitations[{index}].code_hmac")
        if not _HEX_HMAC_RE.fullmatch(digest) or digest in invitations:
            raise ProtocolError("invitation code hashes must be unique SHA-256 HMAC hex values")
        group = row.get("group")
        priority = row.get("priority")
        if group not in {0, 1, 2}:
            raise ProtocolError("invitation groups must be 0, 1, or 2")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise ProtocolError("invitation priorities must be non-negative integers")
        invitations[digest] = Invitation(digest, group, priority)
        by_group[group] += 1
        if priority in priorities_by_group[group]:
            raise ProtocolError("invitation priorities must be unique within a group")
        priorities_by_group[group].add(priority)
        commitment = row.get("priority_seed_commitment")
        if commitment is not None:
            if not isinstance(commitment, str) or not _SHA256_RE.fullmatch(commitment):
                raise ProtocolError("priority seed commitment must be a SHA-256 hex digest")
            commitments.add(commitment)
            commitment_count += 1
    if any(by_group[group] < 12 for group in range(3)):
        raise ProtocolError("the frozen recruitment plan requires at least 12 invitation codes per group")
    if any(priorities_by_group[group] != set(range(by_group[group])) for group in range(3)):
        raise ProtocolError("invitation priorities must be complete locked ranks within each group")
    if commitment_count != len(invitations) or len(commitments) != 1:
        raise ProtocolError("every invitation must carry one identical priority seed commitment")
    return invitations


def _order_key(*, secret: str, study_id_hmac: str, value: str) -> str:
    return hmac.new(secret.encode("utf-8"), f"{study_id_hmac}:{value}".encode("utf-8"), hashlib.sha256).hexdigest()


def build_experimental_trials(
    manifest: Mapping[str, Any], *, group: int, study_id_hmac: str, order_secret: str
) -> list[Trial]:
    rows = _require_mapping(manifest, "manifest").get("experimental_stimuli")
    if not isinstance(rows, list):
        raise ProtocolError("manifest has no experimental stimuli")
    by_item: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        row_map = _require_mapping(row, "experimental stimulus")
        by_item.setdefault(str(row_map["item_id"]), {})[str(row_map["condition"])] = row_map
    item_ids = sorted(by_item)
    if len(item_ids) != 24:
        raise ProtocolError("manifest does not have 24 content families")
    trials: list[Trial] = []
    for item_index, item_id in enumerate(item_ids):
        condition = condition_for(group=group, item_index=item_index)
        row = by_item[item_id].get(condition)
        if row is None:
            raise ProtocolError(f"manifest lacks {condition} for {item_id}")
        trials.append(
            Trial(
                trial_id=f"experiment-{item_id}",
                kind="experimental",
                item_id=item_id,
                stimulus_id=str(row["stimulus_id"]),
                audio_path=str(row["audio_path"]),
                audio_sha256=str(row["audio_sha256"]),
                duration_s=float(row["duration_s"]),
                target_allocation=dict(row["target_allocation"]),
                expected_dominant_emotion=None,
            )
        )
    trials.sort(key=lambda trial: _order_key(secret=order_secret, study_id_hmac=study_id_hmac, value=trial.trial_id))
    return trials


def build_quality_trials(
    manifest: Mapping[str, Any], *, group: int, study_id_hmac: str, order_secret: str
) -> list[Trial]:
    quality = _require_mapping(manifest, "manifest").get("quality_trials")
    if not isinstance(quality, list):
        raise ProtocolError("manifest has no quality trials")
    experimental = build_experimental_trials(
        manifest, group=group, study_id_hmac=study_id_hmac, order_secret=order_secret
    )
    experimental_by_item = {trial.item_id: trial for trial in experimental}
    trials: list[Trial] = []
    for row in quality:
        kind = str(row["kind"])
        if kind == "duplicate":
            source_item_id = str(row["source_item_id"])
            source = experimental_by_item[source_item_id]
            trials.append(
                Trial(
                    trial_id=str(row["trial_id"]),
                    kind=kind,
                    item_id=None,
                    stimulus_id=source.stimulus_id,
                    audio_path=source.audio_path,
                    audio_sha256=source.audio_sha256,
                    duration_s=source.duration_s,
                    target_allocation=None,
                    expected_dominant_emotion=None,
                    source_stimulus_id=source.stimulus_id,
                )
            )
            continue
        trials.append(
            Trial(
                trial_id=str(row["trial_id"]),
                kind=kind,
                item_id=None,
                stimulus_id=str(row["trial_id"]),
                audio_path=str(row["audio_path"]),
                audio_sha256=str(row["audio_sha256"]),
                duration_s=float(row["duration_s"]),
                target_allocation=None,
                expected_dominant_emotion=str(row["expected_dominant_emotion"]),
            )
        )
    trials.sort(key=lambda trial: _order_key(secret=order_secret, study_id_hmac=study_id_hmac, value=trial.trial_id))
    return trials


def interleave_trials(experimental: Iterable[Trial], quality: Iterable[Trial]) -> list[Trial]:
    """Insert quality items deterministically without exposing a duplicate first."""

    experiments = list(experimental)
    quality_trials = list(quality)
    if len(experiments) != 24 or len(quality_trials) != 6:
        raise ProtocolError("a participant schedule must contain 24 experimental and 6 quality trials")
    duplicates_by_source: dict[str, list[Trial]] = {}
    nonduplicates: list[Trial] = []
    for trial in quality_trials:
        if trial.kind == "duplicate":
            if trial.source_stimulus_id is None:
                raise ProtocolError("duplicate quality trial has no source stimulus")
            duplicates_by_source.setdefault(trial.source_stimulus_id, []).append(trial)
        else:
            nonduplicates.append(trial)
    schedule: list[Trial] = []
    nonduplicate_index = 0
    # Four non-duplicate checks are distributed through the experiment order.
    # Duplicate checks are emitted immediately after their exact source, making
    # their hidden-repeat validity independent of the random experiment order.
    for experiment_index, experiment in enumerate(experiments, start=1):
        schedule.append(experiment)
        schedule.extend(duplicates_by_source.pop(experiment.stimulus_id, []))
        if experiment_index in {4, 10, 16, 22}:
            schedule.append(nonduplicates[nonduplicate_index])
            nonduplicate_index += 1
    if duplicates_by_source or nonduplicate_index != len(nonduplicates):
        raise ProtocolError("quality schedule could not be placed after its source")
    if len(schedule) != 30:
        raise ProtocolError("a participant schedule must contain exactly 30 trials")
    return schedule


def validate_response(
    response: Mapping[str, Any], *, trial: Trial, protocol_version: str, expected_manifest_hash: str
) -> dict[str, Any]:
    """Return a normalized response or fail without retaining malformed data."""

    forbidden = {"name", "email", "ip", "user_agent", "microphone", "free_text", "payment_id"}
    supplied_forbidden = forbidden.intersection(response)
    if supplied_forbidden:
        raise ProtocolError(f"prohibited response fields: {', '.join(sorted(supplied_forbidden))}")
    if response.get("protocol_version") != protocol_version:
        raise ProtocolError("response protocol version does not match the frozen protocol")
    if response.get("manifest_hash") != expected_manifest_hash:
        raise ProtocolError("response manifest hash does not match the frozen manifest")
    if response.get("trial_id") != trial.trial_id:
        raise ProtocolError("response trial does not match the assigned schedule")
    if response.get("audio_sha256") != trial.audio_sha256:
        raise ProtocolError("response audio hash does not match the assigned schedule")
    naturalness = response.get("naturalness_score")
    if isinstance(naturalness, bool) or not isinstance(naturalness, int) or not 1 <= naturalness <= 5:
        raise ProtocolError("naturalness_score must be an integer from 1 to 5")
    dominant = response.get("dominant_emotion")
    if dominant not in EMOTIONS:
        raise ProtocolError("dominant_emotion must be a locked emotion")
    allocation = _validate_allocation(response.get("emotion_allocation"), "emotion_allocation")
    playback_ms = response.get("playback_duration_ms")
    if isinstance(playback_ms, bool) or not isinstance(playback_ms, int):
        raise ProtocolError("playback_duration_ms must be an integer")
    if playback_ms < 0:
        raise ProtocolError("playback_duration_ms must not be negative")
    return {
        "protocol_version": protocol_version,
        "manifest_hash": expected_manifest_hash,
        "trial_id": trial.trial_id,
        "audio_sha256": trial.audio_sha256,
        "playback_duration_ms": playback_ms,
        "naturalness_score": naturalness,
        "dominant_emotion": dominant,
        "emotion_allocation": allocation,
    }


def validate_completed_submission(
    *,
    study_id_hmac: str,
    group: int,
    priority: int,
    schedule: Iterable[Trial],
    responses: Iterable[Mapping[str, Any]],
    expected_manifest_hash: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build a complete response object and attach only predeclared flags.

    This function never selects retained participants or deletes a response.
    Retention priority is an allocation fact, while every exclusion reason is
    retained in the immutable record for later reporting.
    """

    if group not in {0, 1, 2} or priority < 0:
        raise ProtocolError("assigned invitation group or priority is invalid")
    schedule_rows = list(schedule)
    supplied = list(responses)
    if len(schedule_rows) != 30 or len(supplied) != 30:
        raise ProtocolError("a complete submission requires exactly 30 assigned trial responses")
    expected_by_id = {trial.trial_id: trial for trial in schedule_rows}
    if len(expected_by_id) != 30:
        raise ProtocolError("assigned schedule contains duplicate trial identifiers")
    normalized: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for response in supplied:
        trial_id = response.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in observed_ids or trial_id not in expected_by_id:
            raise ProtocolError("responses must cover every assigned trial exactly once")
        observed_ids.add(trial_id)
        normalized.append(
            validate_response(
                response,
                trial=expected_by_id[trial_id],
                protocol_version=PROTOCOL_VERSION,
                expected_manifest_hash=expected_manifest_hash,
            )
        )
    if observed_ids != set(expected_by_id):
        raise ProtocolError("responses do not cover the assigned schedule")

    by_trial = {response["trial_id"]: response for response in normalized}
    experiments = [trial for trial in schedule_rows if trial.kind == "experimental"]
    exclusions: list[str] = []
    played_experiments = sum(
        by_trial[trial.trial_id]["playback_duration_ms"] >= round(trial.duration_s * 1000 * PLAYBACK_MIN_FRACTION)
        for trial in experiments
    )
    if played_experiments < MIN_EXPERIMENTAL_PLAYED:
        exclusions.append("fewer_than_22_experimental_locked_playbacks")
    if elapsed_seconds < MIN_TOTAL_COMPLETION_SECONDS:
        exclusions.append("completion_below_locked_minimum")

    attention: dict[str, Any] = {
        "played_experimental": played_experiments,
        "instruction": [],
        "calibration": [],
        "duplicate": [],
    }
    for trial in schedule_rows:
        if trial.kind not in {"instruction", "calibration"}:
            continue
        passed = by_trial[trial.trial_id]["dominant_emotion"] == trial.expected_dominant_emotion
        attention[trial.kind].append({"trial_id": trial.trial_id, "passed": passed})
        if trial.kind == "instruction" and not passed:
            exclusions.append("explicit_instruction_incorrect")

    for trial in schedule_rows:
        if trial.kind != "duplicate" or trial.source_stimulus_id is None:
            continue
        source_trial = next((row for row in experiments if row.stimulus_id == trial.source_stimulus_id), None)
        if source_trial is None:
            raise ProtocolError("duplicate trial source is absent from this participant schedule")
        duplicate = by_trial[trial.trial_id]
        original = by_trial[source_trial.trial_id]
        l1_distance = sum(
            abs(duplicate["emotion_allocation"][emotion] - original["emotion_allocation"][emotion])
            for emotion in EMOTIONS
        )
        dominant_matches = duplicate["dominant_emotion"] == original["dominant_emotion"]
        passed = dominant_matches and l1_distance <= DUPLICATE_MAX_L1_DISTANCE
        attention["duplicate"].append(
            {
                "trial_id": trial.trial_id,
                "source_trial_id": source_trial.trial_id,
                "dominant_matches": dominant_matches,
                "allocation_l1_distance": l1_distance,
                "passed": passed,
            }
        )
        if not passed:
            exclusions.append("duplicate_inconsistent")
    if len(attention["instruction"]) != 2 or len(attention["duplicate"]) != 2 or len(attention["calibration"]) != 2:
        raise ProtocolError("quality schedule does not contain all required quality checks")

    return {
        "study_id_hmac": study_id_hmac,
        "protocol_version": PROTOCOL_VERSION,
        "manifest_hash": expected_manifest_hash,
        "group": group,
        "priority": priority,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "trials": normalized,
        "attention_results": attention,
        "exclusion_reasons": sorted(set(exclusions)),
        "valid_under_preregistered_exclusions": not exclusions,
    }


def assert_private_directory(path: Path) -> None:
    """Require a real private mount; world-readable output fails closed."""

    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise ProtocolError("private study storage directory is missing") from exc
    if not stat.S_ISDIR(stat_result.st_mode):
        raise ProtocolError("private study storage directory is missing")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ProtocolError("private study storage must not be group/world accessible")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ProtocolError("platform lacks required no-follow private-file support")
    probe = path / f".study_write_probe.{secrets.token_hex(16)}"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(probe, flags, 0o600)
        os.close(descriptor)
        probe.unlink()
    except OSError as exc:
        raise ProtocolError("private study storage is not writable") from exc
