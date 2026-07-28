"""Strict GPU runtime contracts for the bounded public Claim 4 proxy.

The adapters deliberately fail closed.  Unit tests cover the pure protocol;
the first real execution must additionally serve as a model/API canary before
any claim verdict is interpreted as empirical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from repro.claim4_evaluation import build_claim4_arms
from repro.claim4_protocol import (
    CLAIM4_ARM_NAMES,
    FrozenClaim4Asset,
    LfsPointer,
    build_claim4_manifest,
    claim4_asset_for_filename,
    ensure_lfs_wav,
    fetch_url_bytes,
    load_frozen_claim4_asset_manifest,
    require_claim4_asset_coverage,
    write_manifest,
)
from repro.config import ReproConfig, load_config
from repro.contracts import ContractError, require_complete_counts, sha256_file
from repro.cremad_selection import load_audio_vote_rows
from repro.hub_artifacts import ARTIFACT_SCHEMA_VERSION, artifact_destination
from repro.claim4_protocol import cremad_actor_id, is_claim4_eligible


_EMOTIONS = ("angry", "happy", "sad", "surprise")
_CANONICAL_EMOTION_LABELS = ("angry", "happy", "neutral", "sad", "surprise")
_CREMAD_TRANSCRIPTS = {
    "DFA": "Don't forget a jacket.",
    "IEO": "It's eleven o'clock.",
    "IOM": "I'm on my way to the meeting.",
    "ITS": "I think I've seen this before.",
    "ITH": "I think I have a doctor's appointment.",
    "IWL": "I would like a new alarm clock.",
    "IWW": "I wonder what this is about.",
    "MTI": "Maybe tomorrow it will be cold.",
    "TAI": "The airplane is almost full.",
    "TIE": "That is exactly what happened.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}


@dataclass(frozen=True)
class Claim4Settings:
    mode: str
    source: dict[str, Any]
    actor_count: int
    alpha: float
    expected_samples: int
    expected_rendered_wavs: int
    vector_contract: dict[str, Any]
    model: dict[str, Any]
    audio_quality: dict[str, Any]
    evaluators: dict[str, Any]
    decisions: dict[str, Any]
    license_attestations: dict[str, Any]
    canary_prerequisite: dict[str, Any]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a mapping")
    return value


def _require_exact_keys(values: Mapping[str, Any], *, label: str, keys: set[str]) -> None:
    actual = set(values)
    if actual != keys:
        raise ContractError(f"{label} keys changed: missing={sorted(keys - actual)}, unexpected={sorted(actual - keys)}")


def _require_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _require_finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ContractError(f"{label} must be a finite numeric value")
    return float(value)


_CLAIM4_LICENSE_COMPONENTS = (
    "cremad_source", "cosyvoice2_model", "emotion2vec", "wavlm", "whisper",
)


def _safe_relative_path(value: object, *, label: str, suffix: str | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ContractError(f"{label} must be a safe relative path")
    if suffix is not None and path.suffix != suffix:
        raise ContractError(f"{label} must end in {suffix}")
    return path


def _require_https_url(value: object, *, label: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"https://[^\s?#]+(?:[?#][^\s]*)?", value):
        raise ContractError(f"{label} must be an https URL")
    if suffix is not None and not value.split("?", 1)[0].split("#", 1)[0].endswith(suffix):
        raise ContractError(f"{label} must end in {suffix}")
    return value


def parse_claim4_settings(config: ReproConfig) -> Claim4Settings:
    canary = config.raw.get("claim4_canary")
    if canary is not None:
        return _parse_claim4_canary_settings(config, canary)
    values = _mapping(config.raw.get("claim4"), "claim4")
    _require_exact_keys(
        values,
        label="claim4",
        keys={
            "protocol", "mode", "source", "selection", "arms", "expected_samples", "expected_rendered_wavs",
            "alpha", "vector_contract", "model", "audio_quality", "evaluators", "decisions",
            "license_attestations", "canary_prerequisite",
        },
    )
    if values["protocol"] != "cremad_mixed_directional_public_proxy_v1":
        raise ContractError("claim4.protocol must be the frozen public proxy protocol")
    mode = values["mode"]
    if mode not in {"full", "canary"}:
        raise ContractError("claim4.mode must be 'full' or 'canary'")
    source = _mapping(values["source"], "claim4.source")
    _require_exact_keys(
        source,
        label="claim4.source",
        keys={
            "revision", "votes_url", "votes_sha256", "asset_manifest_path", "asset_manifest_sha256",
            "expected_audio_only_rows", "expected_eligible_rows", "expected_eligible_actors",
        },
    )
    if not all(isinstance(source[key], str) and source[key] for key in (
        "revision", "votes_url", "votes_sha256", "asset_manifest_path", "asset_manifest_sha256"
    )) or len(source["votes_sha256"]) != 64 or len(source["asset_manifest_sha256"]) != 64:
        raise ContractError("claim4.source must pin the vote table and frozen asset manifest SHA-256 hashes")
    _safe_relative_path(
        source["asset_manifest_path"], label="claim4.source.asset_manifest_path", suffix=".json"
    )
    for key in ("expected_audio_only_rows", "expected_eligible_rows", "expected_eligible_actors"):
        _require_positive_int(source[key], f"claim4.source.{key}")

    selection = _mapping(values["selection"], "claim4.selection")
    _require_exact_keys(selection, label="claim4.selection", keys={"actor_count", "one_item_per_actor", "eligible_rule", "neutral_reference_rule"})
    actor_count = _require_positive_int(selection["actor_count"], "claim4.selection.actor_count")
    if actor_count != 60 or selection["one_item_per_actor"] is not True or selection["eligible_rule"] != "supported_disagreement_and_positive_neutral_and_two_active_non_neutral_and_unique_non_neutral_maximum":
        raise ContractError("Claim 4 selection must remain exactly the frozen 60-actor one-item public proxy")
    if selection["neutral_reference_rule"] != "provided_neutral_label_and_unique_neutral_majority_vote":
        raise ContractError("Claim 4 neutral reference rule changed")

    arms = values["arms"]
    if not isinstance(arms, list) or tuple(arms) != CLAIM4_ARM_NAMES:
        raise ContractError("claim4.arms must list exactly the seven frozen arms in protocol order")
    expected_samples = _require_positive_int(values["expected_samples"], "claim4.expected_samples")
    expected_wavs = _require_positive_int(values["expected_rendered_wavs"], "claim4.expected_rendered_wavs")
    expected_mode_samples = actor_count if mode == "full" else 1
    if expected_samples != expected_mode_samples or expected_wavs != expected_mode_samples * len(CLAIM4_ARM_NAMES):
        raise ContractError("Claim 4 expected sample/WAV accounting does not match its execution mode")
    alpha = _require_finite_number(values["alpha"], "claim4.alpha")
    if not 0.0 < alpha <= 6.0:
        raise ContractError("claim4.alpha must be in (0, 6]")

    vector_contract = _mapping(values["vector_contract"], "claim4.vector_contract")
    _require_exact_keys(vector_contract, label="claim4.vector_contract", keys={"backbone", "operation", "layer", "shape", "paths", "sha256"})
    if vector_contract["backbone"] != "cosyvoice2" or vector_contract["operation"] != "attn_output" or vector_contract["layer"] != 17 or vector_contract["shape"] != [1, 896]:
        raise ContractError("Claim 4 vector contract must pin CosyVoice2 attn_output layer 17 shape [1, 896]")
    paths = _mapping(vector_contract["paths"], "claim4.vector_contract.paths")
    hashes = _mapping(vector_contract["sha256"], "claim4.vector_contract.sha256")
    if tuple(paths) != _EMOTIONS or tuple(hashes) != _EMOTIONS or any(
        not isinstance(paths[emotion], str) or not isinstance(hashes[emotion], str) or len(hashes[emotion]) != 64
        for emotion in _EMOTIONS
    ):
        raise ContractError("Claim 4 vector paths and SHA-256 hashes must cover every released emotion in order")

    model = _mapping(values["model"], "claim4.model")
    _require_exact_keys(model, label="claim4.model", keys={"id", "revision", "source_url", "source_revision", "required_api"})
    if model != {
        "id": "FunAudioLLM/CosyVoice2-0.5B",
        "revision": "eec1ae6c79877dbd9379285cf8789c9e0879293d",
        "source_url": "https://github.com/FunAudioLLM/CosyVoice.git",
        "source_revision": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
        "required_api": "cocoemo.backbones.cosyvoice2.generate_steered_speech",
    }:
        raise ContractError("Claim 4 model pin/API contract changed")

    audio_quality = _mapping(values["audio_quality"], "claim4.audio_quality")
    _require_exact_keys(audio_quality, label="claim4.audio_quality", keys={"max_clipping_fraction", "sample_rate_hz"})
    clipping = _require_finite_number(audio_quality["max_clipping_fraction"], "claim4.audio_quality.max_clipping_fraction")
    if not 0.0 <= clipping < 1.0 or audio_quality["sample_rate_hz"] != 24000:
        raise ContractError("Claim 4 audio integrity contract changed")

    evaluators = _mapping(values["evaluators"], "claim4.evaluators")
    _require_exact_keys(evaluators, label="claim4.evaluators", keys={"emotion2vec", "wavlm", "whisper"})
    emotion2vec = _mapping(evaluators["emotion2vec"], "claim4.evaluators.emotion2vec")
    _require_exact_keys(emotion2vec, label="claim4.evaluators.emotion2vec", keys={"id", "revision", "canonical_labels", "raw_label_map"})
    if emotion2vec["id"] != "emotion2vec/emotion2vec_plus_large" or emotion2vec["revision"] != "6c303ba987b86b93193de93e34bb2b077a6bedc4" or tuple(emotion2vec["canonical_labels"]) != _CANONICAL_EMOTION_LABELS:
        raise ContractError("Claim 4 Emotion2Vec pin/label contract changed")
    raw_map = _mapping(emotion2vec["raw_label_map"], "claim4.evaluators.emotion2vec.raw_label_map")
    if set(raw_map.values()) != set(_CANONICAL_EMOTION_LABELS):
        raise ContractError("Claim 4 Emotion2Vec raw label map must cover exactly the five canonical labels")
    wavlm = _mapping(evaluators["wavlm"], "claim4.evaluators.wavlm")
    if wavlm != {"id": "microsoft/wavlm-base-sv", "revision": "0a23162ffc49adcf42bdf836a00cb2eb45af3601", "metric": "speaker_similarity_against_source_reference"}:
        raise ContractError("Claim 4 WavLM pin/metric contract changed")
    whisper = _mapping(evaluators["whisper"], "claim4.evaluators.whisper")
    if whisper != {"id": "openai/whisper-large-v3", "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1", "language": "en", "metric": "word_error_rate_against_frozen_cremad_transcript"}:
        raise ContractError("Claim 4 Whisper pin/metric contract changed")

    decisions = _mapping(values["decisions"], "claim4.decisions")
    _require_exact_keys(decisions, label="claim4.decisions", keys={"bootstrap_replicates", "bootstrap_seed", "min_rho_advantage", "min_h_rate_advantage", "min_rho_vs_dominant", "max_wer_increase", "min_speaker_similarity_change"})
    if _require_positive_int(decisions["bootstrap_replicates"], "claim4.decisions.bootstrap_replicates") != 10_000 or _require_positive_int(decisions["bootstrap_seed"], "claim4.decisions.bootstrap_seed") != 20260728:
        raise ContractError("Claim 4 must use the predeclared 10k actor-cluster bootstrap")
    if (
        _require_finite_number(decisions["min_rho_advantage"], "claim4.decisions.min_rho_advantage") != 0.0
        or _require_finite_number(decisions["min_h_rate_advantage"], "claim4.decisions.min_h_rate_advantage") != 0.0
        or _require_finite_number(decisions["min_rho_vs_dominant"], "claim4.decisions.min_rho_vs_dominant") != -0.02
        or _require_finite_number(decisions["max_wer_increase"], "claim4.decisions.max_wer_increase") != 0.02
        or _require_finite_number(decisions["min_speaker_similarity_change"], "claim4.decisions.min_speaker_similarity_change") != -0.02
    ):
        raise ContractError("Claim 4 decision thresholds changed")

    license_attestations = _mapping(values["license_attestations"], "claim4.license_attestations")
    _require_exact_keys(
        license_attestations,
        label="claim4.license_attestations",
        keys={"schema_version", "public_artifact_policy", "components"},
    )
    if license_attestations["schema_version"] != "claim4_license_attestations_v1":
        raise ContractError("Claim 4 license-attestation schema changed")
    public_artifact_policy = _mapping(
        license_attestations["public_artifact_policy"], "claim4.license_attestations.public_artifact_policy"
    )
    if public_artifact_policy != {
        "source_audio": "excluded",
        "generated_audio": "excluded",
        "model_weights": "excluded",
        "public_evidence": "derived_metrics_and_hashes_only",
        "generated_audio_redistribution": "not_relied_on",
    }:
        raise ContractError("Claim 4 public artifact and redistribution policy changed")
    components = _mapping(license_attestations["components"], "claim4.license_attestations.components")
    if tuple(components) != _CLAIM4_LICENSE_COMPONENTS:
        raise ContractError("Claim 4 license attestations must cover each frozen component in order")
    expected_components = {
        "cremad_source": {
            "pinned_identifier": f"https://github.com/CheyneyComputerScience/CREMA-D/tree/{source['revision']}",
            "license_identifier": f"https://github.com/CheyneyComputerScience/CREMA-D/blob/{source['revision']}/LICENSE.txt",
            "license_expression": "ODbL-1.0 database; DbCL-1.0 contents",
            "acknowledged": True,
            "redistribution": "not_redistributed_by_this_reproduction",
        },
        "cosyvoice2_model": {
            "pinned_identifier": f"https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/tree/{model['revision']}",
            "license_identifier": f"https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/tree/{model['revision']}",
            "license_expression": "Apache-2.0",
            "acknowledged": True,
            "redistribution": "not_redistributed_by_this_reproduction",
        },
        "emotion2vec": {
            "pinned_identifier": f"https://huggingface.co/emotion2vec/emotion2vec_plus_large/tree/{emotion2vec['revision']}",
            "license_identifier": f"https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/{emotion2vec['revision']}/README.md",
            "license_expression": "model-license",
            "acknowledged": True,
            "redistribution": "not_redistributed_by_this_reproduction",
        },
        "wavlm": {
            "pinned_identifier": f"https://huggingface.co/microsoft/wavlm-base-sv/tree/{wavlm['revision']}",
            "license_identifier": f"https://huggingface.co/microsoft/wavlm-base-sv/blob/{wavlm['revision']}/README.md#license",
            "license_expression": "official_license_linked_by_pinned_model_card",
            "acknowledged": True,
            "redistribution": "not_redistributed_by_this_reproduction",
        },
        "whisper": {
            "pinned_identifier": f"https://huggingface.co/openai/whisper-large-v3/tree/{whisper['revision']}",
            "license_identifier": f"https://huggingface.co/openai/whisper-large-v3/tree/{whisper['revision']}",
            "license_expression": "Apache-2.0",
            "acknowledged": True,
            "redistribution": "not_redistributed_by_this_reproduction",
        },
    }
    if components != expected_components:
        raise ContractError("Claim 4 frozen license identifiers or redistribution attestations changed")

    canary_prerequisite = _mapping(values["canary_prerequisite"], "claim4.canary_prerequisite")
    _require_exact_keys(
        canary_prerequisite,
        label="claim4.canary_prerequisite",
        keys={"receipt_path", "canary_config_path", "required_status"},
    )
    _safe_relative_path(canary_prerequisite["receipt_path"], label="claim4.canary_prerequisite.receipt_path", suffix=".json")
    _safe_relative_path(canary_prerequisite["canary_config_path"], label="claim4.canary_prerequisite.canary_config_path", suffix=".yaml")
    if canary_prerequisite["required_status"] != "complete":
        raise ContractError("Claim 4 full run must require a completed canary receipt")

    evidence = _mapping(config.raw.get("evidence"), "evidence")
    if evidence.get("claim_evidence") is not (mode == "full"):
        raise ContractError("Claim 4 evidence flag must match its execution mode")
    return Claim4Settings(
        mode, source, actor_count, alpha, expected_samples, expected_wavs, vector_contract, model,
        audio_quality, evaluators, decisions, license_attestations, canary_prerequisite,
    )


def _parse_claim4_canary_settings(config: ReproConfig, value: object) -> Claim4Settings:
    canary = _mapping(value, "claim4_canary")
    _require_exact_keys(
        canary,
        label="claim4_canary",
        keys={"base_config", "base_config_sha256", "expected_samples", "expected_rendered_wavs"},
    )
    base_name = canary["base_config"]
    expected_hash = canary["base_config_sha256"]
    if not isinstance(base_name, str) or Path(base_name).name != base_name or not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ContractError("Claim 4 canary must pin a sibling full-config filename and SHA-256")
    base_path = config.source.parent / base_name
    if not base_path.is_file() or sha256_file(base_path) != expected_hash:
        raise ContractError("Claim 4 canary full-config hash mismatch")
    base = parse_claim4_settings(load_config(base_path))
    if base.mode != "full":
        raise ContractError("Claim 4 canary base config must be a full protocol")
    samples = _require_positive_int(canary["expected_samples"], "claim4_canary.expected_samples")
    wavs = _require_positive_int(canary["expected_rendered_wavs"], "claim4_canary.expected_rendered_wavs")
    if samples != 1 or wavs != len(CLAIM4_ARM_NAMES):
        raise ContractError("Claim 4 canary must account for exactly one actor x seven arms")
    evidence = _mapping(config.raw.get("evidence"), "evidence")
    if evidence.get("claim_evidence") is not False:
        raise ContractError("Claim 4 canary must set evidence.claim_evidence: false")
    return replace(base, mode="canary", expected_samples=samples, expected_rendered_wavs=wavs)


def _validate_claim4_canary_receipt_payload(
    *, receipt_bytes: bytes, full_config_sha256: str, canary_config_sha256: str,
    canary_config_digest: str, required_status: str
) -> dict[str, Any]:
    """Validate the terminal receipt independently from its Git commitment."""

    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("Claim 4 canary receipt is not valid JSON") from exc
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_version", "status", "full_config_sha256", "canary_config_sha256", "canary_config_digest",
        "canary_repository_commit", "canary_run_name", "terminal_job_url",
        "remote_hub_commit_oid", "remote_complete_url", "remote_complete_sha256",
    }:
        raise ContractError("Claim 4 canary receipt schema changed")
    if receipt["schema_version"] != "claim4_runtime_canary_receipt_v2":
        raise ContractError("Claim 4 canary receipt schema version changed")
    if receipt["full_config_sha256"] != full_config_sha256:
        raise ContractError("Claim 4 canary receipt does not bind the current full configuration")
    if receipt["canary_config_sha256"] != canary_config_sha256:
        raise ContractError("Claim 4 canary receipt does not bind the current canary configuration")
    if receipt["canary_config_digest"] != canary_config_digest:
        raise ContractError("Claim 4 canary receipt does not bind the current canonical canary configuration digest")
    if receipt["status"] != required_status:
        raise ContractError("Claim 4 full run is blocked until the GPU canary receipt is complete")
    if not isinstance(receipt["canary_repository_commit"], str) or re.fullmatch(
        r"[0-9a-f]{7,64}", receipt["canary_repository_commit"]
    ) is None:
        raise ContractError("Claim 4 canary receipt must pin its repository commit")
    if not isinstance(receipt["canary_config_digest"], str) or re.fullmatch(
        r"[0-9a-f]{64}", receipt["canary_config_digest"]
    ) is None:
        raise ContractError("Claim 4 canary receipt must pin its canonical config digest")
    if not isinstance(receipt["canary_run_name"], str) or not receipt["canary_run_name"]:
        raise ContractError("Claim 4 canary receipt must pin its run name")
    if not isinstance(receipt["remote_hub_commit_oid"], str) or re.fullmatch(
        r"[0-9a-f]{7,64}", receipt["remote_hub_commit_oid"]
    ) is None:
        raise ContractError("Claim 4 canary receipt must pin its immutable Hub commit oid")
    _require_https_url(receipt["terminal_job_url"], label="Claim 4 canary terminal_job_url")
    _require_https_url(
        receipt["remote_complete_url"], label="Claim 4 canary remote_complete_url", suffix="COMPLETE.json"
    )
    if not isinstance(receipt["remote_complete_sha256"], str) or re.fullmatch(
        r"[0-9a-f]{64}", receipt["remote_complete_sha256"]
    ) is None:
        raise ContractError("Claim 4 canary receipt must pin remote COMPLETE.json SHA-256")
    return receipt


def _require_regular_head_file(*, repo: Path, path: Path, label: str) -> tuple[Path, bytes]:
    """Return a repository-relative, immutable file only when it matches HEAD."""

    repo_root = repo.resolve()
    try:
        relative = path.absolute().relative_to(repo_root)
    except ValueError as exc:
        raise ContractError(f"{label} must be inside the repository") from exc
    relative = _safe_relative_path(relative.as_posix(), label=label)
    checked = repo_root / relative
    if checked.is_symlink() or not checked.is_file():
        raise ContractError(f"{label} must be a regular committed file")
    parent = checked.parent
    while parent != repo_root:
        if parent.is_symlink():
            raise ContractError(f"{label} has a symlinked parent")
        parent = parent.parent
    try:
        tracked = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"{label} must be committed at HEAD") from exc
    local = checked.read_bytes()
    if local != tracked:
        raise ContractError(f"{label} has uncommitted content; refusing mutable evidence")
    return relative, local


def _validate_claim4_remote_complete(*, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Fetch and verify the canary's immutable public completion marker."""

    destination = artifact_destination(
        commit=receipt["canary_repository_commit"],
        config_digest=receipt["canary_config_digest"],
        run_name=receipt["canary_run_name"],
    )
    immutable_destination = destination.at_revision(receipt["remote_hub_commit_oid"])
    if receipt["remote_complete_url"] != immutable_destination.complete_url:
        raise ContractError("Claim 4 canary receipt remote COMPLETE URL is not the required immutable run URL")
    try:
        payload = fetch_url_bytes(receipt["remote_complete_url"])
    except Exception as exc:
        raise ContractError("could not fetch the Claim 4 canary immutable remote COMPLETE.json") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != receipt["remote_complete_sha256"]:
        raise ContractError("Claim 4 canary remote COMPLETE.json SHA-256 mismatch")
    try:
        complete = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("Claim 4 canary remote COMPLETE.json is not valid JSON") from exc
    if not isinstance(complete, dict) or set(complete) != {
        "schema_version", "status", "artifact_manifest_path", "artifact_manifest_sha256",
        "files_uploaded", "destination", "remote_prefix", "atomic",
    }:
        raise ContractError("Claim 4 canary remote COMPLETE.json schema changed")
    if complete["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ContractError("Claim 4 canary remote COMPLETE.json schema version changed")
    if complete["status"] != "complete" or complete["atomic"] is not True:
        raise ContractError("Claim 4 canary remote COMPLETE.json is not an atomic completed run")
    if complete["destination"] != destination.as_dict() or complete["remote_prefix"] != destination.remote_prefix:
        raise ContractError("Claim 4 canary remote COMPLETE.json does not bind the canary run identity")
    if complete["artifact_manifest_path"] != "hub_artifact_manifest.json":
        raise ContractError("Claim 4 canary remote COMPLETE.json manifest path changed")
    if not isinstance(complete["artifact_manifest_sha256"], str) or re.fullmatch(
        r"[0-9a-f]{64}", complete["artifact_manifest_sha256"]
    ) is None:
        raise ContractError("Claim 4 canary remote COMPLETE.json has no valid artifact manifest hash")
    if not isinstance(complete["files_uploaded"], list) or not all(
        isinstance(value, str) and value for value in complete["files_uploaded"]
    ):
        raise ContractError("Claim 4 canary remote COMPLETE.json file list is malformed")
    return complete


def _load_claim4_frozen_canary_config(*, canary_config_path: Path) -> ReproConfig:
    """Load the already-HEAD-verified canary configuration for identity checks."""

    try:
        canary_config = load_config(canary_config_path)
    except Exception as exc:
        raise ContractError("Claim 4 frozen canary configuration cannot be loaded") from exc
    return canary_config


def _require_claim4_canary_run_name(*, receipt: Mapping[str, Any], canary_config: ReproConfig) -> None:
    if receipt["canary_run_name"] != canary_config.run_name:
        raise ContractError("Claim 4 canary receipt run name does not bind the frozen canary configuration")


def _require_claim4_full_canary_receipt(
    *, config: ReproConfig, settings: Claim4Settings, repo: Path
) -> dict[str, str]:
    """Require a committed, terminal canary receipt before full GPU work.

    The receipt is intentionally outside the full-config hash lock: replacing
    the committed pending template after a terminal canary must not invalidate
    the canary configuration that produced it.  Instead, the receipt binds the
    current full and canary config bytes and its own bytes must match ``HEAD``.
    """

    receipt_relative = _safe_relative_path(
        settings.canary_prerequisite["receipt_path"],
        label="claim4.canary_prerequisite.receipt_path", suffix=".json",
    )
    canary_relative = _safe_relative_path(
        settings.canary_prerequisite["canary_config_path"],
        label="claim4.canary_prerequisite.canary_config_path", suffix=".yaml",
    )
    receipt_path = repo / receipt_relative
    canary_path = repo / canary_relative
    try:
        _, full_config_bytes = _require_regular_head_file(
            repo=repo, path=config.source, label="Claim 4 full configuration"
        )
        _, canary_config_bytes = _require_regular_head_file(
            repo=repo, path=canary_path, label="Claim 4 frozen canary configuration"
        )
        _, receipt_bytes = _require_regular_head_file(
            repo=repo, path=receipt_path, label="Claim 4 canary receipt"
        )
        receipt_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("Claim 4 full run requires the canary receipt committed at HEAD") from exc
    canary_config = _load_claim4_frozen_canary_config(canary_config_path=canary_path)
    receipt = _validate_claim4_canary_receipt_payload(
        receipt_bytes=receipt_bytes,
        full_config_sha256=hashlib.sha256(full_config_bytes).hexdigest(),
        canary_config_sha256=hashlib.sha256(canary_config_bytes).hexdigest(),
        canary_config_digest=canary_config.digest,
        required_status=settings.canary_prerequisite["required_status"],
    )
    _require_claim4_canary_run_name(receipt=receipt, canary_config=canary_config)
    _validate_claim4_remote_complete(receipt=receipt)
    return {
        "path": receipt_relative.as_posix(),
        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "repository_commit": receipt_commit,
        "terminal_job_url": receipt["terminal_job_url"],
        "canary_repository_commit": receipt["canary_repository_commit"],
        "canary_run_name": receipt["canary_run_name"],
        "remote_hub_commit_oid": receipt["remote_hub_commit_oid"],
        "remote_complete_url": receipt["remote_complete_url"],
        "remote_complete_sha256": receipt["remote_complete_sha256"],
    }


def _atomic_verified_download(*, url: str, expected_sha256: str, target: Path) -> Path:
    if target.exists():
        if sha256_file(target) != expected_sha256:
            raise ContractError(f"cached Claim 4 source hash mismatch: {target}")
        return target
    payload = fetch_url_bytes(url)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ContractError(f"Claim 4 source hash mismatch: expected={expected_sha256}, actual={actual}")
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


def load_claim4_frozen_assets(
    settings: Claim4Settings, *, repo: Path, manifest: list[Any]
) -> dict[str, FrozenClaim4Asset]:
    """Resolve all selected WAVs from the committed manifest, without REST calls."""

    assets = load_frozen_claim4_asset_manifest(
        repo / settings.source["asset_manifest_path"],
        expected_sha256=settings.source["asset_manifest_sha256"],
        revision=settings.source["revision"],
    )
    require_claim4_asset_coverage(manifest, assets)
    return assets


def claim4_assets_for_item(
    assets: Mapping[str, FrozenClaim4Asset], item: Any
) -> tuple[FrozenClaim4Asset, FrozenClaim4Asset]:
    """Return the target/reference assets already verified by the frozen manifest."""

    target = claim4_asset_for_filename(assets, item.file_name)
    reference = claim4_asset_for_filename(assets, item.reference_file_name)
    if target.filename == reference.filename:
        raise ContractError("Claim 4 frozen target and reference assets must differ")
    return target, reference


def materialize_claim4_wav_pair(
    *,
    cache_root: Path,
    item: Any,
    assets: Mapping[str, FrozenClaim4Asset],
    fetch_content: Any = fetch_url_bytes,
) -> tuple[Path, Path, FrozenClaim4Asset, FrozenClaim4Asset]:
    """Cache selected LFS WAVs from immutable media URLs after hash verification."""

    target_asset, reference_asset = claim4_assets_for_item(assets, item)
    target_wav = ensure_lfs_wav(
        cache_root=cache_root,
        item=item,
        pointer=LfsPointer(target_asset.lfs_oid_sha256, target_asset.size_bytes),
        fetch_content=lambda: fetch_content(target_asset.download_url),
    )
    reference_wav = ensure_lfs_wav(
        cache_root=cache_root,
        item=item,
        filename=item.reference_wav_name,
        pointer=LfsPointer(reference_asset.lfs_oid_sha256, reference_asset.size_bytes),
        fetch_content=lambda: fetch_content(reference_asset.download_url),
    )
    return target_wav, reference_wav, target_asset, reference_asset


def _transcript_for_filename(file_name: str) -> str:
    parts = file_name.split("_")
    if len(parts) != 4 or parts[1] not in _CREMAD_TRANSCRIPTS:
        raise ContractError(f"CREMA-D transcript code is unknown: {file_name}")
    return _CREMAD_TRANSCRIPTS[parts[1]]


def claim4_target_and_reference_text(item: Any) -> tuple[str, str]:
    """Return the target synthesis text and the distinct reference prompt text."""

    target_text = _transcript_for_filename(item.file_name)
    reference_text = _transcript_for_filename(item.reference_file_name)
    if item.file_name != item.reference_file_name and target_text == reference_text:
        # Equal text is valid for a same-sentence reference, but distinct audio
        # remains required. The check documents that equality is not assumed.
        return target_text, reference_text
    return target_text, reference_text


def load_claim4_vectors(settings: Claim4Settings, *, repo: Path) -> dict[str, np.ndarray]:
    """Load every released vector only after its file/hash/layer/shape agrees."""

    try:
        import torch
    except ImportError as exc:
        raise ContractError("Claim 4 vector loading requires torch") from exc
    vectors: dict[str, np.ndarray] = {}
    operation = settings.vector_contract["operation"]
    layer = settings.vector_contract["layer"]
    expected_shape = tuple(settings.vector_contract["shape"])
    for emotion in _EMOTIONS:
        path = repo / settings.vector_contract["paths"][emotion]
        if not path.is_file():
            raise ContractError(f"released Claim 4 vector is missing: {path}")
        actual_hash = sha256_file(path)
        expected_hash = settings.vector_contract["sha256"][emotion]
        if actual_hash != expected_hash:
            raise ContractError(f"Claim 4 vector hash mismatch for {emotion}: expected={expected_hash}, actual={actual_hash}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        try:
            vector = checkpoint["steering_vectors"][operation][layer]
        except (KeyError, TypeError) as exc:
            raise ContractError(f"Claim 4 vector lacks {operation} layer {layer}: {path}") from exc
        if tuple(vector.shape) != expected_shape or not torch.isfinite(vector).all():
            raise ContractError(f"Claim 4 vector shape/finiteness contract failed for {emotion}: {tuple(vector.shape)}")
        vectors[emotion] = vector.detach().cpu().numpy().astype(np.float64, copy=True)
    return vectors


def _check_generated_wav(path: Path, *, max_clipping_fraction: float, expected_sample_rate: int) -> dict[str, Any]:
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise ContractError("Claim 4 generated-audio integrity requires torch and torchaudio") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"Claim 4 synthesis did not produce a non-empty WAV: {path}")
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != expected_sample_rate or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ContractError(f"Claim 4 output WAV violates sample-rate/finiteness contract: {path}")
    clipping_fraction = float((waveform.abs() >= 0.999).float().mean().item())
    if clipping_fraction > max_clipping_fraction:
        raise ContractError(
            f"Claim 4 output clipping exceeds limit: path={path}, actual={clipping_fraction}, limit={max_clipping_fraction}"
        )
    return {
        "wav_sha256": sha256_file(path),
        "sample_rate_hz": sample_rate,
        "samples": int(waveform.numel()),
        "clipping_fraction": clipping_fraction,
    }


def _require_claim4_model_sample_rate(*, model: Any, expected_sample_rate: int) -> int:
    """Refuse to synthesize when the pinned model and config disagree."""

    loaded_sample_rate = getattr(model, "sample_rate", None)
    if (
        not isinstance(loaded_sample_rate, int)
        or isinstance(loaded_sample_rate, bool)
        or loaded_sample_rate < 1
    ):
        raise ContractError("Claim 4 loaded CosyVoice model has no valid integer sample_rate")
    if loaded_sample_rate != expected_sample_rate:
        raise ContractError(
            "Claim 4 configured audio sample rate differs from loaded CosyVoice sample rate: "
            f"configured={expected_sample_rate}, loaded={loaded_sample_rate}"
        )
    return loaded_sample_rate


def reset_claim4_generation_seed(seed: int) -> None:
    """Reset every generator before each matched arm, including CUDA."""

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_seeded_claim4_arms(
    arms: Mapping[str, Mapping[str, Any]], *, generate: Any
) -> dict[str, Any]:
    """Run an injectable arm callback after resetting the matched sample seed.

    Keeping this boundary independent from CosyVoice lets offline tests prove
    that changing arm iteration order cannot alter decoder draws.
    """

    outputs: dict[str, Any] = {}
    for arm_name, arm in arms.items():
        seed = arm.get("sample_seed")
        if not isinstance(seed, int):
            raise ContractError(f"Claim 4 arm has no integer sample seed: {arm_name}")
        reset_claim4_generation_seed(seed)
        outputs[arm_name] = generate(arm_name, arm)
    return outputs


def audit_claim4_wav_hashes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Require unique arm outputs except the defined zero-neutral equivalence."""

    by_sample: dict[str, dict[str, str]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        arm = record.get("arm")
        digest = record.get("wav_sha256")
        if not isinstance(sample_id, str) or not isinstance(arm, str) or not isinstance(digest, str):
            raise ContractError("Claim 4 WAV hash record is incomplete")
        arms = by_sample.setdefault(sample_id, {})
        if arm in arms:
            raise ContractError(f"duplicate Claim 4 WAV record: sample={sample_id}, arm={arm}")
        arms[arm] = digest

    expected_collision = {"released_four_way", "neutral_zero_vector_diagnostic"}
    collisions: list[dict[str, Any]] = []
    for sample_id, arms in sorted(by_sample.items()):
        if set(arms) != set(CLAIM4_ARM_NAMES):
            raise ContractError(f"incomplete Claim 4 arm hashes for sample={sample_id}")
        by_hash: dict[str, set[str]] = {}
        for arm, digest in arms.items():
            by_hash.setdefault(digest, set()).add(arm)
        observed = [names for names in by_hash.values() if len(names) > 1]
        if observed != [expected_collision]:
            raise ContractError(
                f"unexpected Claim 4 WAV hash collisions for sample={sample_id}: {observed}"
            )
        digest = next(digest for digest, names in by_hash.items() if names == expected_collision)
        collisions.append({"sample_id": sample_id, "sha256": digest, "arms": sorted(expected_collision)})
    return {"allowed_collision": sorted(expected_collision), "collisions": collisions}


def run_claim4_runtime_canary(config: ReproConfig, *, repo: Path, run_root: Path) -> dict[str, Any]:
    """Run exactly one selected actor through every frozen arm.

    This is deliberately a release/API and accounting canary. It generates no
    empirical Claim 4 verdict, and a full 60x7 evaluation is refused until this
    canary has a recorded successful job URL and model-adapter review.
    """

    settings = parse_claim4_settings(config)
    if settings.mode != "canary":
        raise ContractError("Claim 4 full configuration must not be run through the seven-output canary")
    try:
        import torch
    except ImportError as exc:
        raise ContractError("Claim 4 canary requires torch with CUDA") from exc
    if not torch.cuda.is_available():
        raise ContractError("Claim 4 canary requires CUDA; CPU execution is not claim evidence")
    votes_path = _atomic_verified_download(
        url=settings.source["votes_url"], expected_sha256=settings.source["votes_sha256"],
        target=repo / ".cache" / "cremad" / settings.source["revision"] / "tabulatedVotes.csv",
    )
    rows = load_audio_vote_rows(votes_path)
    if len(rows) != settings.source["expected_audio_only_rows"]:
        raise ContractError("Claim 4 CREMA-D audio-only vote count changed")
    eligible = [row for row in rows if is_claim4_eligible(row)]
    if len(eligible) != settings.source["expected_eligible_rows"] or len(
        {cremad_actor_id(row.file_name) for row in eligible}
    ) != settings.source["expected_eligible_actors"]:
        raise ContractError("Claim 4 CREMA-D eligible-item/actor inventory changed")
    manifest = build_claim4_manifest(rows, actor_count=settings.actor_count, seed=config.seed)
    if len({entry.actor_id for entry in manifest}) != settings.actor_count:
        raise ContractError("Claim 4 60-actor manifest contract failed")
    selected = manifest[:1]
    write_manifest(run_root / "claim4_manifest_full.json", manifest)
    write_manifest(run_root / "claim4_canary_manifest.json", selected)
    assets = load_claim4_frozen_assets(settings, repo=repo, manifest=manifest)
    vectors = load_claim4_vectors(settings, repo=repo)
    from repro.gpu_baseline import _prepare_cosyvoice, _prepare_model

    asset_settings = {
        "source_url": settings.model["source_url"],
        "source_revision": settings.model["source_revision"],
        "model_id": settings.model["id"],
        "model_revision": settings.model["revision"],
    }
    cosyvoice_root = _prepare_cosyvoice(repo, asset_settings)
    model_path = _prepare_model(repo, asset_settings)
    os.environ["COSYVOICE_ROOT"] = str(cosyvoice_root)
    try:
        from cocoemo.backbones import cosyvoice2
    except ImportError as exc:
        raise ContractError("Claim 4 canary could not import the pinned CosyVoice2 adapter") from exc
    if getattr(cosyvoice2, "generate_steered_speech", None) is None:
        raise ContractError("Claim 4 required CosyVoice2 generation API is unavailable")
    model = cosyvoice2.load_model(str(model_path))
    _require_claim4_model_sample_rate(
        model=model, expected_sample_rate=settings.audio_quality["sample_rate_hz"]
    )
    outputs: list[dict[str, Any]] = []
    output_dir = run_root / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = repo / ".cache" / "cremad" / settings.source["revision"] / "AudioWAV"
    started = time.monotonic()
    for item in selected:
        target_wav, reference_wav, target_asset, reference_asset = materialize_claim4_wav_pair(
            cache_root=cache_root, item=item, assets=assets,
        )
        if (
            target_wav == reference_wav
            or item.reference_provided_label != "N"
            or item.reference_majority_labels != ("N",)
        ):
            raise ContractError("Claim 4 canary must use a distinct same-actor neutral reference")
        arms = build_claim4_arms(vectors, item.distribution, seed=config.seed, sample_id=item.file_name)
        target_text, reference_text = claim4_target_and_reference_text(item)
        for arm_name, arm in arms.items():
            output_path = output_dir / f"{item.file_name}__{arm_name}.wav"

            steering_vectors = {settings.vector_contract["operation"]: {settings.vector_contract["layer"]: torch.from_numpy(arm["vector"]).to(torch.float32)}}
            reset_claim4_generation_seed(arm["sample_seed"])
            arm_started = time.monotonic()
            result = cosyvoice2.generate_steered_speech(
                model=model,
                text=target_text,
                reference_audio_path=str(reference_wav),
                prompt_text=reference_text,
                steering_vectors=steering_vectors,
                layers=[settings.vector_contract["layer"]],
                alpha=settings.alpha,
                operations=[settings.vector_contract["operation"]],
                output_path=str(output_path),
            )
            if not isinstance(result, dict) or result.get("audio") is None:
                raise ContractError(f"Claim 4 model API returned no audio for {item.file_name}/{arm_name}")
            integrity = _check_generated_wav(
                output_path,
                max_clipping_fraction=settings.audio_quality["max_clipping_fraction"],
                expected_sample_rate=settings.audio_quality["sample_rate_hz"],
            )
            outputs.append({
                "actor_id": item.actor_id,
                "sample_id": item.file_name,
                "arm": arm_name,
                "sample_seed": arm["sample_seed"],
                "vector_l2_norm": arm["l2_norm"],
                "steering_vector_sha256": hashlib.sha256(
                    np.asarray(arm["vector"], dtype=np.float32).tobytes()
                ).hexdigest(),
                "generation_seconds": time.monotonic() - arm_started,
                "target_git_blob_sha1": target_asset.git_blob_sha1,
                "reference_git_blob_sha1": reference_asset.git_blob_sha1,
                "target_lfs_oid_sha256": target_asset.lfs_oid_sha256,
                "reference_lfs_oid_sha256": reference_asset.lfs_oid_sha256,
                "target_wav": str(target_wav.relative_to(repo)),
                "reference_wav": str(reference_wav.relative_to(repo)),
                "target_text": target_text,
                "reference_text": reference_text,
                "reference_majority_labels": list(item.reference_majority_labels),
                **integrity,
            })
    expected = settings.expected_rendered_wavs
    if len(selected) != settings.expected_samples:
        raise ContractError("Claim 4 canary selected-sample accounting changed")
    require_complete_counts(requested=expected, generated=len(outputs), evaluated=len(outputs))
    hash_audit = audit_claim4_wav_hashes(outputs)
    (run_root / "claim4_canary_outputs.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "stage": "claim4",
        "claim_evidence": False,
        "verdict": "runtime_canary_passed",
        "requested_rendered_wavs": expected,
        "generated_wavs": len(outputs),
        "integrity_checked_wavs": len(outputs),
        "wav_hash_audit": hash_audit,
        "license_attestations": settings.license_attestations,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "This one-actor seven-arm run is a release/API canary, not Claim 4 evidence.",
            "The full 60-actor/420-WAV evaluator run remains blocked on this canary's recorded GPU result and evaluator-adapter review.",
            "No naturalness claim can be verified without blinded human ratings.",
        ],
    }
