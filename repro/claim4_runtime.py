"""Strict GPU runtime contracts for the bounded public Claim 4 proxy.

The adapters deliberately fail closed.  Unit tests cover the pure protocol;
the first real execution must additionally serve as a model/API canary before
any claim verdict is interpreted as empirical evidence.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from repro.claim4_evaluation import build_claim4_arms
from repro.claim4_protocol import (
    CLAIM4_ARM_NAMES,
    LfsPointer,
    build_claim4_manifest,
    ensure_lfs_wav,
    fetch_url_bytes,
    parse_lfs_pointer,
    write_manifest,
)
from repro.config import ReproConfig, load_config
from repro.contracts import ContractError, require_complete_counts, sha256_file
from repro.cremad_selection import load_audio_vote_rows
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
            "revision", "votes_url", "votes_sha256", "content_metadata_url_template", "git_blob_url_template",
            "expected_audio_only_rows", "expected_eligible_rows", "expected_eligible_actors",
        },
    )
    if not all(isinstance(source[key], str) and source[key] for key in (
        "revision", "votes_url", "votes_sha256", "content_metadata_url_template", "git_blob_url_template"
    )) or len(source["votes_sha256"]) != 64:
        raise ContractError("claim4.source must pin non-empty revision/URLs and a SHA-256 vote-table hash")
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
    if not 0.0 <= clipping < 1.0 or audio_quality["sample_rate_hz"] != 22050:
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

    evidence = _mapping(config.raw.get("evidence"), "evidence")
    if evidence.get("claim_evidence") is not (mode == "full"):
        raise ContractError("Claim 4 evidence flag must match its execution mode")
    return Claim4Settings(mode, source, actor_count, alpha, expected_samples, expected_wavs, vector_contract, model, audio_quality, evaluators, decisions)


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


def _json_url(url: str) -> dict[str, Any]:
    try:
        value = json.loads(fetch_url_bytes(url).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"expected JSON from GitHub metadata endpoint: {url}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object from GitHub metadata endpoint: {url}")
    return value


def _lfs_pointer_and_download_url(settings: Claim4Settings, filename: str) -> tuple[LfsPointer, str]:
    metadata_url = settings.source["content_metadata_url_template"].format(
        filename=filename, revision=settings.source["revision"]
    )
    metadata = _json_url(metadata_url)
    blob_sha = metadata.get("sha")
    download_url = metadata.get("download_url")
    if not isinstance(blob_sha, str) or not blob_sha or not isinstance(download_url, str) or not download_url:
        raise ContractError(f"GitHub content metadata lacks blob SHA/download URL for {filename}")
    pointer_json = _json_url(settings.source["git_blob_url_template"].format(git_blob_sha=blob_sha))
    encoded = pointer_json.get("content")
    if not isinstance(encoded, str) or pointer_json.get("encoding") != "base64":
        raise ContractError(f"GitHub blob is not a base64 Git-LFS pointer for {filename}")
    try:
        return parse_lfs_pointer(base64.b64decode("".join(encoded.split()), validate=True)), download_url
    except ValueError as exc:
        raise ContractError(f"invalid base64 Git-LFS pointer for {filename}") from exc


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
    outputs: list[dict[str, Any]] = []
    output_dir = run_root / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = repo / ".cache" / "cremad" / settings.source["revision"] / "AudioWAV"
    started = time.monotonic()
    for item in selected:
        target_pointer, target_download_url = _lfs_pointer_and_download_url(settings, item.file_name)
        reference_pointer, reference_download_url = _lfs_pointer_and_download_url(
            settings, item.reference_file_name
        )
        target_wav = ensure_lfs_wav(
            cache_root=cache_root, item=item, pointer=target_pointer,
            fetch_content=lambda url=target_download_url: fetch_url_bytes(url),
        )
        reference_wav = ensure_lfs_wav(
            cache_root=cache_root, item=item, filename=item.reference_wav_name, pointer=reference_pointer,
            fetch_content=lambda url=reference_download_url: fetch_url_bytes(url),
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
                "target_lfs_oid_sha256": target_pointer.oid_sha256,
                "reference_lfs_oid_sha256": reference_pointer.oid_sha256,
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
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "This one-actor seven-arm run is a release/API canary, not Claim 4 evidence.",
            "The full 60-actor/420-WAV evaluator run remains blocked on this canary's recorded GPU result and evaluator-adapter review.",
            "No naturalness claim can be verified without blinded human ratings.",
        ],
    }
