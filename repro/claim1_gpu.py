from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from repro.claim1_cross_condition import (
    FLOW_FIELDS,
    LLM_TOKEN_FIELDS,
    changed_tensor_fields,
    field_digests,
    merge_cross_conditioned_inputs,
    synthesize_cross_conditioned_with_llm_trace,
)
from repro.claim1_metrics import (
    AcousticMetricSettings,
    Claim1WavPair,
    evaluate_claim1_wav_pairs,
    parse_acoustic_metric_settings,
    summarize_claim1_metrics,
)
from repro.config import ReproConfig
from repro.contracts import ContractError, require_complete_counts, sha256_file
from repro.ravdess_acquisition import (
    RavdessAcquisitionSettings,
    ensure_ravdess_dataset,
    parse_ravdess_acquisition_settings,
)
from repro.ravdess_manifest import EMOTIONS, build_claim1_groups


SMOKE_CONDITIONS = ("slm_driven", "flow_driven", "emotional_both")
SCALED_CONDITIONS = (
    "neutral_both",
    "slm_driven",
    "flow_driven",
    "emotional_both",
    "permuted_emotion_both",
)
# Retained for the small runtime smoke and its existing public API.
CAUSAL_CONDITIONS = SMOKE_CONDITIONS


@dataclass(frozen=True)
class Claim1Settings:
    source_url: str
    source_revision: str
    model_id: str
    model_revision: str
    ravdess: RavdessAcquisitionSettings
    max_groups: int
    selected_group_ids: tuple[str, ...]
    target_emotions: tuple[str, ...]
    conditions: tuple[str, ...]
    llm_embedding_mode: str
    protocol: str = "smoke"
    expected_group_emotion_pairs: int | None = None
    expected_rendered_wavs: int | None = None
    acoustic_metrics: AcousticMetricSettings | None = None
    claim_evidence: bool = False
    max_clipping_fraction: float = 0.001


@dataclass(frozen=True)
class AudioQualitySettings:
    max_clipping_fraction: float


def parse_audio_quality_settings(values: object) -> AudioQualitySettings:
    if not isinstance(values, dict) or set(values) != {"max_clipping_fraction"}:
        raise ContractError("claim1.audio_quality must contain only max_clipping_fraction")
    maximum = values["max_clipping_fraction"]
    if not isinstance(maximum, (float, int)) or isinstance(maximum, bool):
        raise ContractError("claim1.audio_quality.max_clipping_fraction must be numeric")
    maximum = float(maximum)
    if not 0.0 <= maximum < 1.0:
        raise ContractError("claim1.audio_quality.max_clipping_fraction must be in [0, 1)")
    return AudioQualitySettings(max_clipping_fraction=maximum)


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"claim1.{key} must be a non-empty string")
    return value


def _required_positive_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or value < 1:
        raise ContractError(f"claim1.{key} must be a positive integer")
    return value


def _scaled_actor_ids(selected_group_ids: tuple[str, ...]) -> tuple[str, ...]:
    actors: list[str] = []
    for group_id in selected_group_ids:
        parts = group_id.split("_")
        if (
            len(parts) != 6
            or parts[0] != "actor"
            or not parts[1].isdigit()
            or parts[2] != "statement"
            or not parts[3].isdigit()
            or parts[4] != "repetition"
            or not parts[5].isdigit()
        ):
            raise ContractError(
                "scaled Claim 1 selected_group_ids must use actor_XX_statement_XX_repetition_XX"
            )
        actors.append(parts[1])
    return tuple(actors)


def parse_claim1_settings(config: ReproConfig) -> Claim1Settings:
    values = config.raw.get("claim1")
    if not isinstance(values, dict):
        raise ContractError("config.claim1 must be a mapping")
    ravdess = values.get("ravdess")
    if not isinstance(ravdess, dict):
        raise ContractError("config.claim1.ravdess must be a mapping")

    ravdess_settings = parse_ravdess_acquisition_settings(ravdess)
    expected_total_groups = ravdess_settings.expected_total_groups
    max_groups = values.get("max_groups", expected_total_groups)
    if not isinstance(max_groups, int) or not 1 <= max_groups <= expected_total_groups:
        raise ContractError("claim1.max_groups must be between 1 and expected_total_groups")

    selected_raw = values.get("selected_group_ids", [])
    if not isinstance(selected_raw, list) or any(
        not isinstance(value, str) or not value for value in selected_raw
    ):
        raise ContractError("claim1.selected_group_ids must be a list of non-empty strings")
    selected_group_ids = tuple(selected_raw)
    if len(set(selected_group_ids)) != len(selected_group_ids):
        raise ContractError("claim1.selected_group_ids must not contain duplicates")
    if selected_group_ids and len(selected_group_ids) != max_groups:
        raise ContractError("claim1.selected_group_ids length must equal claim1.max_groups")

    target_raw = values.get("target_emotions")
    if not isinstance(target_raw, list) or not target_raw:
        raise ContractError("claim1.target_emotions must be a non-empty list")
    target_emotions = tuple(target_raw)
    supported_targets = set(EMOTIONS.values()).difference({"neutral"})
    if set(target_emotions).difference(supported_targets):
        raise ContractError(
            "claim1.target_emotions may only contain non-neutral RAVDESS emotions: "
            + ", ".join(sorted(supported_targets))
        )
    if len(set(target_emotions)) != len(target_emotions):
        raise ContractError("claim1.target_emotions must not contain duplicates")

    protocol = values.get("protocol")
    if protocol not in {"smoke", "scaled_directional"}:
        raise ContractError("claim1.protocol must be 'smoke' or 'scaled_directional'")
    expected_conditions = SMOKE_CONDITIONS if protocol == "smoke" else SCALED_CONDITIONS
    conditions_raw = values.get("conditions")
    if not isinstance(conditions_raw, list) or tuple(conditions_raw) != expected_conditions:
        raise ContractError(
            f"claim1.conditions must be exactly {', '.join(expected_conditions)} for protocol={protocol}"
        )
    llm_embedding_mode = values.get("llm_embedding_mode", "neutral")
    if llm_embedding_mode not in {"neutral", "native_bundle"}:
        raise ContractError("claim1.llm_embedding_mode must be 'neutral' or 'native_bundle'")

    expected_pairs = _required_positive_int(values, "expected_group_emotion_pairs")
    expected_wavs = _required_positive_int(values, "expected_rendered_wavs")
    calculated_pairs = max_groups * len(target_emotions)
    calculated_wavs = calculated_pairs * len(expected_conditions)
    if expected_pairs != calculated_pairs:
        raise ContractError(
            "claim1.expected_group_emotion_pairs must equal "
            f"max_groups * target_emotions ({calculated_pairs})"
        )
    if expected_wavs != calculated_wavs:
        raise ContractError(
            "claim1.expected_rendered_wavs must equal "
            f"expected_group_emotion_pairs * conditions ({calculated_wavs})"
        )

    evidence = config.raw.get("evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("claim_evidence"), bool):
        raise ContractError("evidence.claim_evidence must be an explicit boolean")
    claim_evidence = evidence["claim_evidence"]
    if protocol == "smoke":
        if claim_evidence:
            raise ContractError("Claim 1 smoke runs must set evidence.claim_evidence: false")
        if "acoustic_metrics" in values:
            raise ContractError("Claim 1 smoke runs must not report acoustic metric evidence")
    else:
        if not claim_evidence:
            raise ContractError("scaled Claim 1 runs must set evidence.claim_evidence: true")
        if set(target_emotions) != supported_targets:
            raise ContractError(
                "scaled Claim 1 runs must use every public non-neutral RAVDESS target emotion"
            )
        if max_groups != 20 or len(selected_group_ids) != 20:
            raise ContractError("scaled Claim 1 runs require exactly 20 explicitly selected groups")
        actor_ids = _scaled_actor_ids(selected_group_ids)
        if len(set(actor_ids)) != 20:
            raise ContractError("scaled Claim 1 selected groups must use exactly 20 distinct RAVDESS actors")
        if "acoustic_metrics" not in values:
            raise ContractError("scaled Claim 1 runs must pin claim1.acoustic_metrics")
        if "audio_quality" not in values:
            raise ContractError("scaled Claim 1 runs must pin claim1.audio_quality")
    audio_quality = parse_audio_quality_settings(values.get("audio_quality"))

    return Claim1Settings(
        source_url=_required_string(values, "source_url"),
        source_revision=_required_string(values, "source_revision"),
        model_id=_required_string(values, "model_id"),
        model_revision=_required_string(values, "model_revision"),
        ravdess=ravdess_settings,
        max_groups=max_groups,
        selected_group_ids=selected_group_ids,
        target_emotions=target_emotions,
        conditions=tuple(conditions_raw),
        llm_embedding_mode=llm_embedding_mode,
        protocol=protocol,
        expected_group_emotion_pairs=expected_pairs,
        expected_rendered_wavs=expected_wavs,
        acoustic_metrics=(
            parse_acoustic_metric_settings(values["acoustic_metrics"])
            if protocol == "scaled_directional"
            else None
        ),
        claim_evidence=claim_evidence,
        max_clipping_fraction=audio_quality.max_clipping_fraction,
    )


def select_claim1_groups(
    groups: list[dict[str, object]], settings: Claim1Settings
) -> list[dict[str, object]]:
    if len(groups) != settings.ravdess.expected_total_groups:
        raise ContractError(
            "RAVDESS group count mismatch: "
            f"expected={settings.ravdess.expected_total_groups}, actual={len(groups)}"
        )
    by_id = {str(group["group_id"]): group for group in groups}
    if len(by_id) != len(groups):
        raise ContractError("RAVDESS group IDs must be unique")
    group_ids = (
        settings.selected_group_ids
        if settings.selected_group_ids
        else tuple(sorted(by_id)[: settings.max_groups])
    )
    missing = sorted(set(group_ids).difference(by_id))
    if missing:
        raise ContractError("configured RAVDESS groups are missing: " + ", ".join(missing))
    selected = [by_id[group_id] for group_id in group_ids]
    if len(selected) != settings.max_groups:
        raise ContractError(
            f"selected group count mismatch: expected={settings.max_groups}, actual={len(selected)}"
        )
    if settings.protocol == "scaled_directional":
        actors = [str(group.get("actor", "")) for group in selected]
        if len(actors) != 20 or len(set(actors)) != 20:
            raise ContractError("scaled Claim 1 selected groups must resolve to exactly 20 distinct actors")
    return selected


def allowed_changed_fields(condition: str, *, llm_embedding_mode: str) -> set[str]:
    if condition == "neutral_both":
        result: set[str] = set()
    elif condition == "slm_driven":
        result = set(LLM_TOKEN_FIELDS)
    elif condition == "flow_driven":
        result = set(FLOW_FIELDS)
    elif condition in {"emotional_both", "permuted_emotion_both"}:
        result = set(LLM_TOKEN_FIELDS).union(FLOW_FIELDS)
    else:
        raise ContractError(f"unsupported Claim 1 condition: {condition}")
    if llm_embedding_mode == "native_bundle" and condition in {
        "slm_driven",
        "emotional_both",
        "permuted_emotion_both",
    }:
        result.add("llm_embedding")
    return result


def required_changed_signal_fields(condition: str) -> set[str]:
    if condition == "neutral_both":
        return set()
    if condition == "slm_driven":
        return {"llm_prompt_speech_token"}
    if condition == "flow_driven":
        return {"flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"}
    if condition in {"emotional_both", "permuted_emotion_both"}:
        return required_changed_signal_fields("slm_driven").union(
            required_changed_signal_fields("flow_driven")
        )
    raise ContractError(f"unsupported Claim 1 condition: {condition}")


def validate_causal_field_deltas(
    neutral: dict[str, Any],
    merged: dict[str, Any],
    *,
    condition: str,
    llm_embedding_mode: str,
) -> set[str]:
    changed = changed_tensor_fields(neutral, merged)
    allowed = allowed_changed_fields(condition, llm_embedding_mode=llm_embedding_mode)
    unexpected = changed.difference(allowed)
    if unexpected:
        raise ContractError(
            f"causal field isolation changed fields outside {condition} ownership: "
            + ", ".join(sorted(unexpected))
        )
    missing_signals = required_changed_signal_fields(condition).difference(changed)
    if missing_signals:
        raise ContractError(
            f"causal field isolation did not change required {condition} signal fields: "
            + ", ".join(sorted(missing_signals))
        )
    return changed


def audit_condition_wav_hashes(
    records: list[dict[str, Any]], *, conditions: tuple[str, ...]
) -> dict[str, object]:
    """Validate complete accounting and preserve any equal hashes as evidence.

    A no-effect or exact deterministic collision is a scientifically relevant
    outcome.  Hashes establish byte provenance, not prosodic distinctness, so
    this function must never turn equality into a failed run.
    """
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        group_id = record.get("group_id")
        emotion = record.get("target_emotion")
        condition = record.get("condition")
        audio = record.get("audio")
        if not all(isinstance(value, str) and value for value in (group_id, emotion, condition)):
            raise ContractError("condition WAV record is missing group, emotion, or condition")
        if not isinstance(audio, dict) or not isinstance(audio.get("sha256"), str):
            raise ContractError("condition WAV record is missing audio SHA256")
        key = (group_id, emotion)
        hashes = by_pair.setdefault(key, {})
        if condition in hashes:
            raise ContractError(f"duplicate condition WAV record: group={group_id}, emotion={emotion}, condition={condition}")
        hashes[condition] = audio["sha256"]

    expected_conditions = set(conditions)
    collisions: list[dict[str, object]] = []
    for (group_id, emotion), hashes in sorted(by_pair.items()):
        actual_conditions = set(hashes)
        if actual_conditions != expected_conditions:
            raise ContractError(
                f"condition WAV accounting mismatch for group={group_id}, emotion={emotion}: "
                f"expected={sorted(expected_conditions)}, actual={sorted(actual_conditions)}"
            )
        duplicate_hashes: dict[str, list[str]] = {}
        for condition, digest in sorted(hashes.items()):
            duplicate_hashes.setdefault(digest, []).append(condition)
        for digest, repeated_conditions in duplicate_hashes.items():
            if len(repeated_conditions) > 1:
                collisions.append(
                    {
                        "group_id": group_id,
                        "target_emotion": emotion,
                        "sha256": digest,
                        "conditions": repeated_conditions,
                    }
                )
    return {
        "complete_group_emotion_pairs": len(by_pair),
        "conditions": list(conditions),
        "equal_hash_collisions": collisions,
        "note": "equal WAV hashes are diagnostic evidence and do not invalidate a run",
    }


def require_distinct_condition_wavs(
    records: list[dict[str, Any]], *, conditions: tuple[str, ...]
) -> dict[str, object]:
    """Backward-compatible name for the accounting-only hash audit."""

    return audit_condition_wav_hashes(records, conditions=conditions)


def deterministic_permuted_emotions(target_emotions: tuple[str, ...]) -> dict[str, str]:
    """Return a stable derangement used by the scaled mismatched-emotion arm."""

    ordered = tuple(sorted(target_emotions))
    if len(ordered) < 2:
        raise ContractError("a permuted-emotion control requires at least two target emotions")
    mapping = {
        emotion: ordered[(index + 1) % len(ordered)]
        for index, emotion in enumerate(ordered)
    }
    if any(source == destination for source, destination in mapping.items()):
        raise ContractError("permuted-emotion mapping must have no fixed points")
    return mapping


def derive_claim1_draw_seed(*, base_seed: int, group_id: str, target_emotion: str) -> int:
    """Derive a render seed independent of condition iteration order."""

    if not group_id or not target_emotion:
        raise ContractError("Claim 1 draw seed needs non-empty group_id and target_emotion")
    payload = f"claim1-ravdess-draw-v1\0{base_seed}\0{group_id}\0{target_emotion}".encode("ascii")
    # CUDA accepts this non-negative range across supported PyTorch versions.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _prepare_frontend_input(
    cosyvoice: Any, *, reference_audio: Path, prompt_text: str, tts_text: str
) -> dict[str, Any]:
    from cosyvoice.utils.file_utils import load_wav

    prompt_speech_16k = load_wav(str(reference_audio), 16000)
    frontend = getattr(cosyvoice, "frontend", None)
    if frontend is None:
        raise ContractError("loaded CosyVoice2 model has no frontend")
    model_input = frontend.frontend_zero_shot(
        tts_text=tts_text,
        prompt_text=prompt_text,
        prompt_speech_16k=prompt_speech_16k,
        resample_rate=cosyvoice.sample_rate,
        zero_shot_spk_id="",
    )
    if not isinstance(model_input, dict):
        raise ContractError("CosyVoice2 frontend_zero_shot did not return a mapping")
    source_speech_token = model_input.get("source_speech_token")
    if source_speech_token is None:
        # CosyVoice accepts an empty source token to select the text-to-token
        # LLM branch. Make that branch explicit rather than relying on a
        # default argument hidden inside the backbone implementation.
        model_input = dict(model_input)
        model_input["source_speech_token"] = torch.empty((1, 0), dtype=torch.int32)
    elif not isinstance(source_speech_token, torch.Tensor) or source_speech_token.numel() != 0:
        raise ContractError("frontend_zero_shot must provide an empty source_speech_token")
    return model_input


def _write_wav_and_record(
    *, output_path: Path, audio: torch.Tensor, sample_rate: int, max_clipping_fraction: float
) -> dict[str, Any]:
    waveform = audio.detach().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.numel() == 0:
        raise ContractError(f"cross-conditioned synthesis returned invalid waveform: {output_path}")
    if not torch.isfinite(waveform).all():
        raise ContractError(f"cross-conditioned synthesis returned non-finite audio: {output_path}")
    torchaudio.save(str(output_path), waveform, sample_rate)
    loaded, loaded_sample_rate = torchaudio.load(output_path)
    if loaded.numel() == 0 or loaded_sample_rate != sample_rate:
        raise ContractError(f"saved output is invalid: {output_path}")
    if not torch.isfinite(loaded).all():
        raise ContractError(f"saved output contains non-finite audio: {output_path}")
    duration = loaded.shape[-1] / loaded_sample_rate
    rms = float(torch.sqrt(torch.mean(loaded.float().square())).item())
    clipping_fraction = float((loaded.abs() >= 0.999).float().mean().item())
    if duration <= 0.05 or rms <= 1e-5:
        raise ContractError(
            f"degenerate cross-conditioned audio: path={output_path}, duration={duration}, rms={rms}"
        )
    if clipping_fraction > max_clipping_fraction:
        raise ContractError(
            "clipping fraction exceeds configured threshold: "
            f"path={output_path}, clipping_fraction={clipping_fraction}, "
            f"max_clipping_fraction={max_clipping_fraction}"
        )
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "sample_rate": loaded_sample_rate,
        "frames": int(loaded.shape[-1]),
        "duration_seconds": duration,
        "rms": rms,
        "clipping_fraction": clipping_fraction,
    }


def _require_reference(path: Path, *, group_id: str, emotion: str) -> Path:
    if not path.is_file():
        raise ContractError(f"RAVDESS reference is missing: group={group_id}, emotion={emotion}, path={path}")
    return path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_claim1_gpu(config: ReproConfig, *, repo: Path, run_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ContractError("Claim 1 CosyVoice2 smoke requires CUDA; CPU execution is not claim evidence")
    settings = parse_claim1_settings(config)
    _seed_everything(config.seed)

    ravdess_root = ensure_ravdess_dataset(repo=repo, settings=settings.ravdess)
    wav_paths = sorted(ravdess_root.rglob("*.wav"))
    if not wav_paths:
        raise ContractError(f"no RAVDESS WAV files found under: {ravdess_root}")
    groups = build_claim1_groups(wav_paths)
    selected_groups = select_claim1_groups(groups, settings)

    # Asset preparation is shared with the baseline, including exact source checkout and model revision.
    from repro.gpu_baseline import _prepare_cosyvoice, _prepare_model

    asset_settings = {
        "source_url": settings.source_url,
        "source_revision": settings.source_revision,
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
    }
    cosyvoice_root = _prepare_cosyvoice(repo, asset_settings)
    model_dir = _prepare_model(repo, asset_settings)
    os.environ["COSYVOICE_ROOT"] = str(cosyvoice_root)

    import cocoemo.backbones.cosyvoice2 as cv2

    cosyvoice = cv2.load_model(str(model_dir))
    cosyvoice.model.eval()
    output_dir = run_root / "synthesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = len(selected_groups) * len(settings.target_emotions) * len(settings.conditions)
    if settings.expected_rendered_wavs is not None and requested != settings.expected_rendered_wavs:
        raise ContractError(
            f"Claim 1 configured render count mismatch: expected={settings.expected_rendered_wavs}, actual={requested}"
        )
    group_emotion_pairs = len(selected_groups) * len(settings.target_emotions)
    if (
        settings.expected_group_emotion_pairs is not None
        and group_emotion_pairs != settings.expected_group_emotion_pairs
    ):
        raise ContractError(
            "Claim 1 configured group-emotion count mismatch: "
            f"expected={settings.expected_group_emotion_pairs}, actual={group_emotion_pairs}"
        )
    permuted_emotions = (
        deterministic_permuted_emotions(settings.target_emotions)
        if "permuted_emotion_both" in settings.conditions
        else {}
    )
    records: list[dict[str, Any]] = []
    for group in selected_groups:
        group_id = str(group["group_id"])
        target_text = str(group["target_text"])
        references = group.get("references")
        if not isinstance(references, dict):
            raise ContractError(f"RAVDESS group has no reference mapping: {group_id}")
        neutral = references.get("neutral")
        if not isinstance(neutral, dict) or "path" not in neutral:
            raise ContractError(f"RAVDESS group has no neutral reference: {group_id}")
        neutral_path = _require_reference(Path(str(neutral["path"])), group_id=group_id, emotion="neutral")
        neutral_input = _prepare_frontend_input(
            cosyvoice,
            reference_audio=neutral_path,
            prompt_text=target_text,
            tts_text=target_text,
        )

        emotional_paths: dict[str, Path] = {}
        emotional_inputs: dict[str, dict[str, Any]] = {}
        for emotion in settings.target_emotions:
            emotional = references.get(emotion)
            if not isinstance(emotional, dict) or "path" not in emotional:
                raise ContractError(f"RAVDESS group has no {emotion} reference: {group_id}")
            emotional_path = _require_reference(
                Path(str(emotional["path"])), group_id=group_id, emotion=emotion
            )
            emotional_paths[emotion] = emotional_path
            emotional_inputs[emotion] = _prepare_frontend_input(
                cosyvoice,
                reference_audio=emotional_path,
                prompt_text=target_text,
                tts_text=target_text,
            )

        for emotion in settings.target_emotions:
            target_emotional_path = emotional_paths[emotion]
            draw_seed = derive_claim1_draw_seed(
                base_seed=config.seed, group_id=group_id, target_emotion=emotion
            )
            for condition in settings.conditions:
                if condition == "neutral_both":
                    conditioning_emotion = "neutral"
                    conditioning_path = neutral_path
                    conditioning_input = neutral_input
                    merge_condition = "neutral_both"
                elif condition == "permuted_emotion_both":
                    conditioning_emotion = permuted_emotions[emotion]
                    conditioning_path = emotional_paths[conditioning_emotion]
                    conditioning_input = emotional_inputs[conditioning_emotion]
                    merge_condition = "emotional_both"
                else:
                    conditioning_emotion = emotion
                    conditioning_path = emotional_paths[emotion]
                    conditioning_input = emotional_inputs[emotion]
                    merge_condition = condition
                merged = merge_cross_conditioned_inputs(
                    neutral_input,
                    conditioning_input,
                    condition=merge_condition,
                    llm_embedding_mode=settings.llm_embedding_mode,
                )
                changed = validate_causal_field_deltas(
                    neutral_input,
                    merged,
                    condition=condition,
                    llm_embedding_mode=settings.llm_embedding_mode,
                )
                # Rewind every matched arm to the same draw state. The derived
                # seed omits condition by design, so condition ordering cannot
                # change stochastic decoding for a group/emotion pair.
                _seed_everything(draw_seed)
                with torch.inference_mode():
                    audio, llm_inference_calls = synthesize_cross_conditioned_with_llm_trace(
                        cosyvoice.model, merged
                    )
                output_path = output_dir / f"{group_id}-{emotion}-{condition}.wav"
                audio_record = _write_wav_and_record(
                    output_path=output_path,
                    audio=audio,
                    sample_rate=int(cosyvoice.sample_rate),
                    max_clipping_fraction=settings.max_clipping_fraction,
                )
                records.append(
                    {
                        "group_id": group_id,
                        "actor": str(group["actor"]),
                        "target_emotion": emotion,
                        "draw_seed": draw_seed,
                        "conditioning_emotion": conditioning_emotion,
                        "condition": condition,
                        "target_text": target_text,
                        "reference_audio": {
                            "neutral": {
                                "filename": neutral_path.name,
                                "sha256": sha256_file(neutral_path),
                            },
                            "target_emotional": {
                                "filename": target_emotional_path.name,
                                "sha256": sha256_file(target_emotional_path),
                            },
                            "conditioning": {
                                "filename": conditioning_path.name,
                                "sha256": sha256_file(conditioning_path),
                            },
                        },
                        "field_hashes": {
                            "neutral_frontend": field_digests(neutral_input),
                            "target_emotional_frontend": field_digests(emotional_inputs[emotion]),
                            "conditioning_frontend": field_digests(conditioning_input),
                            "merged": field_digests(merged),
                            "changed_from_neutral": sorted(changed),
                        },
                        "execution_trace": {"llm_inference_calls": llm_inference_calls},
                        "audio": audio_record,
                    }
                )

    if settings.acoustic_metrics is None:
        evaluated = len(records)
        acoustic_summary: dict[str, object] | None = None
    else:
        # RAVDESS actors are directory-scoped; reconstruct the exact reference
        # path from the already verified group map rather than a filename guess.
        references_by_group_emotion = {
            (str(group["group_id"]), emotion): Path(str(group["references"][emotion]["path"]))
            for group in selected_groups
            for emotion in settings.target_emotions
        }
        wav_pairs = [
            Claim1WavPair(
                group_id=str(record["group_id"]),
                target_emotion=str(record["target_emotion"]),
                condition=str(record["condition"]),
                generated_wav=output_dir / str(record["audio"]["path"]),
                emotional_reference_wav=references_by_group_emotion[
                    (str(record["group_id"]), str(record["target_emotion"]))
                ],
            )
            for record in records
        ]
        metrics = evaluate_claim1_wav_pairs(
            wav_pairs,
            conditions=settings.conditions,
            settings=settings.acoustic_metrics,
        )
        evaluated = len(metrics)
        acoustic_summary = summarize_claim1_metrics(
            metrics,
            conditions=settings.conditions,
            settings=settings.acoustic_metrics,
        )
        (run_root / "per_sample_acoustic.jsonl").write_text(
            "".join(json.dumps(asdict(metric), sort_keys=True) + "\n" for metric in metrics),
            encoding="utf-8",
        )
        (run_root / "acoustic_summary.json").write_text(
            json.dumps(acoustic_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    require_complete_counts(requested=requested, generated=len(records), evaluated=evaluated)
    hash_audit = audit_condition_wav_hashes(records, conditions=settings.conditions)
    (run_root / "per_sample.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    return {
        "stage": "claim1",
        "claim_evidence": settings.claim_evidence,
        "verdict": (
            str(acoustic_summary["preregistered_directional_assessment"]["directional_outcome"])
            if acoustic_summary is not None
            else "runtime_causal_replacement_pass"
        ),
        "requested": requested,
        "generated": len(records),
        "evaluated": evaluated,
        "protocol": {
            "conditions": list(settings.conditions),
            "llm_embedding_mode": settings.llm_embedding_mode,
            "target_emotions": list(settings.target_emotions),
            "expected_total_groups": settings.ravdess.expected_total_groups,
            "selected_groups": [str(group["group_id"]) for group in selected_groups],
            "group_emotion_pairs": group_emotion_pairs,
            "rendered_wavs": requested,
            "permuted_emotion_mapping": permuted_emotions,
            "draw_seed_schedule": {
                f"{str(group['group_id'])}:{emotion}": derive_claim1_draw_seed(
                    base_seed=config.seed,
                    group_id=str(group["group_id"]),
                    target_emotion=emotion,
                )
                for group in selected_groups
                for emotion in settings.target_emotions
            },
            "draw_seed_scope": "Each group-emotion pair is reset to its listed seed before every condition.",
            "max_clipping_fraction": settings.max_clipping_fraction,
        },
        "assets": {
            "cosyvoice_source_revision": settings.source_revision,
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "ravdess_source_url": settings.ravdess.source_url,
            "ravdess_release": settings.ravdess.release,
            "ravdess_archive_url": settings.ravdess.archive_url,
            "ravdess_archive_md5": settings.ravdess.archive_md5,
            "ravdess_license": settings.ravdess.license,
        },
        "records": records,
        "wav_hash_audit": hash_audit,
        "acoustic_evaluation": acoustic_summary,
        "limitations": [
            (
                "This smoke run proves the strict field-level causal replacement executes; it does not estimate Claim 1 CCCs."
                if acoustic_summary is None
                else "This uses public RAVDESS with a frozen proxy evaluator and is directional evidence, not an exact Table 1 reproduction."
            ),
            "The runtime trace proves one SLM inference call per render; it does not prove that the intervention caused any observed prosody difference.",
            "Audio naturalness and perceived emotion require blinded human evaluation.",
        ],
    }
