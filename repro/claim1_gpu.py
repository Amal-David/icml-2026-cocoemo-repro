from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
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
    synthesize_cross_conditioned,
)
from repro.config import ReproConfig
from repro.contracts import ContractError, require_complete_counts, sha256_file
from repro.ravdess_manifest import EMOTIONS, build_claim1_groups


CAUSAL_CONDITIONS = ("slm_driven", "flow_driven", "emotional_both")


@dataclass(frozen=True)
class Claim1Settings:
    source_url: str
    source_revision: str
    model_id: str
    model_revision: str
    ravdess_source_url: str
    ravdess_release: str
    ravdess_root: str
    expected_total_groups: int
    max_groups: int
    selected_group_ids: tuple[str, ...]
    target_emotions: tuple[str, ...]
    conditions: tuple[str, ...]
    llm_embedding_mode: str


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


def parse_claim1_settings(config: ReproConfig) -> Claim1Settings:
    values = config.raw.get("claim1")
    if not isinstance(values, dict):
        raise ContractError("config.claim1 must be a mapping")
    ravdess = values.get("ravdess")
    if not isinstance(ravdess, dict):
        raise ContractError("config.claim1.ravdess must be a mapping")

    expected_total_groups = _required_positive_int(ravdess, "expected_total_groups")
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

    conditions_raw = values.get("conditions", list(CAUSAL_CONDITIONS))
    if not isinstance(conditions_raw, list) or tuple(conditions_raw) != CAUSAL_CONDITIONS:
        raise ContractError(
            "claim1.conditions must be exactly " + ", ".join(CAUSAL_CONDITIONS)
        )
    llm_embedding_mode = values.get("llm_embedding_mode", "neutral")
    if llm_embedding_mode not in {"neutral", "native_bundle"}:
        raise ContractError("claim1.llm_embedding_mode must be 'neutral' or 'native_bundle'")

    return Claim1Settings(
        source_url=_required_string(values, "source_url"),
        source_revision=_required_string(values, "source_revision"),
        model_id=_required_string(values, "model_id"),
        model_revision=_required_string(values, "model_revision"),
        ravdess_source_url=_required_string(ravdess, "source_url"),
        ravdess_release=_required_string(ravdess, "release"),
        ravdess_root=_required_string(ravdess, "root"),
        expected_total_groups=expected_total_groups,
        max_groups=max_groups,
        selected_group_ids=selected_group_ids,
        target_emotions=target_emotions,
        conditions=tuple(conditions_raw),
        llm_embedding_mode=llm_embedding_mode,
    )


def select_claim1_groups(
    groups: list[dict[str, object]], settings: Claim1Settings
) -> list[dict[str, object]]:
    if len(groups) != settings.expected_total_groups:
        raise ContractError(
            "RAVDESS group count mismatch: "
            f"expected={settings.expected_total_groups}, actual={len(groups)}"
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
    return selected


def allowed_changed_fields(condition: str, *, llm_embedding_mode: str) -> set[str]:
    if condition == "slm_driven":
        result = set(LLM_TOKEN_FIELDS)
    elif condition == "flow_driven":
        result = set(FLOW_FIELDS)
    elif condition == "emotional_both":
        result = set(LLM_TOKEN_FIELDS).union(FLOW_FIELDS)
    else:
        raise ContractError(f"unsupported Claim 1 condition: {condition}")
    if llm_embedding_mode == "native_bundle" and condition in {
        "slm_driven",
        "emotional_both",
    }:
        result.add("llm_embedding")
    return result


def required_changed_signal_fields(condition: str) -> set[str]:
    if condition == "slm_driven":
        return {"llm_prompt_speech_token"}
    if condition == "flow_driven":
        return {"flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"}
    if condition == "emotional_both":
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


def require_distinct_condition_wavs(
    records: list[dict[str, Any]], *, conditions: tuple[str, ...]
) -> None:
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
    for (group_id, emotion), hashes in sorted(by_pair.items()):
        actual_conditions = set(hashes)
        if actual_conditions != expected_conditions:
            raise ContractError(
                f"condition WAV accounting mismatch for group={group_id}, emotion={emotion}: "
                f"expected={sorted(expected_conditions)}, actual={sorted(actual_conditions)}"
            )
        if len(set(hashes.values())) != len(conditions):
            raise ContractError(
                f"causal conditions produced non-distinct WAVs for group={group_id}, emotion={emotion}"
            )


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
    return model_input


def _write_wav_and_record(
    *, output_path: Path, audio: torch.Tensor, sample_rate: int
) -> dict[str, Any]:
    waveform = audio.detach().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.numel() == 0:
        raise ContractError(f"cross-conditioned synthesis returned invalid waveform: {output_path}")
    torchaudio.save(str(output_path), waveform, sample_rate)
    loaded, loaded_sample_rate = torchaudio.load(output_path)
    if loaded.numel() == 0 or loaded_sample_rate != sample_rate:
        raise ContractError(f"saved output is invalid: {output_path}")
    duration = loaded.shape[-1] / loaded_sample_rate
    rms = float(torch.sqrt(torch.mean(loaded.float().square())).item())
    clipping_fraction = float((loaded.abs() >= 0.999).float().mean().item())
    if duration <= 0.05 or rms <= 1e-5:
        raise ContractError(
            f"degenerate cross-conditioned audio: path={output_path}, duration={duration}, rms={rms}"
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

    ravdess_root = repo / settings.ravdess_root
    if not ravdess_root.is_dir():
        raise ContractError(f"configured RAVDESS root does not exist: {ravdess_root}")
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

        for emotion in settings.target_emotions:
            emotional = references.get(emotion)
            if not isinstance(emotional, dict) or "path" not in emotional:
                raise ContractError(f"RAVDESS group has no {emotion} reference: {group_id}")
            emotional_path = _require_reference(
                Path(str(emotional["path"])), group_id=group_id, emotion=emotion
            )
            emotional_input = _prepare_frontend_input(
                cosyvoice,
                reference_audio=emotional_path,
                prompt_text=target_text,
                tts_text=target_text,
            )
            for condition in settings.conditions:
                merged = merge_cross_conditioned_inputs(
                    neutral_input,
                    emotional_input,
                    condition=condition,
                    llm_embedding_mode=settings.llm_embedding_mode,
                )
                changed = validate_causal_field_deltas(
                    neutral_input,
                    merged,
                    condition=condition,
                    llm_embedding_mode=settings.llm_embedding_mode,
                )
                with torch.inference_mode():
                    audio = synthesize_cross_conditioned(cosyvoice.model, merged)
                output_path = output_dir / f"{group_id}-{emotion}-{condition}.wav"
                audio_record = _write_wav_and_record(
                    output_path=output_path, audio=audio, sample_rate=int(cosyvoice.sample_rate)
                )
                records.append(
                    {
                        "group_id": group_id,
                        "target_emotion": emotion,
                        "condition": condition,
                        "target_text": target_text,
                        "reference_audio": {
                            "neutral": {
                                "filename": neutral_path.name,
                                "sha256": sha256_file(neutral_path),
                            },
                            "emotional": {
                                "filename": emotional_path.name,
                                "sha256": sha256_file(emotional_path),
                            },
                        },
                        "field_hashes": {
                            "neutral_frontend": field_digests(neutral_input),
                            "emotional_frontend": field_digests(emotional_input),
                            "merged": field_digests(merged),
                            "changed_from_neutral": sorted(changed),
                        },
                        "audio": audio_record,
                    }
                )

    require_complete_counts(requested=requested, generated=len(records), evaluated=len(records))
    require_distinct_condition_wavs(records, conditions=settings.conditions)
    (run_root / "per_sample.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    return {
        "stage": "claim1",
        "claim_evidence": False,
        "verdict": "runtime_causal_replacement_pass",
        "requested": requested,
        "generated": len(records),
        "evaluated": len(records),
        "protocol": {
            "conditions": list(settings.conditions),
            "llm_embedding_mode": settings.llm_embedding_mode,
            "target_emotions": list(settings.target_emotions),
            "expected_total_groups": settings.expected_total_groups,
            "selected_groups": [str(group["group_id"]) for group in selected_groups],
        },
        "assets": {
            "cosyvoice_source_revision": settings.source_revision,
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "ravdess_source_url": settings.ravdess_source_url,
            "ravdess_release": settings.ravdess_release,
        },
        "records": records,
        "limitations": [
            "This smoke run proves the strict field-level causal replacement executes; it does not estimate Claim 1 CCCs.",
            "The preregistered full run expands only the configured groups and target emotions with these same conditions and record schema.",
            "Audio naturalness and perceived emotion require blinded human evaluation.",
        ],
    }
