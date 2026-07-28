"""GPU runner for the bounded public Claim 2/3 RAVDESS proxy.

No router calls this module until its CUDA canary succeeds.  The runner does
not synthesize or upload source audio; it writes derived activation/vector
artifacts only and aborts on the first malformed input.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.claim23_analysis import (
    actor_cluster_bootstrap,
    claim2_directional_verdict,
    claim2_heldout_controls,
    claim3_site_verdict,
    fit_five_class_centroids,
    nearest_centroid_predictions,
    paired_mean_difference_vectors,
    score_all_sites,
    select_top_sites,
    shuffled_train_vectors,
    norm_matched_random_vectors,
    vector_sha256,
)
from repro.claim23_extraction import COSYVOICE_HOOK_TYPES, LAYERS, extract_unistream_activations
from repro.claim23_manifest import (
    CLAIM23_EMOTIONS,
    EXPECTED_CLIPS_BY_SPLIT,
    TARGET_EMOTIONS,
    build_claim23_manifest,
    build_claim23_pair_ledger,
    manifest_records,
    pair_records,
    select_claim23_canary,
)
from repro.config import ReproConfig, load_config
from repro.contracts import ContractError, sha256_file
from repro.ravdess_acquisition import RavdessAcquisitionSettings, ensure_ravdess_dataset, parse_ravdess_acquisition_settings


@dataclass(frozen=True)
class Claim23Settings:
    mode: str
    source_url: str
    source_revision: str
    model_id: str
    model_revision: str
    ravdess: RavdessAcquisitionSettings
    expected_clips: int
    expected_activation_records: int
    evidence: bool
    primary_claim2_site: tuple[str, int]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a mapping")
    return value


def _parse_full_settings(config: ReproConfig) -> Claim23Settings:
    values = _mapping(config.raw.get("claim23"), "claim23")
    required = {
        "protocol", "mode", "source_url", "source_revision", "model_id", "model_revision", "ravdess",
        "emotions", "operations", "layers", "batch_size", "expected_clips", "expected_activation_records",
        "claim2_primary_site", "claim3",
    }
    if set(values) != required:
        raise ContractError(f"claim23 keys changed: missing={sorted(required - set(values))}, unexpected={sorted(set(values) - required)}")
    if values["protocol"] != "ravdess_paired_sites_public_proxy_v1" or values["mode"] != "full":
        raise ContractError("Claim 2/3 full config must use the frozen public proxy protocol")
    for key in ("source_url", "source_revision", "model_id", "model_revision"):
        if not isinstance(values[key], str) or not values[key]:
            raise ContractError(f"claim23.{key} must be a non-empty string")
    if values["source_url"] != "https://github.com/FunAudioLLM/CosyVoice.git" or values["source_revision"] != "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc" or values["model_id"] != "FunAudioLLM/CosyVoice2-0.5B" or values["model_revision"] != "eec1ae6c79877dbd9379285cf8789c9e0879293d":
        raise ContractError("Claim 2/3 CosyVoice source/model pins changed")
    ravdess = parse_ravdess_acquisition_settings(_mapping(values["ravdess"], "claim23.ravdess"))
    if tuple(values["emotions"]) != CLAIM23_EMOTIONS or tuple(values["operations"]) != COSYVOICE_HOOK_TYPES or tuple(values["layers"]) != LAYERS or values["batch_size"] != 1:
        raise ContractError("Claim 2/3 must use the frozen five emotions, ten sites, 24 layers, and batch size one")
    if values["expected_clips"] != 480 or values["expected_activation_records"] != 480 * 10 * 24:
        raise ContractError("Claim 2/3 full output accounting changed")
    site = _mapping(values["claim2_primary_site"], "claim23.claim2_primary_site")
    if site != {"operation": "attn_output", "layer": 17}:
        raise ContractError("Claim 2 primary site must stay attn_output layer 17")
    claim3 = _mapping(values["claim3"], "claim23.claim3")
    if claim3 != {"selection_split": "validation", "top_sites": 2, "max_t_permutations": 200, "bootstrap_replicates": 10000, "bootstrap_seed": 20260728}:
        raise ContractError("Claim 3 fixed selection/null/bootstrap contract changed")
    evidence = _mapping(config.raw.get("evidence"), "evidence").get("claim_evidence")
    if evidence is not True:
        raise ContractError("Claim 2/3 full run must be marked claim evidence")
    return Claim23Settings("full", values["source_url"], values["source_revision"], values["model_id"], values["model_revision"], ravdess, 480, 480 * 10 * 24, True, ("attn_output", 17))


def parse_claim23_settings(config: ReproConfig) -> Claim23Settings:
    canary = config.raw.get("claim23_canary")
    if canary is None:
        return _parse_full_settings(config)
    values = _mapping(canary, "claim23_canary")
    if set(values) != {"base_config", "base_config_sha256", "expected_clips", "expected_activation_records"}:
        raise ContractError("Claim 2/3 canary keys changed")
    base_name, base_hash = values["base_config"], values["base_config_sha256"]
    if not isinstance(base_name, str) or Path(base_name).name != base_name or not isinstance(base_hash, str) or len(base_hash) != 64:
        raise ContractError("Claim 2/3 canary must hash-lock its sibling full config")
    base_path = config.source.parent / base_name
    if not base_path.is_file() or sha256_file(base_path) != base_hash:
        raise ContractError("Claim 2/3 canary full-config hash mismatch")
    base = _parse_full_settings(load_config(base_path))
    if values["expected_clips"] != 12 or values["expected_activation_records"] != 12 * 10 * 24:
        raise ContractError("Claim 2/3 canary must contain exactly 12 clips x 240 sites")
    evidence = _mapping(config.raw.get("evidence"), "evidence").get("claim_evidence")
    if evidence is not False:
        raise ContractError("Claim 2/3 canary must not be claim evidence")
    return replace(base, mode="canary", expected_clips=12, expected_activation_records=12 * 10 * 24, evidence=False)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _activation_rows(*, clip: Any, vectors: dict[tuple[str, int], torch.Tensor], terminal: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (operation, layer), activation in sorted(vectors.items()):
        rows.append({
            "clip_id": clip.clip_id, "group_id": clip.group_id, "actor": clip.actor, "split": clip.split,
            "emotion": clip.emotion, "operation": operation, "layer": layer, "activation": activation,
            "activation_shape": list(activation.shape),
            **terminal,
        })
    return rows


def _clip_failure(path: Path, *, clip_id: str, exc: BaseException) -> None:
    _json(path, {"failed_clip_id": clip_id, "error_type": type(exc).__name__, "error": str(exc)})


def _test_accuracy_interval(predictions: list[dict[str, Any]]) -> Any:
    return actor_cluster_bootstrap(
        [{"actor": row["actor"], "value": 1.0 if row["correct"] else 0.0} for row in predictions],
        value_key="value",
    )


def _confusion_matrix(predictions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {truth: {prediction: 0 for prediction in CLAIM23_EMOTIONS} for truth in CLAIM23_EMOTIONS}
    for row in predictions:
        truth, prediction = row.get("truth"), row.get("prediction")
        if truth not in matrix or prediction not in matrix[truth]:
            raise ContractError("Claim 3 prediction has an invalid class for confusion accounting")
        matrix[truth][prediction] += 1
    if sum(sum(row.values()) for row in matrix.values()) != len(predictions):
        raise ContractError("Claim 3 confusion accounting is incomplete")
    return matrix


def _claim3_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = score_all_sites(rows)
    selected = select_top_sites(scores)
    paper_sites = [("attn_output", 14), ("attn_output", 17)]
    report_sites = list(selected)
    for site in paper_sites:
        if site not in report_sites:
            report_sites.append(site)
    ranks = {
        site: index + 1
        for index, site in enumerate(sorted(scores, key=lambda site: (-scores[site], site[0], site[1])))
    }
    test: dict[str, Any] = {}
    for site in report_sites:
        predictions = nearest_centroid_predictions(rows, centroids=fit_five_class_centroids(rows, site=site), site=site, split="test")
        test[f"{site[0]}@{site[1]}"] = {
            "validation_accuracy": scores[site], "validation_rank": ranks[site],
            "accuracy": _test_accuracy_interval(predictions).__dict__, "confusion_matrix": _confusion_matrix(predictions),
            "predictions": predictions,
        }
    # The expensive max-T null is intentionally deferred to the full run only.
    # It is imported lazily so the 12-clip structural canary cannot accidentally invoke it.
    from repro.claim23_analysis import max_t_permutation_pvalue
    max_t_p, null = max_t_permutation_pvalue(rows, observed_max=max(scores.values()))
    primary = selected[0]
    primary_predictions = test[f"{primary[0]}@{primary[1]}"]["predictions"]
    primary_interval = _test_accuracy_interval(primary_predictions)
    if primary[0] == "attn_output":
        primary_by_clip = {row["clip_id"]: row for row in primary_predictions}
        controls: dict[str, list[bool]] = {clip_id: [] for clip_id in primary_by_clip}
        for operation in COSYVOICE_HOOK_TYPES:
            if operation == "attn_output":
                continue
            site = (operation, primary[1])
            predictions = nearest_centroid_predictions(
                rows, centroids=fit_five_class_centroids(rows, site=site), site=site, split="test"
            )
            for row in predictions:
                controls[row["clip_id"]].append(bool(row["correct"]))
        comparison_rows = []
        for clip_id, primary_row in primary_by_clip.items():
            control_values = sorted(float(value) for value in controls[clip_id])
            median = control_values[len(control_values) // 2]
            comparison_rows.append({"actor": primary_row["actor"], "value": float(primary_row["correct"]) - median})
        comparison = actor_cluster_bootstrap(comparison_rows, value_key="value")
    else:
        # The preregistered positive decision cannot be met outside attention;
        # use a neutral contrast solely to preserve a complete diagnostic report.
        comparison = actor_cluster_bootstrap(
            [{"actor": row["actor"], "value": 0.0} for row in primary_predictions], value_key="value"
        )
    return {
        "validation_scores": {f"{op}@{layer}": value for (op, layer), value in scores.items()},
        "selected_sites": [f"{op}@{layer}" for op, layer in selected], "max_t_p_value": max_t_p, "max_t_null": null,
        "paper_sites": ["attn_output@14", "attn_output@17"],
        "validation_ranks": {f"{op}@{layer}": ranks[(op, layer)] for op, layer in report_sites},
        "test": test,
        "verdict": claim3_site_verdict(selected_site=primary, max_t_p_value=max_t_p, selected_test=primary_interval, attention_minus_median=comparison),
    }


def run_claim23_gpu(config: ReproConfig, *, repo: Path, run_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ContractError("Claim 2/3 public proxy requires CUDA; CPU execution is not evidence")
    settings = parse_claim23_settings(config)
    ravdess_root = ensure_ravdess_dataset(repo=repo, settings=settings.ravdess)
    manifest = build_claim23_manifest(sorted(ravdess_root.rglob("*.wav")))
    selected = manifest if settings.mode == "full" else select_claim23_canary(manifest)
    if len(selected) != settings.expected_clips:
        raise ContractError("Claim 2/3 selected clip accounting changed")
    from repro.gpu_baseline import _prepare_cosyvoice, _prepare_model
    asset_settings = {"source_url": settings.source_url, "source_revision": settings.source_revision, "model_id": settings.model_id, "model_revision": settings.model_revision}
    cosyvoice_root = _prepare_cosyvoice(repo, asset_settings)
    model_dir = _prepare_model(repo, asset_settings)
    os.environ["COSYVOICE_ROOT"] = str(cosyvoice_root)
    import cocoemo.backbones.cosyvoice2 as cv2
    cosyvoice = cv2.load_model(str(model_dir))
    cosyvoice.model.eval()
    activation_rows: list[dict[str, Any]] = []
    run_root.mkdir(parents=True, exist_ok=True)
    failed = run_root / "failed_inputs.json"
    for clip in selected:
        try:
            vectors, terminal = extract_unistream_activations(cosyvoice.model, cosyvoice.frontend, audio_path=clip.path, transcript=clip.transcript)
            activation_rows.extend(_activation_rows(clip=clip, vectors=vectors, terminal=terminal))
        except BaseException as exc:
            _clip_failure(failed, clip_id=clip.clip_id, exc=exc)
            raise
    if len(activation_rows) != settings.expected_activation_records:
        raise ContractError(f"Claim 2/3 activation count mismatch: expected={settings.expected_activation_records}, actual={len(activation_rows)}")
    torch.save(activation_rows, run_root / "activations.pt")
    _json(run_root / "manifest.json", manifest_records(selected))
    shape_map: dict[str, list[int]] = {}
    for row in activation_rows:
        key = f"{row['operation']}@{row['layer']}"
        shape = row["activation_shape"]
        if key in shape_map and shape_map[key] != shape:
            raise ContractError(f"Claim 2/3 hook shape changed across clips: {key}")
        shape_map[key] = shape
    if len(shape_map) != len(COSYVOICE_HOOK_TYPES) * len(LAYERS):
        raise ContractError("Claim 2/3 hook-shape map is incomplete")
    _json(run_root / "hook_shapes.json", shape_map)
    result: dict[str, Any] = {"mode": settings.mode, "claim_evidence": settings.evidence, "clips": len(selected), "activation_records": len(activation_rows), "failed_inputs": None}
    if settings.mode == "full":
        pairs = build_claim23_pair_ledger(manifest)
        pairs_as_records = pair_records(pairs)
        _json(run_root / "pair_ledger.json", pairs_as_records)
        vectors, ledger = paired_mean_difference_vectors(activation_rows, pairs_as_records)
        vector_manifest = {
            emotion: {f"{op}@{layer}": {"sha256": vector_sha256(vector), "shape": list(vector.shape), "pair_ids": ledger[emotion]} for (op, layer), vector in sites.items()}
            for emotion, sites in vectors.items()
        }
        torch.save(vectors, run_root / "train_vectors.pt")
        _json(run_root / "train_vector_manifest.json", vector_manifest)
        shuffled = shuffled_train_vectors(activation_rows, pairs_as_records, site=settings.primary_claim2_site)
        claim2_results = {}
        for emotion in TARGET_EMOTIONS:
            vector = vectors[emotion][settings.primary_claim2_site]
            random_seed = config.seed + TARGET_EMOTIONS.index(emotion)
            claim2_results[emotion] = claim2_heldout_controls(
                activation_rows,
                pairs_as_records,
                target_emotion=emotion,
                site=settings.primary_claim2_site,
                target_vector=vector,
                shuffled_vectors=shuffled[emotion],
                random_vectors=norm_matched_random_vectors(vector, seed=random_seed),
                shuffled_seed=config.seed,
                random_seed=random_seed,
            )
        control_ledger = {emotion: values["control_ledger"] for emotion, values in claim2_results.items()}
        _json(run_root / "claim2_control_ledger.json", control_ledger)
        claim2 = {
            "primary_site": "attn_output@17",
            "per_emotion": {
                emotion: {key: value.__dict__ if hasattr(value, "__dict__") else value for key, value in values.items() if key != "control_ledger"}
                for emotion, values in claim2_results.items()
            },
            "control_ledger": "claim2_control_ledger.json", "pair_ledger": "pair_ledger.json",
            "verdict": claim2_directional_verdict(claim2_results),
        }
        _json(run_root / "claim2.json", claim2)
        claim3 = _claim3_summary(activation_rows)
        _json(run_root / "claim3.json", claim3)
        result.update({"pair_ledger": len(pairs), "claim2": claim2, "claim3": claim3})
    _json(run_root / "summary.json", result)
    return result
