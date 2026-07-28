"""Full 60-actor Claim 4 public-proxy execution and evaluation.

This runner deliberately uses the same frozen source, manifest, vector,
CosyVoice API, seed reset, and WAV-integrity functions as the seven-output
canary.  It differs only in selecting all manifest rows and attaching the
pinned evaluators after generation is complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from repro.claim4_evaluation import build_claim4_arms, decide_claim4_directional_proxy
from repro.claim4_evaluators import claim4_proportion_metrics, load_frozen_claim4_evaluators
from repro.claim4_protocol import CLAIM4_ARM_NAMES, build_claim4_manifest, write_manifest
from repro.claim4_runtime import (
    _atomic_verified_download,
    _check_generated_wav,
    audit_claim4_wav_hashes,
    claim4_target_and_reference_text,
    load_claim4_frozen_assets,
    load_claim4_vectors,
    materialize_claim4_wav_pair,
    parse_claim4_settings,
    _require_claim4_full_canary_receipt,
    _require_claim4_model_sample_rate,
    reset_claim4_generation_seed,
)
from repro.claim4_protocol import cremad_actor_id, is_claim4_eligible
from repro.config import ReproConfig
from repro.contracts import ContractError, require_complete_counts
from repro.cremad_selection import load_audio_vote_rows


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_claim4_runtime_full(config: ReproConfig, *, repo: Path, run_root: Path) -> dict[str, Any]:
    """Generate and score exactly 60 actors x seven arms, or fail closed."""

    settings = parse_claim4_settings(config)
    if settings.mode != "full":
        raise ContractError("Claim 4 full runner requires the frozen full configuration")
    canary_receipt = _require_claim4_full_canary_receipt(
        config=config, settings=settings, repo=repo
    )
    try:
        import torch
    except ImportError as exc:
        raise ContractError("Claim 4 full runner requires torch with CUDA") from exc
    if not torch.cuda.is_available():
        raise ContractError("Claim 4 full runner requires CUDA; CPU execution is not claim evidence")

    votes_path = _atomic_verified_download(
        url=settings.source["votes_url"], expected_sha256=settings.source["votes_sha256"],
        target=repo / ".cache" / "cremad" / settings.source["revision"] / "tabulatedVotes.csv",
    )
    rows = load_audio_vote_rows(votes_path)
    eligible = [row for row in rows if is_claim4_eligible(row)]
    if len(rows) != settings.source["expected_audio_only_rows"]:
        raise ContractError("Claim 4 CREMA-D audio-only vote count changed")
    if len(eligible) != settings.source["expected_eligible_rows"] or len({cremad_actor_id(row.file_name) for row in eligible}) != settings.source["expected_eligible_actors"]:
        raise ContractError("Claim 4 CREMA-D eligible-item/actor inventory changed")
    manifest = build_claim4_manifest(rows, actor_count=settings.actor_count, seed=config.seed)
    if len(manifest) != settings.expected_samples or len({item.actor_id for item in manifest}) != settings.actor_count:
        raise ContractError("Claim 4 full manifest accounting changed")
    write_manifest(run_root / "claim4_manifest_full.json", manifest)
    assets = load_claim4_frozen_assets(settings, repo=repo, manifest=manifest)

    vectors = load_claim4_vectors(settings, repo=repo)
    from repro.gpu_baseline import _prepare_cosyvoice, _prepare_model

    asset_settings = {
        "source_url": settings.model["source_url"], "source_revision": settings.model["source_revision"],
        "model_id": settings.model["id"], "model_revision": settings.model["revision"],
    }
    cosyvoice_root = _prepare_cosyvoice(repo, asset_settings)
    model_path = _prepare_model(repo, asset_settings)
    os.environ["COSYVOICE_ROOT"] = str(cosyvoice_root)
    try:
        from cocoemo.backbones import cosyvoice2
    except ImportError as exc:
        raise ContractError("Claim 4 full runner could not import the pinned CosyVoice2 adapter") from exc
    if getattr(cosyvoice2, "generate_steered_speech", None) is None:
        raise ContractError("Claim 4 required CosyVoice2 generation API is unavailable")
    model = cosyvoice2.load_model(str(model_path))
    _require_claim4_model_sample_rate(
        model=model, expected_sample_rate=settings.audio_quality["sample_rate_hz"]
    )

    output_dir = run_root / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = run_root / "claim4_generation.jsonl"
    cache_root = repo / ".cache" / "cremad" / settings.source["revision"] / "AudioWAV"
    generated: list[dict[str, Any]] = []
    started = time.monotonic()
    for item in manifest:
        target_wav, reference_wav, target_asset, reference_asset = materialize_claim4_wav_pair(
            cache_root=cache_root, item=item, assets=assets,
        )
        if target_wav == reference_wav or item.reference_provided_label != "N" or item.reference_majority_labels != ("N",):
            raise ContractError("Claim 4 full runner must use a distinct same-actor neutral reference")
        arms = build_claim4_arms(vectors, item.distribution, seed=config.seed, sample_id=item.file_name)
        target_text, reference_text = claim4_target_and_reference_text(item)
        for arm_name, arm in arms.items():
            output_path = output_dir / f"{item.file_name}__{arm_name}.wav"
            steering_vectors = {
                settings.vector_contract["operation"]: {
                    settings.vector_contract["layer"]: torch.from_numpy(arm["vector"]).to(torch.float32)
                }
            }
            reset_claim4_generation_seed(arm["sample_seed"])
            arm_started = time.monotonic()
            result = cosyvoice2.generate_steered_speech(
                model=model, text=target_text, reference_audio_path=str(reference_wav), prompt_text=reference_text,
                steering_vectors=steering_vectors, layers=[settings.vector_contract["layer"]], alpha=settings.alpha,
                operations=[settings.vector_contract["operation"]], output_path=str(output_path),
            )
            if not isinstance(result, dict) or result.get("audio") is None:
                raise ContractError(f"Claim 4 model API returned no audio for {item.file_name}/{arm_name}")
            integrity = _check_generated_wav(
                output_path, max_clipping_fraction=settings.audio_quality["max_clipping_fraction"],
                expected_sample_rate=settings.audio_quality["sample_rate_hz"],
            )
            record = {
                "actor_id": item.actor_id, "sample_id": item.file_name, "arm": arm_name,
                "sample_seed": arm["sample_seed"], "vector_l2_norm": arm["l2_norm"],
                "steering_vector_sha256": hashlib.sha256(np.asarray(arm["vector"], dtype=np.float32).tobytes()).hexdigest(),
                "generation_seconds": time.monotonic() - arm_started,
                "target_git_blob_sha1": target_asset.git_blob_sha1,
                "reference_git_blob_sha1": reference_asset.git_blob_sha1,
                "target_lfs_oid_sha256": target_asset.lfs_oid_sha256,
                "reference_lfs_oid_sha256": reference_asset.lfs_oid_sha256,
                "target_wav": str(target_wav.relative_to(repo)), "reference_wav": str(reference_wav.relative_to(repo)),
                "target_text": target_text, "reference_text": reference_text,
                "target_distribution": item.distribution, "reference_majority_labels": list(item.reference_majority_labels),
                "output_wav": str(output_path.relative_to(run_root)), **integrity,
            }
            generated.append(record)
            _append_jsonl(generation_path, record)
    require_complete_counts(requested=settings.expected_rendered_wavs, generated=len(generated), evaluated=len(generated))
    hash_audit = audit_claim4_wav_hashes(generated)

    evaluators = load_frozen_claim4_evaluators(settings)
    by_sample_arm = {(record["sample_id"], record["arm"]): record for record in generated}
    if len(by_sample_arm) != len(generated):
        raise ContractError("Claim 4 generation records are not unique")
    evaluation_path = run_root / "claim4_per_sample.jsonl"
    evaluated: list[dict[str, Any]] = []
    for item in manifest:
        identity = by_sample_arm.get((item.file_name, "identity_neutral_control"))
        if identity is None:
            raise ContractError(f"Claim 4 is missing identity baseline for {item.file_name}")
        identity_metrics = evaluators.evaluate(
            wav_path=run_root / identity["output_wav"], reference_wav=repo / identity["reference_wav"],
            reference_text=identity["target_text"],
        )
        for arm_name in CLAIM4_ARM_NAMES:
            generated_record = by_sample_arm.get((item.file_name, arm_name))
            if generated_record is None:
                raise ContractError(f"Claim 4 is missing generated arm {item.file_name}/{arm_name}")
            metrics = identity_metrics if arm_name == "identity_neutral_control" else evaluators.evaluate(
                wav_path=run_root / generated_record["output_wav"], reference_wav=repo / generated_record["reference_wav"],
                reference_text=generated_record["target_text"],
            )
            if arm_name == "identity_neutral_control":
                # A probability vector minus itself has zero variance, so rho
                # is mathematically undefined.  The complete-matrix contract
                # needs finite placeholders, while the decision function uses
                # this arm only for WavLM and WER safety comparisons.
                rho, h_rate = 0.0, 0.0
                proportion_metric_status = "identity_self_baseline_not_applicable"
            else:
                rho, h_rate = claim4_proportion_metrics(
                    target_distribution=item.distribution,
                    arm_probabilities=metrics["emotion_probabilities"],
                    identity_probabilities=identity_metrics["emotion_probabilities"],
                )
                proportion_metric_status = "defined_against_identity_neutral_control"
            record = {
                **generated_record,
                **metrics,
                "proportion_spearman_rho": rho,
                "h_rate": h_rate,
                "proportion_metric_status": proportion_metric_status,
            }
            evaluated.append(record)
            _append_jsonl(evaluation_path, record)
    require_complete_counts(requested=settings.expected_rendered_wavs, generated=len(generated), evaluated=len(evaluated))
    decision = decide_claim4_directional_proxy(
        evaluated,
        bootstrap_replicates=settings.decisions["bootstrap_replicates"], bootstrap_seed=settings.decisions["bootstrap_seed"],
        min_rho_advantage=settings.decisions["min_rho_advantage"], min_h_rate_advantage=settings.decisions["min_h_rate_advantage"],
        min_rho_vs_dominant=settings.decisions["min_rho_vs_dominant"], max_wer_increase=settings.decisions["max_wer_increase"],
        min_speaker_similarity_change=settings.decisions["min_speaker_similarity_change"],
    )
    return {
        "stage": "claim4", "claim_evidence": True, "verdict": decision.verdict,
        "requested_rendered_wavs": settings.expected_rendered_wavs, "generated_wavs": len(generated),
        "evaluated_wavs": len(evaluated), "wav_hash_audit": hash_audit,
        "canary_receipt": canary_receipt,
        "license_attestations": settings.license_attestations,
        "decision": {"comparisons": decision.comparisons, "rule": decision.rule},
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "This is a public 60-actor CREMA-D proxy, not the paper's unreleased 772-item manifest or IEMOCAP result.",
            "No naturalness claim can be verified without blinded human ratings.",
        ],
    }
