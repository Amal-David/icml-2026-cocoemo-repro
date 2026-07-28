"""GPU execution for the frozen public Claim 5 categorical proxy."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from repro.claim4_evaluators import load_frozen_claim4_evaluators
from repro.claim5_evaluation import cosine_similarity, decide_claim5_proxy, leave_one_actor_out_prototypes, require_complete_claim5_metric_matrix
from repro.claim5_protocol import CLAIM5_ARMS, Claim5Cell, build_claim5_arms, build_claim5_cells, load_claim5_catalog, load_claim5_vectors, parse_claim5_settings
from repro.claim5_runtime import render_claim5_arm, reset_claim5_generation_seed
from repro.config import ReproConfig
from repro.contracts import ContractError, require_complete_counts, sha256_file
from repro.ravdess_acquisition import ensure_ravdess_dataset
from repro.ravdess_manifest import STATEMENTS, build_claim1_groups


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _draw_seed(base_seed: int, cell_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"claim5-draw-v1\0{base_seed}\0{cell_id}".encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def _vector_hashes(arm: dict[str, Any]) -> dict[str, str] | None:
    vectors = arm.get("vectors")
    if not isinstance(vectors, dict):
        return None
    return {str(layer): hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest() for layer, vector in vectors.items()}


def _cell_manifest(cell: Claim5Cell) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id, "actor_id": cell.actor_id, "target": cell.target,
        "statement": cell.statement, "repetition": cell.repetition, "content_id": cell.content_id,
        "text": cell.text, "neutral_reference": cell.neutral_reference,
        "emotional_reference": cell.emotional_reference,
    }


def _require_sample_rate(*, configured_rate: int, cosyvoice: Any) -> int:
    loaded_rate = getattr(cosyvoice, "sample_rate", None)
    if not isinstance(loaded_rate, int) or isinstance(loaded_rate, bool) or loaded_rate < 1:
        raise ContractError("Claim 5 loaded CosyVoice model has no valid integer sample_rate")
    if loaded_rate != configured_rate:
        raise ContractError(
            "Claim 5 configured audio sample rate differs from loaded CosyVoice sample rate: "
            f"configured={configured_rate}, loaded={loaded_rate}"
        )
    return loaded_rate


def _require_generated_sample_rate(*, audio: dict[str, Any], expected_rate: int) -> None:
    if audio.get("sample_rate") != expected_rate:
        raise ContractError(
            "Claim 5 generated WAV sample rate differs from the frozen configured rate: "
            f"expected={expected_rate}, actual={audio.get('sample_rate')}"
        )


def run_claim5_runtime(config: ReproConfig, *, repo: Path, run_root: Path) -> dict[str, Any]:
    settings = parse_claim5_settings(config)
    if run_root.exists() and any(run_root.iterdir()):
        raise ContractError(
            "Claim 5 run root is nonempty without a completed immutable result; "
            "refusing to append stale generation or evaluation records"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except ImportError as exc:
        raise ContractError("Claim 5 runtime requires torch with CUDA") from exc
    if not torch.cuda.is_available():
        raise ContractError("Claim 5 runtime requires CUDA; CPU execution is not claim evidence")
    catalog = load_claim5_catalog(repo=repo, settings=settings)
    root = ensure_ravdess_dataset(repo=repo, settings=settings.ravdess)
    groups = build_claim1_groups(sorted(root.rglob("*.wav")))
    full_cells = build_claim5_cells(
        groups, catalog=catalog, actor_ids=tuple(f"{actor:02d}" for actor in range(1, 25))
    )
    cells = [cell for cell in full_cells if cell.actor_id in set(settings.actor_ids)]
    if len(cells) != settings.expected_cells:
        raise ContractError("Claim 5 cell accounting changed after RAVDESS construction")
    (run_root / "claim5_cells.json").write_text(json.dumps([_cell_manifest(cell) for cell in cells], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vectors = load_claim5_vectors(repo=repo, settings=settings)

    from repro.gpu_baseline import _prepare_cosyvoice, _prepare_model
    assets = {
        "source_url": settings.model["source_url"], "source_revision": settings.model["source_revision"],
        "model_id": settings.model["id"], "model_revision": settings.model["revision"],
    }
    cosyvoice_root = _prepare_cosyvoice(repo, assets)
    model_path = _prepare_model(repo, assets)
    os.environ["COSYVOICE_ROOT"] = str(cosyvoice_root)
    try:
        from cocoemo.backbones import cosyvoice2
    except ImportError as exc:
        raise ContractError("Claim 5 cannot import the pinned CoCoEmo CosyVoice2 adapter") from exc
    cosyvoice = cosyvoice2.load_model(str(model_path))
    cosyvoice.model.eval()
    sample_rate = _require_sample_rate(
        configured_rate=settings.audio_quality["sample_rate_hz"], cosyvoice=cosyvoice
    )
    evaluator = load_frozen_claim4_evaluators(settings)

    # Freeze the source emotional-reference embedding bank before evaluating
    # generated outputs. Each prototype will exclude the generated actor.
    reference_records: list[dict[str, Any]] = []
    for cell in full_cells:
        reference_path = Path(cell.emotional_reference)
        if not reference_path.is_file():
            raise ContractError(f"Claim 5 emotional reference is missing: {reference_path}")
        reference_records.append({"actor_id": cell.actor_id, "target": cell.target, "embedding": evaluator.emotion_embedding(reference_path).tolist(), "reference_sha256": sha256_file(reference_path)})
    prototypes = leave_one_actor_out_prototypes(reference_records)
    reference_embeddings_path = run_root / "claim5_reference_embeddings.jsonl"
    reference_embeddings_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in reference_records),
        encoding="utf-8",
    )
    # The vectors are required to reproduce the local LOAO calculation, but
    # are not public artifacts.  Publish only the bank's cardinality and a
    # digest that binds the hidden local computation to the derived metrics.
    (run_root / "claim5_reference_embedding_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "aggregate_and_hash_only_no_raw_embeddings",
                "source_cells": len(full_cells),
                "source_embeddings": len(reference_records),
                "reference_embeddings_sha256": sha256_file(reference_embeddings_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = run_root / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = run_root / "claim5_generation.jsonl"
    generated: list[dict[str, Any]] = []
    started = time.monotonic()
    for cell in cells:
        neutral, emotional = Path(cell.neutral_reference), Path(cell.emotional_reference)
        if not neutral.is_file() or not emotional.is_file() or neutral == emotional:
            raise ContractError(f"Claim 5 has an invalid same-group reference pair: {cell.cell_id}")
        prompt_text = STATEMENTS[cell.statement]
        arms = build_claim5_arms(vectors, target=cell.target, base_seed=config.seed, cell_id=cell.cell_id)
        draw_seed = _draw_seed(config.seed, cell.cell_id)
        for arm_name in CLAIM5_ARMS:
            arm = arms[arm_name]
            reset_claim5_generation_seed(draw_seed)
            output_path = output_dir / f"{cell.cell_id}__{arm_name}.wav"
            audio = render_claim5_arm(
                cosyvoice=cosyvoice, backbone=cosyvoice2, arm_name=arm_name, arm=arm, text=cell.text,
                neutral_reference=neutral, emotional_reference=emotional, prompt_text=prompt_text,
                output_path=output_path, sample_rate=sample_rate,
                max_clipping_fraction=settings.audio_quality["max_clipping_fraction"],
            )
            _require_generated_sample_rate(audio=audio, expected_rate=sample_rate)
            record = {
                **_cell_manifest(cell), "arm": arm_name, "draw_seed": draw_seed,
                "neutral_reference_sha256": sha256_file(neutral), "emotional_reference_sha256": sha256_file(emotional),
                "vector_hashes": _vector_hashes(arm), "output_wav": str(output_path.relative_to(run_root)), "audio": audio,
            }
            generated.append(record)
            _append(generation_path, record)
    require_complete_counts(requested=settings.expected_rendered_wavs, generated=len(generated), evaluated=len(generated))

    index = {(record["cell_id"], record["arm"]): record for record in generated}
    if len(index) != len(generated):
        raise ContractError("Claim 5 generation records have duplicate cell/arm entries")
    evaluated: list[dict[str, Any]] = []
    evaluation_path = run_root / "claim5_per_sample.jsonl"
    for cell in cells:
        same_actor_embedding = next(np.asarray(item["embedding"], dtype=np.float32) for item in reference_records if item["actor_id"] == cell.actor_id and item["target"] == cell.target)
        prototype = prototypes[(cell.actor_id, cell.target)]
        for arm_name in CLAIM5_ARMS:
            generated_record = index.get((cell.cell_id, arm_name))
            if generated_record is None:
                raise ContractError(f"Claim 5 evaluator is missing generated arm: {cell.cell_id}/{arm_name}")
            wav_path = run_root / str(generated_record["output_wav"])
            metrics = evaluator.evaluate(wav_path=wav_path, reference_wav=Path(cell.neutral_reference), reference_text=cell.text)
            embedding = evaluator.emotion_embedding(wav_path)
            target_probability = metrics["emotion_probabilities"].get(cell.target)
            if not isinstance(target_probability, (int, float)):
                raise ContractError("Claim 5 Emotion2Vec output lacks the configured target probability")
            record = {
                **generated_record, **metrics,
                "target_emotion_probability": float(target_probability),
                "same_actor_target_esim": cosine_similarity(embedding, same_actor_embedding),
                "leave_one_actor_out_esim": cosine_similarity(embedding, prototype),
            }
            evaluated.append(record)
            _append(evaluation_path, record)
    require_complete_counts(requested=settings.expected_rendered_wavs, generated=len(generated), evaluated=len(evaluated))
    if settings.mode == "full":
        require_complete_claim5_metric_matrix(evaluated, expected_cells=settings.expected_cells)
        decision = decide_claim5_proxy(
            evaluated, bootstrap_replicates=settings.decisions["bootstrap_replicates"],
            bootstrap_seed=settings.decisions["bootstrap_seed"], min_primary_effect=settings.decisions["min_primary_effect"],
            min_wavlm_change=settings.decisions["min_wavlm_change"], max_wer_increase=settings.decisions["max_wer_increase"],
        )
        verdict, decision_payload = decision.verdict, {"comparisons": decision.comparisons, "rule": decision.rule}
    else:
        verdict, decision_payload = "not_evaluated", {"rule": "Canary verifies all four target paths and eight-arm accounting; it is not Claim 5 evidence."}
    return {
        "stage": "claim5", "claim_evidence": settings.mode == "full", "verdict": verdict,
        "requested_rendered_wavs": settings.expected_rendered_wavs, "generated_wavs": len(generated),
        "evaluated_wavs": len(evaluated), "decision": decision_payload,
        "reference_embedding_bank": {
            "source_cells": len(full_cells), "source_embeddings": len(reference_records),
            "purpose": "frozen full 24-actor leave-one-actor-out prototype bank",
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "This is a public RAVDESS categorical text-target semantic-conflict proxy, not an exact Table 3 reconstruction.",
            "The author-owned text catalog supplies controlled categories; it is not an automated semantic or VA screen.",
            "Naturalness remains unmeasured without blinded human listeners.",
        ],
    }
