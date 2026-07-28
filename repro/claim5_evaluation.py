"""Metric and verdict contracts for Claim 5's categorical public proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from repro.claim5_protocol import CLAIM5_ARMS, TARGETS
from repro.contracts import ContractError
from repro.stats import BootstrapResult


PRIMARY_METRICS = ("target_emotion_probability", "leave_one_actor_out_esim")
SPECIFICITY_ARMS = (
    "wrong_target_alpha6", "negative_target_alpha6", "random_norm_matched_alpha6", "flow_side_emotional_reference",
)


def l2_normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size < 2 or not np.all(np.isfinite(vector)):
        raise ContractError("Claim 5 embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ContractError("Claim 5 embedding must have nonzero L2 norm")
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    x, y = l2_normalize(left), l2_normalize(right)
    if x.shape != y.shape:
        raise ContractError("Claim 5 cosine similarity requires equal embedding shapes")
    value = float(np.dot(x, y))
    if not np.isfinite(value) or not -1.000001 <= value <= 1.000001:
        raise ContractError("Claim 5 cosine similarity is invalid")
    return value


def leave_one_actor_out_prototypes(reference_embeddings: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    """Build exact target prototypes from normalized source RAVDESS embeddings."""

    by_target: dict[str, list[tuple[str, np.ndarray]]] = {target: [] for target in TARGETS}
    actors: set[str] = set()
    for record in reference_embeddings:
        actor, target, embedding = record.get("actor_id"), record.get("target"), record.get("embedding")
        if not isinstance(actor, str) or not isinstance(target, str) or target not in TARGETS:
            raise ContractError("Claim 5 reference embedding needs valid actor_id and target")
        if any(existing == actor for existing, _ in by_target[target]):
            raise ContractError(f"Claim 5 has duplicate frozen reference embedding: {actor}/{target}")
        by_target[target].append((actor, l2_normalize(np.asarray(embedding))))
        actors.add(actor)
    if len(actors) != 24 or any(len(by_target[target]) != 24 for target in TARGETS):
        raise ContractError("Claim 5 LOAO prototypes require 24 actors for every target")
    output: dict[tuple[str, str], np.ndarray] = {}
    for target, entries in by_target.items():
        for actor in sorted(actors):
            remaining = [embedding for candidate, embedding in entries if candidate != actor]
            if len(remaining) != 23:
                raise ContractError("Claim 5 LOAO prototype exclusion changed")
            output[(actor, target)] = l2_normalize(np.mean(np.stack(remaining), axis=0))
    return output


def require_complete_claim5_metric_matrix(records: Sequence[Mapping[str, Any]], *, expected_cells: int = 96) -> None:
    cells: dict[str, set[str]] = {}
    hashes: set[str] = set()
    required = set(PRIMARY_METRICS + ("same_actor_target_esim", "wavlm_speaker_similarity", "whisper_wer"))
    for record in records:
        cell, arm = record.get("cell_id"), record.get("arm")
        if not isinstance(cell, str) or not cell or arm not in CLAIM5_ARMS:
            raise ContractError("Claim 5 metric matrix has an invalid cell or arm")
        arms = cells.setdefault(cell, set())
        if arm in arms:
            raise ContractError(f"Claim 5 metric matrix has a duplicate cell/arm: {cell}/{arm}")
        arms.add(arm)
        audio = record.get("audio")
        digest = audio.get("sha256") if isinstance(audio, Mapping) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise ContractError("Claim 5 metric matrix requires every generated WAV hash")
        if digest in hashes:
            raise ContractError("Claim 5 generated WAV collision detected")
        hashes.add(digest)
        missing = required.difference(record)
        if missing:
            raise ContractError(f"Claim 5 metric matrix misses required metrics: {sorted(missing)}")
        for metric in required:
            value = record[metric]
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                raise ContractError(f"Claim 5 metric {metric} is missing or non-finite")
    expected = set(CLAIM5_ARMS)
    incomplete = {cell: sorted(expected - arms) for cell, arms in cells.items() if arms != expected}
    if incomplete or len(cells) != expected_cells or len(records) != expected_cells * len(CLAIM5_ARMS):
        raise ContractError("Claim 5 full metric matrix must contain every cell and all eight arms")


@dataclass(frozen=True)
class Claim5Decision:
    verdict: str
    comparisons: dict[str, dict[str, float | int]]
    rule: str


def actor_cluster_bootstrap(
    records: Sequence[Mapping[str, Any]], *, left_arm: str, right_arm: str, metric: str,
    seed: int, replicates: int, target: str | None = None,
) -> BootstrapResult:
    """Bootstrap actor-level paired effects over the observed balanced cells.

    Each actor is the independent unit.  For a target-specific contrast there
    is one matched cell per actor; for a global contrast, each actor's four
    preassigned cells are averaged before resampling.  The frozen modulo-four
    allocation balances every content family across six actors, so content is
    represented descriptively in each actor-level estimate rather than being
    resampled as a sparse second axis.
    """

    if replicates < 100:
        raise ContractError("Claim 5 actor-cluster bootstrap requires at least 100 replicates")
    by_cell: dict[tuple[str, str], dict[str, float]] = {}
    for row in records:
        if target is not None and row.get("target") != target:
            continue
        actor, cell, arm = row.get("actor_id"), row.get("cell_id"), row.get("arm")
        value = row.get(metric)
        if not isinstance(actor, str) or not actor or not isinstance(cell, str) or not cell or not isinstance(arm, str):
            raise ContractError("Claim 5 actor-cluster bootstrap rows require actor_id, cell_id, and arm")
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise ContractError(f"Claim 5 actor-cluster bootstrap has non-finite {metric}")
        values = by_cell.setdefault((actor, cell), {})
        if arm in values:
            raise ContractError(f"Claim 5 actor-cluster bootstrap has duplicate arm: {actor}/{cell}/{arm}")
        values[arm] = float(value)
    by_actor: dict[str, list[float]] = {}
    for (actor, cell), values in by_cell.items():
        if left_arm not in values or right_arm not in values:
            raise ContractError(f"Claim 5 actor-cluster bootstrap has incomplete paired arms: {actor}/{cell}")
        by_actor.setdefault(actor, []).append(values[left_arm] - values[right_arm])
    actors = sorted(by_actor)
    expected_cells_per_actor = 1 if target is not None else 4
    if len(actors) != 24 or any(len(by_actor[actor]) != expected_cells_per_actor for actor in actors):
        raise ContractError(
            "Claim 5 actor-cluster bootstrap requires 24 actors with "
            f"{expected_cells_per_actor} observed paired cells each"
        )
    actor_effects = np.asarray([np.mean(by_actor[actor]) for actor in actors], dtype=np.float64)
    if not np.all(np.isfinite(actor_effects)):
        raise ContractError("Claim 5 actor-cluster bootstrap produced a non-finite actor effect")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draws[index] = float(np.mean(actor_effects[rng.integers(0, len(actor_effects), size=len(actor_effects))]))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return BootstrapResult(
        estimate=float(np.mean(actor_effects)), lower=float(lower), upper=float(upper),
        replicates=replicates, seed=seed,
    )


def _comparison(records: Sequence[Mapping[str, Any]], *, left: str, right: str, metric: str, seed: int, replicates: int, target: str | None = None) -> BootstrapResult:
    return actor_cluster_bootstrap(records, left_arm=left, right_arm=right, metric=metric, seed=seed, replicates=replicates, target=target)


def decide_claim5_proxy(
    records: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int,
    min_primary_effect: float, min_wavlm_change: float, max_wer_increase: float,
) -> Claim5Decision:
    """Apply the frozen Claim 5 decision rule with all target/global contrasts."""

    require_complete_claim5_metric_matrix(records)
    comparisons: dict[str, BootstrapResult] = {}
    offset = 0
    for target in TARGETS:
        for metric in PRIMARY_METRICS:
            comparisons[f"{target}_{metric}_alpha6_minus_instruction"] = _comparison(records, left="cocoemo_alpha6", right="instruction_target", metric=metric, seed=bootstrap_seed + offset, replicates=bootstrap_replicates, target=target)
            offset += 1
    for metric in PRIMARY_METRICS:
        comparisons[f"global_{metric}_alpha6_minus_no_steer"] = _comparison(records, left="cocoemo_alpha6", right="no_steer", metric=metric, seed=bootstrap_seed + offset, replicates=bootstrap_replicates)
        offset += 1
        for control in SPECIFICITY_ARMS:
            comparisons[f"global_{metric}_alpha6_minus_{control}"] = _comparison(records, left="cocoemo_alpha6", right=control, metric=metric, seed=bootstrap_seed + offset, replicates=bootstrap_replicates)
            offset += 1
        comparisons[f"global_{metric}_alpha6_minus_alpha3_diagnostic"] = _comparison(records, left="cocoemo_alpha6", right="cocoemo_alpha3", metric=metric, seed=bootstrap_seed + offset, replicates=bootstrap_replicates)
        offset += 1
    comparisons["global_wavlm_alpha6_minus_no_steer"] = _comparison(records, left="cocoemo_alpha6", right="no_steer", metric="wavlm_speaker_similarity", seed=bootstrap_seed + offset, replicates=bootstrap_replicates)
    offset += 1
    comparisons["global_wer_alpha6_minus_no_steer"] = _comparison(records, left="cocoemo_alpha6", right="no_steer", metric="whisper_wer", seed=bootstrap_seed + offset, replicates=bootstrap_replicates)

    target_primary = [comparisons[f"{target}_{metric}_alpha6_minus_instruction"] for target in TARGETS for metric in PRIMARY_METRICS]
    global_primary = [comparisons[f"global_{metric}_alpha6_minus_no_steer"] for metric in PRIMARY_METRICS]
    specificity = [comparisons[f"global_{metric}_alpha6_minus_{control}"] for metric in PRIMARY_METRICS for control in SPECIFICITY_ARMS]
    wavlm = comparisons["global_wavlm_alpha6_minus_no_steer"]
    wer = comparisons["global_wer_alpha6_minus_no_steer"]
    supported = all(result.lower > min_primary_effect for result in target_primary + global_primary + specificity) and wavlm.lower >= min_wavlm_change and wer.upper <= max_wer_increase
    reverse_by_target = any(
        all(comparisons[f"{target}_{metric}_alpha6_minus_instruction"].upper < min_primary_effect for metric in PRIMARY_METRICS)
        for target in TARGETS
    )
    quality_breach = wavlm.upper < min_wavlm_change or wer.lower > max_wer_increase
    verdict = "directional_supported" if supported else "directional_failed" if reverse_by_target or quality_breach else "directional_inconclusive"
    return Claim5Decision(
        verdict=verdict,
        comparisons={name: asdict(result) for name, result in comparisons.items()},
        rule=(
            "Supported iff every target has positive lower 95% actor-cluster bootstrap bounds for "
            "target-emotion probability and leave-one-actor-out E-SIM against instruction, both metrics are "
            "positive globally against no-steer and every predeclared specificity control, and WavLM/WER stay "
            "within the predeclared quality margins. Failed iff both primary instruction contrasts confidently "
            "reverse for any target or either quality margin confidently breaches; otherwise inconclusive. "
            "This categorical text-target proxy is not an exact Table 3 result."
        ),
    )
