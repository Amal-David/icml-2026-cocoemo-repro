"""Pure mixing and statistical decision logic for the bounded Claim 4 proxy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from repro.claim4_protocol import CLAIM4_ARM_NAMES
from repro.contracts import ContractError, require_metrics
from repro.stats import BootstrapResult, paired_bootstrap


NON_NEUTRAL_EMOTIONS = ("angry", "happy", "sad", "surprise")
TARGET_KEYS = tuple(f"p_{emotion}" for emotion in (*NON_NEUTRAL_EMOTIONS, "neutral"))


@dataclass(frozen=True)
class Claim4Decision:
    verdict: str
    comparisons: dict[str, dict[str, float | int]]
    rule: str


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    if not isinstance(base_seed, int) or not sample_id:
        raise ContractError("Claim 4 requires an integer seed and non-empty sample identifier")
    digest = hashlib.sha256(f"claim4:{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _distribution(distribution: Mapping[str, float]) -> dict[str, float]:
    unexpected = sorted(set(distribution).difference(TARGET_KEYS))
    missing = sorted(set(TARGET_KEYS).difference(distribution))
    if unexpected or missing:
        raise ContractError(f"invalid Claim 4 distribution: missing={missing}, unexpected={unexpected}")
    result = {key: float(distribution[key]) for key in TARGET_KEYS}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ContractError("Claim 4 distributions must be finite and non-negative")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError("Claim 4 distributions must sum to one")
    return result


def _vectors(vectors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if tuple(vectors) != NON_NEUTRAL_EMOTIONS:
        raise ContractError(
            "Claim 4 vector map must contain exactly " + ", ".join(NON_NEUTRAL_EMOTIONS)
        )
    result = {emotion: np.asarray(vectors[emotion], dtype=np.float64) for emotion in NON_NEUTRAL_EMOTIONS}
    first = result[NON_NEUTRAL_EMOTIONS[0]]
    if first.ndim < 1 or first.size == 0 or not np.all(np.isfinite(first)):
        raise ContractError("Claim 4 steering vectors must be non-empty finite arrays")
    if any(value.shape != first.shape or not np.all(np.isfinite(value)) for value in result.values()):
        raise ContractError("Claim 4 steering vectors must be finite and share one shape")
    return result


def _linear_mix(vectors: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    return sum((float(weights[emotion]) * vectors[emotion] for emotion in NON_NEUTRAL_EMOTIONS), np.zeros_like(vectors["angry"]))


def _norm_match(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    candidate_norm = float(np.linalg.norm(vector.reshape(-1)))
    if not math.isfinite(reference_norm) or not math.isfinite(candidate_norm) or candidate_norm == 0:
        raise ContractError("cannot norm-match a zero or non-finite Claim 4 control vector")
    return vector * (reference_norm / candidate_norm)


def deterministic_nonfixed_shuffle(weights: Mapping[str, float], *, seed: int, sample_id: str) -> dict[str, float]:
    """Reassign non-neutral weights through a deterministic derangement."""

    order = sorted(
        NON_NEUTRAL_EMOTIONS,
        key=lambda emotion: hashlib.sha256(f"{seed}:{sample_id}:{emotion}".encode("utf-8")).hexdigest(),
    )
    shuffled = {emotion: 0.0 for emotion in NON_NEUTRAL_EMOTIONS}
    for index, source in enumerate(order):
        destination = order[(index + 1) % len(order)]
        shuffled[destination] = float(weights[source])
    if any(shuffled[emotion] != float(weights[emotion]) for emotion in NON_NEUTRAL_EMOTIONS):
        # Equal-valued probabilities can make a valid non-fixed label mapping
        # numerically identical.  The arm metadata records the label mapping.
        pass
    return shuffled


def build_claim4_arms(
    vectors: Mapping[str, np.ndarray], distribution: Mapping[str, float], *, seed: int, sample_id: str
) -> dict[str, dict[str, Any]]:
    """Build the seven frozen steering arms for one input sample.

    All stochastic controls derive from the same sample seed.  The returned
    arrays are not model inputs yet; model adapters must preserve the contract
    and record their hashes before synthesis.
    """

    checked_vectors = _vectors(vectors)
    probs = _distribution(distribution)
    sample_seed = stable_sample_seed(seed, sample_id)
    raw_weights = {emotion: probs[f"p_{emotion}"] for emotion in NON_NEUTRAL_EMOTIONS}
    released = _linear_mix(checked_vectors, raw_weights)
    released_norm = float(np.linalg.norm(released.reshape(-1)))
    if released_norm == 0:
        raise ContractError("released mixed vector has zero norm; bounded Claim 4 protocol cannot continue")
    dominant = max(NON_NEUTRAL_EMOTIONS, key=lambda emotion: (raw_weights[emotion], emotion))
    shuffled_weights = deterministic_nonfixed_shuffle(raw_weights, seed=sample_seed, sample_id=sample_id)
    rng = np.random.default_rng(sample_seed)
    random_control = rng.standard_normal(released.shape)
    retained_mass = sum(raw_weights.values())
    if retained_mass <= 0:
        raise ContractError("renormalized diagnostic is undefined for all-neutral target distributions")

    result = {
        "released_four_way": {"vector": released, "semantics": "released_unrenormalized_four_way"},
        "dominant_non_neutral": {
            "vector": _norm_match(checked_vectors[dominant], released),
            "dominant_emotion": dominant,
            "semantics": "norm_matched_dominant_non_neutral",
        },
        "shuffled_distribution": {
            "vector": _norm_match(_linear_mix(checked_vectors, shuffled_weights), released),
            "weights": shuffled_weights,
            "semantics": "norm_matched_nonfixed_label_shuffle",
        },
        "random_norm_matched": {
            "vector": _norm_match(random_control, released),
            "semantics": "sample_seeded_norm_matched_random",
        },
        "renormalized_non_neutral_diagnostic": {
            "vector": released / retained_mass,
            "retained_non_neutral_mass": retained_mass,
            "semantics": "diagnostic_non_neutral_renormalization",
        },
        "neutral_zero_vector_diagnostic": {
            "vector": released.copy(),
            "semantics": "diagnostic_explicit_zero_neutral_equivalent",
        },
        "identity_neutral_control": {
            "vector": np.zeros_like(released),
            "semantics": "zero_steering_identity_control",
        },
    }
    if tuple(result) != CLAIM4_ARM_NAMES:
        raise ContractError("Claim 4 arm definition drifted from frozen protocol")
    for name, arm in result.items():
        vector = np.asarray(arm["vector"], dtype=np.float64)
        if vector.shape != released.shape or not np.all(np.isfinite(vector)):
            raise ContractError(f"Claim 4 arm {name} violates vector finiteness/shape contract")
        arm["sample_seed"] = sample_seed
        arm["l2_norm"] = float(np.linalg.norm(vector.reshape(-1)))
    return result


def actor_cluster_bootstrap(
    records: Sequence[Mapping[str, Any]], *, left_arm: str, right_arm: str, metric: str,
    replicates: int, seed: int,
) -> BootstrapResult:
    """Bootstrap matched arm differences after enforcing one row per actor/arm."""

    require_metrics(records, ("actor_id", "arm", metric))
    by_actor: dict[str, dict[str, float]] = {}
    for record in records:
        actor = str(record["actor_id"])
        arm = str(record["arm"])
        if arm not in CLAIM4_ARM_NAMES:
            raise ContractError(f"unexpected Claim 4 arm in metric records: {arm}")
        value = float(record[metric])
        if not math.isfinite(value):
            raise ContractError(f"non-finite {metric} for actor={actor}, arm={arm}")
        if arm in by_actor.setdefault(actor, {}):
            raise ContractError(f"duplicate Claim 4 metric row for actor={actor}, arm={arm}")
        by_actor[actor][arm] = value
    missing = [actor for actor, values in by_actor.items() if left_arm not in values or right_arm not in values]
    if missing:
        raise ContractError("missing paired Claim 4 metrics for actors: " + ", ".join(sorted(missing)))
    actors = sorted(by_actor)
    return paired_bootstrap(
        [by_actor[actor][left_arm] for actor in actors],
        [by_actor[actor][right_arm] for actor in actors],
        replicates=replicates,
        seed=seed,
    )


def require_complete_claim4_metric_matrix(
    records: Sequence[Mapping[str, Any]], *, expected_actor_count: int = 60
) -> None:
    """Reject a full verdict unless it has exactly 60 actors x seven arms."""

    by_actor: dict[str, set[str]] = {}
    for record in records:
        actor = str(record.get("actor_id", ""))
        arm = str(record.get("arm", ""))
        if not actor or arm not in CLAIM4_ARM_NAMES:
            raise ContractError("Claim 4 metric matrix has an invalid actor or arm")
        arms = by_actor.setdefault(actor, set())
        if arm in arms:
            raise ContractError(f"duplicate Claim 4 metric matrix arm: actor={actor}, arm={arm}")
        arms.add(arm)
    expected = set(CLAIM4_ARM_NAMES)
    missing = {actor: sorted(expected - arms) for actor, arms in by_actor.items() if arms != expected}
    if missing:
        raise ContractError(f"incomplete Claim 4 metric matrix: {missing}")
    if len(by_actor) != expected_actor_count or len(records) != expected_actor_count * len(CLAIM4_ARM_NAMES):
        raise ContractError(
            "Claim 4 full metric matrix must contain exactly "
            f"{expected_actor_count} actors and {expected_actor_count * len(CLAIM4_ARM_NAMES)} rows"
        )


def decide_claim4_directional_proxy(
    records: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int,
    min_rho_advantage: float, min_h_rate_advantage: float, min_rho_vs_dominant: float,
    max_wer_increase: float, min_speaker_similarity_change: float,
) -> Claim4Decision:
    """Apply the predeclared compositional-proxy decision matrix.

    ``proportion_spearman_rho`` and ``h_rate`` are evaluator outputs defined by
    the later frozen adapter. They are intentionally not substituted with a
    single target-probability score: a mixed-label claim needs proportion-rank
    and H-rate evidence against both shuffled and random controls.
    """

    if not 0.0 <= max_wer_increase < 1.0:
        raise ContractError("claim4 max WER increase must be in [0, 1)")
    if not -1.0 <= min_speaker_similarity_change <= 1.0:
        raise ContractError("claim4 minimum speaker similarity change must be in [-1, 1]")
    if not -1.0 <= min_rho_advantage <= 1.0 or not -1.0 <= min_h_rate_advantage <= 1.0 or not -1.0 <= min_rho_vs_dominant <= 1.0:
        raise ContractError("Claim 4 proportion thresholds must be in [-1, 1]")
    require_metrics(records, ("proportion_spearman_rho", "h_rate", "wavlm_speaker_similarity", "whisper_wer"))
    require_complete_claim4_metric_matrix(records)
    comparisons = {
        "rho_released_minus_shuffled": actor_cluster_bootstrap(
            records, left_arm="released_four_way", right_arm="shuffled_distribution",
            metric="proportion_spearman_rho", replicates=bootstrap_replicates, seed=bootstrap_seed,
        ),
        "rho_released_minus_random": actor_cluster_bootstrap(
            records, left_arm="released_four_way", right_arm="random_norm_matched",
            metric="proportion_spearman_rho", replicates=bootstrap_replicates, seed=bootstrap_seed + 1,
        ),
        "h_rate_released_minus_shuffled": actor_cluster_bootstrap(
            records, left_arm="released_four_way", right_arm="shuffled_distribution",
            metric="h_rate", replicates=bootstrap_replicates, seed=bootstrap_seed + 2,
        ),
        "h_rate_released_minus_random": actor_cluster_bootstrap(
            records, left_arm="released_four_way", right_arm="random_norm_matched",
            metric="h_rate", replicates=bootstrap_replicates, seed=bootstrap_seed + 3,
        ),
        "rho_released_minus_dominant": actor_cluster_bootstrap(
            records, left_arm="released_four_way", right_arm="dominant_non_neutral",
            metric="proportion_spearman_rho", replicates=bootstrap_replicates, seed=bootstrap_seed + 4,
        ),
        "speaker_released_minus_identity": actor_cluster_bootstrap(
            records, left_arm="released_four_way", right_arm="identity_neutral_control",
            metric="wavlm_speaker_similarity", replicates=bootstrap_replicates, seed=bootstrap_seed + 5,
        ),
        "wer_released_minus_identity": actor_cluster_bootstrap(
            records, left_arm="released_four_way", right_arm="identity_neutral_control",
            metric="whisper_wer", replicates=bootstrap_replicates, seed=bootstrap_seed + 6,
        ),
    }
    rho_shuffled = comparisons["rho_released_minus_shuffled"]
    rho_random = comparisons["rho_released_minus_random"]
    h_rate_shuffled = comparisons["h_rate_released_minus_shuffled"]
    h_rate_random = comparisons["h_rate_released_minus_random"]
    rho_dominant = comparisons["rho_released_minus_dominant"]
    speaker = comparisons["speaker_released_minus_identity"]
    wer = comparisons["wer_released_minus_identity"]
    supported = (
        rho_shuffled.lower > min_rho_advantage
        and rho_random.lower > min_rho_advantage
        and h_rate_shuffled.lower > min_h_rate_advantage
        and h_rate_random.lower > min_h_rate_advantage
        and rho_dominant.lower >= min_rho_vs_dominant
        and speaker.lower >= min_speaker_similarity_change
        and wer.upper <= max_wer_increase
    )
    failed = (
        rho_shuffled.upper < min_rho_advantage
        or rho_random.upper < min_rho_advantage
        or h_rate_shuffled.upper < min_h_rate_advantage
        or h_rate_random.upper < min_h_rate_advantage
        or rho_dominant.upper < min_rho_vs_dominant
        or speaker.upper < min_speaker_similarity_change
        or wer.lower > max_wer_increase
    )
    verdict = "directional_supported" if supported else "directional_failed" if failed else "directional_inconclusive"
    return Claim4Decision(
        verdict=verdict,
        comparisons={name: asdict(result) for name, result in comparisons.items()},
        rule=(
            "supported iff released mixing has actor-cluster bootstrap lower bounds above configured "
            "rho and H-rate margins against both shuffled and random controls, is non-worse than "
            "dominant on rho, preserves WavLM speaker similarity, and stays within the WER margin; "
            "failed on any confident reverse/safety breach; otherwise inconclusive. This alpha=5-only "
            "proxy does not test an alpha=5 versus alpha=3 comparison."
        ),
    )
