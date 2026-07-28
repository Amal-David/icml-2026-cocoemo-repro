from __future__ import annotations

import numpy as np
import pytest

from repro.claim4_evaluation import (
    CLAIM4_ARM_NAMES,
    build_claim4_arms,
    decide_claim4_directional_proxy,
    deterministic_nonfixed_shuffle,
    stable_sample_seed,
)
from repro.contracts import ContractError


def _vectors() -> dict[str, np.ndarray]:
    return {
        "angry": np.array([1.0, 0.0]),
        "happy": np.array([0.0, 2.0]),
        "sad": np.array([1.0, 1.0]),
        "surprise": np.array([2.0, -1.0]),
    }


def _distribution() -> dict[str, float]:
    return {"p_angry": 0.2, "p_happy": 0.3, "p_sad": 0.1, "p_surprise": 0.1, "p_neutral": 0.3}


def test_all_seven_arms_are_deterministic_and_norm_matched_where_required() -> None:
    first = build_claim4_arms(_vectors(), _distribution(), seed=5, sample_id="1001_IEO_HAP_XX")
    second = build_claim4_arms(_vectors(), _distribution(), seed=5, sample_id="1001_IEO_HAP_XX")
    assert tuple(first) == CLAIM4_ARM_NAMES
    assert first["released_four_way"]["sample_seed"] == stable_sample_seed(5, "1001_IEO_HAP_XX")
    assert np.array_equal(first["random_norm_matched"]["vector"], second["random_norm_matched"]["vector"])
    released_norm = first["released_four_way"]["l2_norm"]
    for name in ("dominant_non_neutral", "shuffled_distribution", "random_norm_matched"):
        assert first[name]["l2_norm"] == pytest.approx(released_norm)
    assert np.array_equal(first["released_four_way"]["vector"], first["neutral_zero_vector_diagnostic"]["vector"])
    assert first["identity_neutral_control"]["l2_norm"] == 0.0


def test_nonfixed_shuffle_reassigns_labels_even_when_probabilities_match() -> None:
    weights = {"angry": 0.1, "happy": 0.2, "sad": 0.3, "surprise": 0.4}
    shuffled = deterministic_nonfixed_shuffle(weights, seed=1, sample_id="a")
    assert sum(shuffled.values()) == pytest.approx(sum(weights.values()))
    assert set(shuffled) == set(weights)


def _records(*, actor_count: int = 60, reversed_emotion: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(actor_count):
        actor = f"{1001 + index:04d}"
        released = 0.2 if reversed_emotion else 0.8
        shuffled = 0.8 if reversed_emotion else 0.2
        for arm, rho, h_rate, wavlm, wer in (
            ("released_four_way", released, released, 0.80, 0.10),
            ("shuffled_distribution", shuffled, shuffled, 0.80, 0.10),
            ("random_norm_matched", shuffled, shuffled, 0.80, 0.10),
            ("dominant_non_neutral", 0.75, 0.75, 0.80, 0.10),
            ("identity_neutral_control", shuffled, shuffled, 0.80, 0.0),
            ("renormalized_non_neutral_diagnostic", shuffled, shuffled, 0.80, 0.10),
            ("neutral_zero_vector_diagnostic", released, released, 0.80, 0.10),
        ):
            rows.append({"actor_id": actor, "arm": arm, "proportion_spearman_rho": rho, "h_rate": h_rate, "wavlm_speaker_similarity": wavlm, "whisper_wer": wer})
    return rows


def test_decision_rule_supports_and_falsifies_only_from_complete_finite_pairs() -> None:
    supported = decide_claim4_directional_proxy(
        _records(), bootstrap_replicates=100, bootstrap_seed=7, max_wer_increase=0.20,
        min_speaker_similarity_change=-0.20, min_rho_advantage=0.0,
        min_h_rate_advantage=0.0, min_rho_vs_dominant=-0.20,
    )
    assert supported.verdict == "directional_supported"
    safety_failed = decide_claim4_directional_proxy(
        _records(), bootstrap_replicates=100, bootstrap_seed=7, max_wer_increase=0.02,
        min_speaker_similarity_change=-0.20, min_rho_advantage=0.0,
        min_h_rate_advantage=0.0, min_rho_vs_dominant=-0.20,
    )
    assert safety_failed.verdict == "directional_failed"
    failed = decide_claim4_directional_proxy(
        _records(reversed_emotion=True), bootstrap_replicates=100, bootstrap_seed=7,
        max_wer_increase=0.20, min_speaker_similarity_change=-0.20,
        min_rho_advantage=0.0, min_h_rate_advantage=0.0, min_rho_vs_dominant=-0.20,
    )
    assert failed.verdict == "directional_failed"
    with pytest.raises(ContractError, match="incomplete Claim 4 metric matrix"):
        decide_claim4_directional_proxy(
            _records()[:-1], bootstrap_replicates=100, bootstrap_seed=7,
            max_wer_increase=0.2, min_speaker_similarity_change=-0.20,
            min_rho_advantage=0.0, min_h_rate_advantage=0.0, min_rho_vs_dominant=-0.20,
        )


@pytest.mark.parametrize("actor_count", [59, 61])
def test_full_decision_rejects_any_actor_count_other_than_frozen_sixty(actor_count: int) -> None:
    with pytest.raises(ContractError, match="exactly 60 actors and 420 rows"):
        decide_claim4_directional_proxy(
            _records(actor_count=actor_count), bootstrap_replicates=100, bootstrap_seed=7,
            max_wer_increase=0.20, min_speaker_similarity_change=-0.20,
            min_rho_advantage=0.0, min_h_rate_advantage=0.0, min_rho_vs_dominant=-0.20,
        )
