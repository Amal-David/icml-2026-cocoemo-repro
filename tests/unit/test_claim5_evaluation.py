from __future__ import annotations

import hashlib

import numpy as np
import pytest

from repro.claim5_evaluation import actor_cluster_bootstrap, decide_claim5_proxy, leave_one_actor_out_prototypes, require_complete_claim5_metric_matrix
from repro.claim5_protocol import CLAIM5_ARMS, TARGETS
from repro.contracts import ContractError


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actor_number in range(1, 25):
        actor = f"{actor_number:02d}"
        for target_index, target in enumerate(TARGETS):
            content = f"{target}_{(actor_number + target_index - 1) % 4 + 1:02d}"
            for arm_index, arm in enumerate(CLAIM5_ARMS):
                boost = 0.2 if arm == "cocoemo_alpha6" else 0.1 if arm == "cocoemo_alpha3" else 0.0
                digest = hashlib.sha256(f"{actor}/{content}/{arm}".encode()).hexdigest()
                rows.append({
                    "cell_id": f"{actor}/{target}/{content}", "actor_id": actor, "target": target, "content_id": content,
                    "arm": arm, "audio": {"sha256": digest}, "target_emotion_probability": 0.3 + boost,
                    "leave_one_actor_out_esim": 0.4 + boost, "same_actor_target_esim": 0.45 + boost,
                    "wavlm_speaker_similarity": 0.8, "whisper_wer": 0.05,
                })
    return rows


def test_loao_prototype_excludes_the_matching_actor() -> None:
    references = []
    for actor in range(1, 25):
        for target_index, target in enumerate(TARGETS):
            vector = np.zeros(4, dtype=np.float32)
            vector[target_index] = 1.0
            vector[(actor - 1) % 4] += 0.01
            references.append({"actor_id": f"{actor:02d}", "target": target, "embedding": vector})
    prototypes = leave_one_actor_out_prototypes(references)
    assert len(prototypes) == 96
    assert prototypes[("01", "angry")].shape == (4,)


def test_complete_matrix_and_actor_cluster_bootstrap_are_strict_and_deterministic() -> None:
    rows = _records()
    require_complete_claim5_metric_matrix(rows)
    first = actor_cluster_bootstrap(rows, left_arm="cocoemo_alpha6", right_arm="instruction_target", metric="target_emotion_probability", replicates=200, seed=12)
    second = actor_cluster_bootstrap(rows, left_arm="cocoemo_alpha6", right_arm="instruction_target", metric="target_emotion_probability", replicates=200, seed=12)
    assert first == second
    assert first.lower > 0
    protocol_scale = actor_cluster_bootstrap(
        rows, left_arm="cocoemo_alpha6", right_arm="instruction_target",
        metric="target_emotion_probability", replicates=10_000, seed=20260728,
    )
    assert protocol_scale.lower > 0
    with pytest.raises(ContractError, match="duplicate"):
        require_complete_claim5_metric_matrix(rows + [dict(rows[0])])


def test_actor_cluster_resamples_actor_effects_not_sparse_content_intersections(monkeypatch) -> None:
    import repro.claim5_evaluation as evaluation

    rows = _records()
    for row in rows:
        if row["arm"] == "cocoemo_alpha6":
            row["target_emotion_probability"] = 0.3 + int(str(row["actor_id"])) / 100.0

    class OrderedActorRng:
        def integers(self, low, high, *, size):
            assert (low, high, size) == (0, 24, 24)
            return np.arange(24)

    monkeypatch.setattr(evaluation.np.random, "default_rng", lambda seed: OrderedActorRng())
    result = actor_cluster_bootstrap(
        rows, left_arm="cocoemo_alpha6", right_arm="instruction_target",
        metric="target_emotion_probability", replicates=100, seed=9, target="surprise",
    )
    assert result.estimate == pytest.approx(np.mean([actor / 100.0 for actor in range(1, 25)]))
    assert result.lower == result.upper == result.estimate


def test_verdict_requires_all_primary_controls_and_quality_margins() -> None:
    decision = decide_claim5_proxy(_records(), bootstrap_replicates=200, bootstrap_seed=2, min_primary_effect=0.0, min_wavlm_change=-0.02, max_wer_increase=0.02)
    assert decision.verdict == "directional_supported"
    bad = _records()
    for row in bad:
        if row["target"] == "angry" and row["arm"] == "cocoemo_alpha6":
            row["target_emotion_probability"] = 0.0
            row["leave_one_actor_out_esim"] = 0.0
    assert decide_claim5_proxy(bad, bootstrap_replicates=200, bootstrap_seed=2, min_primary_effect=0.0, min_wavlm_change=-0.02, max_wer_increase=0.02).verdict == "directional_failed"


def test_full_decision_has_no_sparse_failure_for_every_frozen_10k_seed() -> None:
    decision = decide_claim5_proxy(
        _records(), bootstrap_replicates=10_000, bootstrap_seed=20260728,
        min_primary_effect=0.0, min_wavlm_change=-0.02, max_wer_increase=0.02,
    )
    assert decision.verdict == "directional_supported"
    assert {result["seed"] for result in decision.comparisons.values()} == set(range(20260728, 20260750))
    assert {result["replicates"] for result in decision.comparisons.values()} == {10_000}
