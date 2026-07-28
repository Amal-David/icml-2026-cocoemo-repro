import pytest
import torch

from repro.claim23_analysis import (
    BootstrapInterval,
    actor_cluster_bootstrap,
    claim2_directional_verdict,
    claim2_heldout_controls,
    cosine_alignment,
    paired_delta_alignment_controls,
    vector_sha256,
)
from repro.contracts import ContractError


def test_actor_bootstrap_is_seeded_and_clusters_before_resampling() -> None:
    rows = [
        {"actor": "01", "value": 1.0}, {"actor": "01", "value": 1.0},
        {"actor": "02", "value": 0.0},
    ]
    first = actor_cluster_bootstrap(rows, value_key="value", replicates=100, seed=7)
    second = actor_cluster_bootstrap(rows, value_key="value", replicates=100, seed=7)

    assert first == second
    assert first.estimate == pytest.approx(0.5)


def test_claim2_requires_all_four_targets_to_clear_both_controls() -> None:
    supported = {
        emotion: {"matched": BootstrapInterval(0.8, 0.4, 0.9), "mismatched": BootstrapInterval(0.0, -0.2, 0.2), "shuffled": BootstrapInterval(0.0, -0.2, 0.2), "random_99": 0.3}
        for emotion in ("angry", "happy", "sad", "surprise")
    }
    assert claim2_directional_verdict(supported) == "proxy_supported"
    supported["sad"] = {"matched": BootstrapInterval(-0.5, -0.9, -0.3), "mismatched": BootstrapInterval(0.0, -0.2, 0.2), "shuffled": BootstrapInterval(0.0, -0.2, 0.2), "random_99": 0.3}
    assert claim2_directional_verdict(supported) == "proxy_failed"


def test_cosine_and_hash_fail_closed_for_degenerate_vectors() -> None:
    assert cosine_alignment(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == pytest.approx(1.0)
    assert len(vector_sha256(torch.tensor([1.0, 2.0]))) == 64
    with pytest.raises(ContractError, match="zero"):
        cosine_alignment(torch.zeros(2), torch.ones(2))
    with pytest.raises(ContractError, match="finite"):
        vector_sha256(torch.tensor([float("nan")]))


def test_claim2_alignment_requires_exact_held_out_pairs_and_all_controls() -> None:
    records, pairs = [], []
    for index in range(28):
        actor = f"{18 + index // 4:02d}"
        emotional, neutral = f"e{index}", f"n{index}"
        for clip_id, activation in ((emotional, [2.0, 0.0]), (neutral, [1.0, 0.0])):
            records.append({"clip_id": clip_id, "operation": "attn_output", "layer": 17, "activation": activation})
        pairs.append({"split": "test", "emotion": "angry", "actor": actor, "emotional_clip_id": emotional, "neutral_clip_id": neutral})
    result = paired_delta_alignment_controls(
        records, pairs, target_emotion="angry", site=("attn_output", 17), target_vector=torch.tensor([1.0, 0.0]),
        shuffled_vectors=[torch.tensor([0.0, 1.0]) for _ in range(100)],
        random_vectors=[torch.tensor([0.0, 1.0]) for _ in range(100)],
    )
    assert result["pair_count"] == 28
    assert result["matched"].low == pytest.approx(1.0)
    assert result["random_99"] == pytest.approx(0.0)
    with pytest.raises(ContractError, match="exactly 100"):
        paired_delta_alignment_controls(
            records, pairs, target_emotion="angry", site=("attn_output", 17), target_vector=torch.tensor([1.0, 0.0]),
            shuffled_vectors=[], random_vectors=[],
        )


def test_claim2_control_ledger_records_every_deterministic_mismatch_choice() -> None:
    records, pairs = [], []
    for index in range(28):
        actor = f"{18 + index // 4:02d}"
        emotional, neutral = f"e{index}", f"n{index}"
        for clip_id, activation in ((emotional, [2.0, 0.0]), (neutral, [1.0, 0.0])):
            records.append({"clip_id": clip_id, "operation": "attn_output", "layer": 17, "activation": activation})
        pairs.append({
            "pair_id": f"actor_{actor}_statement_01_repetition_01_emotion_angry_{index}", "split": "test", "emotion": "angry",
            "actor": actor, "statement": "01", "repetition": "01", "emotional_clip_id": emotional, "neutral_clip_id": neutral,
        })
    result = claim2_heldout_controls(
        records, pairs, target_emotion="angry", site=("attn_output", 17), target_vector=torch.tensor([1.0, 0.0]),
        shuffled_vectors=[torch.tensor([0.0, 1.0]) for _ in range(100)], random_vectors=[torch.tensor([0.0, 1.0]) for _ in range(100)],
        shuffled_seed=4, random_seed=5,
    )
    assert len(result["control_ledger"]) == 28
    first = result["control_ledger"][0]
    assert first["matched_neutral_clip_id"] != first["mismatched_neutral_clip_id"]
    # This synthetic fixture gives each target pair a distinct neutral ID; the
    # real RAVDESS ledger deduplicates its shared neutral counterpart per actor.
    assert first["mismatch_candidate_count"] == 24
    assert result["control_recompute"] == {
        "site": "attn_output@17", "split": "test", "shuffled_draw_count": 100, "shuffled_seed": 4,
        "random_draw_count": 100, "random_seed": 5, "random_global_99": 0.0,
    }
