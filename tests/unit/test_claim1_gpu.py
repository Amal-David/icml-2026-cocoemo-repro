from pathlib import Path

import pytest

from repro.claim1_gpu import (
    CAUSAL_CONDITIONS,
    Claim1Settings,
    allowed_changed_fields,
    parse_claim1_settings,
    require_distinct_condition_wavs,
    select_claim1_groups,
    validate_causal_field_deltas,
)
from repro.claim1_cross_condition import merge_cross_conditioned_inputs
from repro.config import load_config
from repro.contracts import ContractError


REPO = Path(__file__).resolve().parents[2]


def settings(*, expected_total_groups: int = 1, selected: tuple[str, ...] = ("g1",)) -> Claim1Settings:
    return Claim1Settings(
        source_url="https://example.invalid/cosyvoice.git",
        source_revision="a" * 40,
        model_id="example/model",
        model_revision="b" * 40,
        ravdess_source_url="https://example.invalid/ravdess",
        ravdess_release="v1",
        ravdess_root=".cache/ravdess",
        expected_total_groups=expected_total_groups,
        max_groups=len(selected),
        selected_group_ids=selected,
        target_emotions=("angry",),
        conditions=CAUSAL_CONDITIONS,
        llm_embedding_mode="neutral",
    )


def test_committed_claim1_smoke_config_is_strict_and_valid() -> None:
    config = load_config(REPO / "configs" / "claim1-ravdess-gpu-smoke.yaml")
    parsed = parse_claim1_settings(config)

    assert config.stage == "claim1"
    assert parsed.expected_total_groups == 96
    assert parsed.max_groups == 1
    assert parsed.conditions == CAUSAL_CONDITIONS


def test_selection_rejects_missing_or_wrong_sized_public_scope() -> None:
    groups = [{"group_id": "g1"}]

    with pytest.raises(ContractError, match="expected=2, actual=1"):
        select_claim1_groups(groups, settings(expected_total_groups=2))
    with pytest.raises(ContractError, match="configured RAVDESS groups are missing: g2"):
        select_claim1_groups(groups, settings(selected=("g2",)))


def _frontend_fixture(offset: int, *, same_lengths: bool = False) -> dict:
    import torch

    length_offset = 0 if same_lengths else offset
    return {
        "text": torch.tensor([[1, 2]]),
        "text_len": torch.tensor([2]),
        "prompt_text": torch.tensor([[3, 4]]),
        "prompt_text_len": torch.tensor([2]),
        "llm_prompt_speech_token": torch.tensor([[10 + offset, 11 + offset]]),
        "llm_prompt_speech_token_len": torch.tensor([2 + length_offset]),
        "flow_prompt_speech_token": torch.tensor([[20 + offset, 21 + offset]]),
        "flow_prompt_speech_token_len": torch.tensor([2 + length_offset]),
        "prompt_speech_feat": torch.full((1, 2, 3), 30.0 + offset),
        "prompt_speech_feat_len": torch.tensor([2 + length_offset]),
        "llm_embedding": torch.full((1, 4), 40.0 + offset),
        "flow_embedding": torch.full((1, 4), 50.0 + offset),
    }


def test_field_delta_validation_allows_equal_length_tensors() -> None:
    neutral = _frontend_fixture(0)
    emotional = _frontend_fixture(1, same_lengths=True)
    slm = merge_cross_conditioned_inputs(neutral, emotional, condition="slm_driven")
    flow = merge_cross_conditioned_inputs(neutral, emotional, condition="flow_driven")

    assert validate_causal_field_deltas(
        neutral, slm, condition="slm_driven", llm_embedding_mode="neutral"
    ) == {"llm_prompt_speech_token"}
    assert validate_causal_field_deltas(
        neutral, flow, condition="flow_driven", llm_embedding_mode="neutral"
    ) == {"flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"}


def test_field_delta_validation_rejects_missing_signal_or_foreign_change() -> None:
    neutral = _frontend_fixture(0)
    unchanged = dict(neutral)
    with pytest.raises(ContractError, match="required slm_driven signal"):
        validate_causal_field_deltas(
            neutral, unchanged, condition="slm_driven", llm_embedding_mode="neutral"
        )

    flow = merge_cross_conditioned_inputs(neutral, _frontend_fixture(1), condition="flow_driven")
    flow["llm_prompt_speech_token"] = _frontend_fixture(1)["llm_prompt_speech_token"]
    with pytest.raises(ContractError, match="outside flow_driven ownership"):
        validate_causal_field_deltas(
            neutral, flow, condition="flow_driven", llm_embedding_mode="neutral"
        )


def test_allowed_field_deltas_are_condition_specific() -> None:
    assert allowed_changed_fields("slm_driven", llm_embedding_mode="neutral") == {
        "llm_prompt_speech_token",
        "llm_prompt_speech_token_len",
    }
    assert "llm_embedding" not in allowed_changed_fields(
        "emotional_both", llm_embedding_mode="neutral"
    )
    assert "llm_embedding" in allowed_changed_fields(
        "emotional_both", llm_embedding_mode="native_bundle"
    )


def _wav_records(hashes: list[str]) -> list[dict]:
    return [
        {
            "group_id": "g1",
            "target_emotion": "angry",
            "condition": condition,
            "audio": {"sha256": sha256},
        }
        for condition, sha256 in zip(CAUSAL_CONDITIONS, hashes)
    ]


def test_condition_wav_accounting_requires_distinct_complete_outputs() -> None:
    require_distinct_condition_wavs(
        _wav_records(["a" * 64, "b" * 64, "c" * 64]), conditions=CAUSAL_CONDITIONS
    )
    with pytest.raises(ContractError, match="non-distinct WAVs"):
        require_distinct_condition_wavs(
            _wav_records(["a" * 64, "a" * 64, "c" * 64]), conditions=CAUSAL_CONDITIONS
        )
    with pytest.raises(ContractError, match="accounting mismatch"):
        require_distinct_condition_wavs(
            _wav_records(["a" * 64, "b" * 64]), conditions=CAUSAL_CONDITIONS
        )
