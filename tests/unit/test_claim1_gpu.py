from copy import deepcopy
from pathlib import Path

import pytest
import torch

from repro.claim1_gpu import (
    CAUSAL_CONDITIONS,
    SCALED_CONDITIONS,
    Claim1Settings,
    allowed_changed_fields,
    derive_claim1_draw_seed,
    deterministic_permuted_emotions,
    parse_claim1_settings,
    require_distinct_condition_wavs,
    select_claim1_groups,
    validate_causal_field_deltas,
    _write_wav_and_record,
)
from repro.claim1_cross_condition import merge_cross_conditioned_inputs
from repro.config import ReproConfig, load_config
from repro.contracts import ContractError
from repro.ravdess_acquisition import RavdessAcquisitionSettings


REPO = Path(__file__).resolve().parents[2]


def settings(*, expected_total_groups: int = 1, selected: tuple[str, ...] = ("g1",)) -> Claim1Settings:
    return Claim1Settings(
        source_url="https://example.invalid/cosyvoice.git",
        source_revision="a" * 40,
        model_id="example/model",
        model_revision="b" * 40,
        ravdess=RavdessAcquisitionSettings(
            source_url="https://example.invalid/ravdess",
            release="v1",
            root=".cache/ravdess",
            acquire=False,
            license="test",
            license_acknowledged=True,
            archive_url="https://example.invalid/ravdess.zip",
            archive_filename="ravdess.zip",
            archive_size_bytes=1,
            archive_md5="0" * 32,
            expected_archive_wavs=1,
            expected_total_groups=expected_total_groups,
        ),
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
    assert parsed.ravdess.expected_total_groups == 96
    assert parsed.max_groups == 1
    assert parsed.conditions == CAUSAL_CONDITIONS
    assert parsed.claim_evidence is False
    assert parsed.expected_rendered_wavs == 3


def test_committed_scaled_config_locks_controls_denominators_and_metrics() -> None:
    config = load_config(REPO / "configs" / "claim1-ravdess-scaled-directional.yaml")
    parsed = parse_claim1_settings(config)

    assert parsed.protocol == "scaled_directional"
    assert parsed.claim_evidence is True
    assert parsed.conditions == SCALED_CONDITIONS
    assert parsed.max_groups == 20
    assert parsed.expected_group_emotion_pairs == 80
    assert parsed.expected_rendered_wavs == 400
    assert parsed.acoustic_metrics is not None
    assert parsed.max_clipping_fraction == pytest.approx(0.001)


def test_scaled_config_rejects_duplicate_actor_even_with_distinct_groups() -> None:
    config = load_config(REPO / "configs" / "claim1-ravdess-scaled-directional.yaml")
    raw = deepcopy(config.raw)
    raw["claim1"]["selected_group_ids"][1] = "actor_01_statement_02_repetition_01"
    duplicate_actor_config = ReproConfig(source=config.source, raw=raw, digest=config.digest)

    with pytest.raises(ContractError, match="20 distinct RAVDESS actors"):
        parse_claim1_settings(duplicate_actor_config)


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
        "source_speech_token": torch.empty((1, 0), dtype=torch.int32),
    }


def test_field_delta_validation_allows_equal_length_tensors() -> None:
    neutral = _frontend_fixture(0)
    emotional = _frontend_fixture(1, same_lengths=True)
    all_neutral = merge_cross_conditioned_inputs(neutral, neutral, condition="neutral_both")
    slm = merge_cross_conditioned_inputs(neutral, emotional, condition="slm_driven")
    flow = merge_cross_conditioned_inputs(neutral, emotional, condition="flow_driven")

    assert validate_causal_field_deltas(
        neutral, all_neutral, condition="neutral_both", llm_embedding_mode="neutral"
    ) == set()
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
    assert allowed_changed_fields("neutral_both", llm_embedding_mode="neutral") == set()
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


def test_condition_wav_accounting_records_equal_hashes_without_failing() -> None:
    audit = require_distinct_condition_wavs(
        _wav_records(["a" * 64, "b" * 64, "c" * 64]), conditions=CAUSAL_CONDITIONS
    )
    assert audit["equal_hash_collisions"] == []
    collision = require_distinct_condition_wavs(
        _wav_records(["a" * 64, "a" * 64, "c" * 64]), conditions=CAUSAL_CONDITIONS
    )
    assert collision["equal_hash_collisions"] == [
        {
            "group_id": "g1",
            "target_emotion": "angry",
            "sha256": "a" * 64,
            "conditions": ["flow_driven", "slm_driven"],
        }
    ]
    with pytest.raises(ContractError, match="accounting mismatch"):
        require_distinct_condition_wavs(
            _wav_records(["a" * 64, "b" * 64]), conditions=CAUSAL_CONDITIONS
        )


def test_permuted_emotion_control_is_deterministic_and_has_no_fixed_points() -> None:
    mapping = deterministic_permuted_emotions(("angry", "happy", "sad", "surprise"))

    assert mapping == {
        "angry": "happy",
        "happy": "sad",
        "sad": "surprise",
        "surprise": "angry",
    }
    assert all(source != destination for source, destination in mapping.items())


def test_draw_seed_is_condition_order_independent_and_shared_by_matched_arms() -> None:
    conditions = SCALED_CONDITIONS
    forward = {
        condition: derive_claim1_draw_seed(
            base_seed=20260728,
            group_id="actor_01_statement_01_repetition_01",
            target_emotion="angry",
        )
        for condition in conditions
    }
    reverse = {
        condition: derive_claim1_draw_seed(
            base_seed=20260728,
            group_id="actor_01_statement_01_repetition_01",
            target_emotion="angry",
        )
        for condition in reversed(conditions)
    }

    assert forward == reverse
    assert len(set(forward.values())) == 1
    assert forward["slm_driven"] != derive_claim1_draw_seed(
        base_seed=20260728,
        group_id="actor_01_statement_01_repetition_01",
        target_emotion="happy",
    )


def test_audio_writer_rejects_nonfinite_and_excessively_clipped_waveforms(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="non-finite"):
        _write_wav_and_record(
            output_path=tmp_path / "nan.wav",
            audio=torch.tensor([[float("nan")]]),
            sample_rate=16_000,
            max_clipping_fraction=0.001,
        )
    with pytest.raises(ContractError, match="clipping fraction exceeds"):
        _write_wav_and_record(
            output_path=tmp_path / "clipped.wav",
            audio=torch.ones((1, 2_000)),
            sample_rate=16_000,
            max_clipping_fraction=0.001,
        )
