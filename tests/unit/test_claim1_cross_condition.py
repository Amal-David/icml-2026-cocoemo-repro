import torch

from repro.claim1_cross_condition import (
    FLOW_FIELDS,
    LLM_TOKEN_FIELDS,
    changed_tensor_fields,
    merge_cross_conditioned_inputs,
    synthesize_cross_conditioned,
)


def frontend_fixture(offset: int) -> dict[str, torch.Tensor]:
    common = {
        "text": torch.tensor([[1, 2]]),
        "text_len": torch.tensor([2]),
        "prompt_text": torch.tensor([[3, 4]]),
        "prompt_text_len": torch.tensor([2]),
    }
    varying = {
        "llm_prompt_speech_token": torch.tensor([[10 + offset, 11 + offset]]),
        "llm_prompt_speech_token_len": torch.tensor([2 + offset]),
        "flow_prompt_speech_token": torch.tensor([[20 + offset, 21 + offset]]),
        "flow_prompt_speech_token_len": torch.tensor([2 + offset]),
        "prompt_speech_feat": torch.full((1, 2, 3), 30.0 + offset),
        "prompt_speech_feat_len": torch.tensor([2 + offset]),
        "llm_embedding": torch.full((1, 4), 40.0 + offset),
        "flow_embedding": torch.full((1, 4), 50.0 + offset),
    }
    return {**common, **varying}


def test_slm_driven_changes_only_llm_token_fields_in_strict_mode() -> None:
    neutral = frontend_fixture(0)
    emotional = frontend_fixture(1)
    merged = merge_cross_conditioned_inputs(neutral, emotional, condition="slm_driven")

    assert changed_tensor_fields(neutral, merged) == set(LLM_TOKEN_FIELDS)


def test_flow_driven_changes_only_flow_fields() -> None:
    neutral = frontend_fixture(0)
    emotional = frontend_fixture(1)
    merged = merge_cross_conditioned_inputs(neutral, emotional, condition="flow_driven")

    assert changed_tensor_fields(neutral, merged) == set(FLOW_FIELDS)


def test_native_bundle_sensitivity_also_changes_llm_embedding() -> None:
    neutral = frontend_fixture(0)
    emotional = frontend_fixture(1)
    merged = merge_cross_conditioned_inputs(
        neutral,
        emotional,
        condition="slm_driven",
        llm_embedding_mode="native_bundle",
    )

    assert changed_tensor_fields(neutral, merged) == set(LLM_TOKEN_FIELDS).union({"llm_embedding"})


def test_synthesis_uses_language_model_path() -> None:
    class FakeModel:
        def tts(self, **kwargs):
            assert "source_speech_token" not in kwargs
            assert not kwargs["stream"]
            yield {"tts_speech": torch.ones(1, 16)}

    output = synthesize_cross_conditioned(FakeModel(), frontend_fixture(0))
    assert output.shape == (1, 16)
