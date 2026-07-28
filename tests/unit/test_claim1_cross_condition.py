import torch
import pytest

from repro.claim1_cross_condition import (
    FLOW_FIELDS,
    LLM_TOKEN_FIELDS,
    changed_tensor_fields,
    merge_cross_conditioned_inputs,
    synthesize_cross_conditioned_with_llm_trace,
)
from repro.contracts import ContractError


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
        "source_speech_token": torch.empty((1, 0), dtype=torch.int32),
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
    class FakeLLM:
        def inference(self):
            yield torch.tensor([1])

    class FakeModel:
        def __init__(self) -> None:
            self.llm = FakeLLM()

        def tts(self, **kwargs):
            assert kwargs["source_speech_token"].numel() == 0
            assert not kwargs["stream"]
            next(self.llm.inference())
            yield {"tts_speech": torch.ones(1, 16)}

    output, calls = synthesize_cross_conditioned_with_llm_trace(FakeModel(), frontend_fixture(0))
    assert output.shape == (1, 16)
    assert calls == 1


def test_llm_trace_rejects_skipped_or_multiple_calls_and_restores_method() -> None:
    class FakeLLM:
        def inference(self):
            yield torch.tensor([1])

    class NoCallModel:
        def __init__(self) -> None:
            self.llm = FakeLLM()

        def tts(self, **kwargs):
            yield {"tts_speech": torch.ones(1, 16)}

    class TwoCallModel(NoCallModel):
        def tts(self, **kwargs):
            next(self.llm.inference())
            next(self.llm.inference())
            yield {"tts_speech": torch.ones(1, 16)}

    for model, observed in ((NoCallModel(), 0), (TwoCallModel(), 2)):
        original = model.llm.inference
        with pytest.raises(ContractError, match=f"observed={observed}"):
            synthesize_cross_conditioned_with_llm_trace(model, frontend_fixture(0))
        assert model.llm.inference.__self__ is model.llm
        assert model.llm.inference.__func__ is original.__func__


def test_merge_requires_explicit_empty_source_speech_token() -> None:
    neutral = frontend_fixture(0)
    emotional = frontend_fixture(1)
    without_source = dict(neutral)
    without_source.pop("source_speech_token")
    with pytest.raises(ContractError, match="source_speech_token"):
        merge_cross_conditioned_inputs(without_source, emotional, condition="slm_driven")

    nonempty = dict(neutral)
    nonempty["source_speech_token"] = torch.tensor([[1]], dtype=torch.int32)
    with pytest.raises(ContractError, match="empty tensor"):
        merge_cross_conditioned_inputs(nonempty, emotional, condition="slm_driven")
