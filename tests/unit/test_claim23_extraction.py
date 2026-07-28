from types import SimpleNamespace

import pytest
import torch

from repro.claim23_extraction import (
    COSYVOICE_HOOK_TYPES,
    build_unistream_teacher_forced_input,
    register_full_capture_hooks,
    true_terminal_speech_token_index,
)
from repro.contracts import ContractError


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Identity()
        self.self_attn.k_proj = torch.nn.Identity()
        self.self_attn.v_proj = torch.nn.Identity()
        self.self_attn.o_proj = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = torch.nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_layernorm(value)
        value = self.self_attn.q_proj(value)
        value = self.self_attn.k_proj(value)
        value = self.self_attn.v_proj(value)
        value = self.self_attn.o_proj(value)
        value = self.post_attention_layernorm(value)
        return self.mlp(value)


class _Runner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        nested = torch.nn.Module()
        nested.embed_tokens = torch.nn.Embedding(64, 4)
        nested.layers = torch.nn.ModuleList([_Layer() for _ in range(24)])
        self.model = torch.nn.Module()
        self.model.model = nested

    def forward(self, sequence: torch.Tensor, _lengths: torch.Tensor) -> torch.Tensor:
        for layer in self.model.model.layers:
            sequence = layer(sequence)
        return sequence


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        qwen = torch.nn.Module()
        qwen.llm = _Runner()
        qwen.speech_embedding = torch.nn.Embedding(64, 4)
        qwen.llm_embedding = torch.nn.Embedding(2, 4)
        qwen.sos = 0
        qwen.task_id = 1
        self.llm = qwen


def test_unistream_terminal_index_uses_true_speech_end_not_padding_end() -> None:
    model = _Model()
    frontend = {
        "text": torch.tensor([[1, 2, 3]]), "text_len": torch.tensor([2]),
        "llm_prompt_speech_token": torch.tensor([[4, 5, 6, 7]]),
        "llm_prompt_speech_token_len": torch.tensor([3]),
    }
    sequence, lengths, terminal = build_unistream_teacher_forced_input(model, frontend, torch.device("cpu"))

    assert sequence.shape == (1, 7, 4)
    assert lengths.tolist() == [7]
    assert terminal == 6
    assert terminal == true_terminal_speech_token_index(text_token_len=2, speech_token_len=3)


def test_full_capture_retains_all_sites_then_indexes_terminal_position() -> None:
    model = _Model()
    captures, handles = register_full_capture_hooks(model)
    sequence = torch.ones((1, 7, 4))
    model.llm.llm(sequence, torch.tensor([7], dtype=torch.int32))
    for handle in handles:
        handle.remove()

    assert len(captures) == len(COSYVOICE_HOOK_TYPES) * 24
    assert all(value is not None and value.shape == (1, 7, 4) for value in captures.values())


def test_terminal_index_rejects_zero_length_sequences() -> None:
    with pytest.raises(ContractError, match="positive"):
        true_terminal_speech_token_index(text_token_len=0, speech_token_len=1)
