"""Fail-closed, unistream terminal-token activation extraction for Claims 2/3."""

from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Any, Iterable

import torch

from repro.contracts import ContractError


COSYVOICE_HOOK_TYPES = (
    "emb_pre_attn_post_ln",
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_output",
    "W0_x_attn_output",
    "emb_post_attn_pre_ln",
    "emb_post_attn_post_ln",
    "emb_post_mlp_residual",
    "layer_output",
)
LAYERS = tuple(range(24))


def true_terminal_speech_token_index(*, text_token_len: int, speech_token_len: int) -> int:
    """Index the final speech token in ``[SOS, text, task-id, speech]``."""

    if text_token_len < 1 or speech_token_len < 1:
        raise ContractError("unistream terminal position requires positive text and speech token lengths")
    return 2 + text_token_len + speech_token_len - 1


def _only_batch_one(value: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1:
        raise ContractError(f"Claim 2/3 frontend must provide a batch-one rank-two {label} tensor")
    return value


def build_unistream_teacher_forced_input(model: Any, frontend_input: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the Qwen2LM unistream branch without its stochastic bistream choice."""

    try:
        text = _only_batch_one(frontend_input["text"], "text").to(device)
        speech = _only_batch_one(frontend_input["llm_prompt_speech_token"], "llm_prompt_speech_token").to(device)
        text_len = int(_only_batch_one(frontend_input["text_len"].reshape(1, 1), "text_len")[0, 0].item())
        speech_len = int(_only_batch_one(frontend_input["llm_prompt_speech_token_len"].reshape(1, 1), "llm_prompt_speech_token_len")[0, 0].item())
        qwen = model.llm
        text_embeddings = qwen.llm.model.model.embed_tokens(text)[:, :text_len, :]
        speech_embeddings = qwen.speech_embedding(speech)[:, :speech_len, :]
        sos = qwen.llm_embedding.weight[qwen.sos].reshape(1, 1, -1)
        task_id = qwen.llm_embedding.weight[qwen.task_id].reshape(1, 1, -1)
    except (AttributeError, KeyError, IndexError, RuntimeError) as exc:
        raise ContractError("pinned CosyVoice2 unistream teacher-forced API contract is unavailable") from exc
    sequence = torch.cat((sos, text_embeddings, task_id, speech_embeddings), dim=1)
    terminal = true_terminal_speech_token_index(text_token_len=text_len, speech_token_len=speech_len)
    if terminal != sequence.shape[1] - 1:
        raise ContractError("unistream terminal-token accounting does not match constructed sequence")
    return sequence, torch.tensor([sequence.shape[1]], dtype=torch.int32, device=device), terminal


def _capture_tensor(value: object, *, hook_type: str) -> torch.Tensor:
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
        raise ContractError(f"Claim 2/3 hook {hook_type} did not expose a batch-one [B,T,D] tensor")
    return value.detach()


def register_full_capture_hooks(model: Any, *, operations: Iterable[str] = COSYVOICE_HOOK_TYPES, layers: Iterable[int] = LAYERS) -> tuple[dict[tuple[str, int], torch.Tensor | None], list[Any]]:
    """Register hooks retaining complete sequences until terminal indexing is audited."""

    try:
        from cocoemo.steering._core_cosyvoice import get_cosyvoice_op_dict
    except ImportError as exc:
        raise ContractError("CoCoEmo CosyVoice hook map is unavailable") from exc
    operations = tuple(operations)
    layers = tuple(layers)
    if operations != COSYVOICE_HOOK_TYPES or layers != LAYERS:
        raise ContractError("Claim 2/3 must capture all ten frozen operations across all 24 layers")
    op_map = get_cosyvoice_op_dict(list(operations))
    if tuple(op_map) != operations:
        raise ContractError("pinned CosyVoice operation map differs from the ten-site protocol")
    captures: dict[tuple[str, int], torch.Tensor | None] = {}
    handles: list[Any] = []
    for layer in layers:
        for operation in operations:
            spec = op_map[operation]
            module_name = spec["module"].format(layer=str(layer))
            try:
                module = reduce(getattr, module_name.split("."), model)
            except AttributeError as exc:
                raise ContractError(f"Claim 2/3 hook module is unavailable: {module_name}") from exc
            key = (operation, layer)
            captures[key] = None
            if spec["hook type"] == "forward_pre":
                def pre_hook(_module: Any, inputs: tuple[Any, ...], *, key: tuple[str, int] = key) -> None:
                    captures[key] = _capture_tensor(inputs[0], hook_type=f"{key[0]}@{key[1]}")

                handles.append(module.register_forward_pre_hook(pre_hook))
            elif spec["hook type"] == "forward":
                def post_hook(_module: Any, _inputs: tuple[Any, ...], output: object, *, key: tuple[str, int] = key) -> None:
                    captures[key] = _capture_tensor(output, hook_type=f"{key[0]}@{key[1]}")

                handles.append(module.register_forward_hook(post_hook))
            else:
                raise ContractError(f"unsupported Claim 2/3 hook type: {spec['hook type']}")
    return captures, handles


def extract_unistream_activations(model: Any, frontend: Any, *, audio_path: str | Path, transcript: str) -> tuple[dict[tuple[str, int], torch.Tensor], dict[str, int]]:
    """Extract one complete all-site activation record or fail the whole run."""

    if not transcript:
        raise ContractError("Claim 2/3 extraction requires a non-empty frozen transcript")
    try:
        from cosyvoice.utils.file_utils import load_wav
    except ImportError as exc:
        raise ContractError("pinned CosyVoice audio loader is unavailable") from exc
    reference = Path(audio_path)
    if not reference.is_file():
        raise ContractError(f"Claim 2/3 source audio is missing: {reference}")
    try:
        audio = load_wav(str(reference), 16000)
        frontend_input = frontend.frontend_zero_shot(
            tts_text=transcript,
            prompt_text=transcript,
            prompt_speech_16k=audio,
            resample_rate=24000,
            zero_shot_spk_id="",
        )
    except Exception as exc:  # the record is invalid; it must never be skipped
        raise ContractError(f"Claim 2/3 frontend failed for {reference.name}") from exc
    if not isinstance(frontend_input, dict):
        raise ContractError("Claim 2/3 frontend did not return a mapping")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ContractError("Claim 2/3 model has no discoverable parameter device") from exc
    sequence, lengths, terminal = build_unistream_teacher_forced_input(model, frontend_input, device)
    captures, handles = register_full_capture_hooks(model)
    try:
        with torch.inference_mode():
            model.llm.llm(sequence, lengths)
    except Exception as exc:
        raise ContractError(f"Claim 2/3 unistream forward failed for {reference.name}") from exc
    finally:
        for handle in handles:
            handle.remove()
    expected = len(COSYVOICE_HOOK_TYPES) * len(LAYERS)
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    for key, captured in captures.items():
        if captured is None:
            raise ContractError(f"Claim 2/3 missing hook activation: {key[0]}@{key[1]}")
        if terminal >= captured.shape[1]:
            raise ContractError(f"Claim 2/3 terminal position exceeds hook sequence: {key[0]}@{key[1]}")
        vector = captured[0, terminal, :].cpu().contiguous()
        if not torch.isfinite(vector).all():
            raise ContractError(f"Claim 2/3 non-finite activation: {key[0]}@{key[1]}")
        vectors[key] = vector
    if len(vectors) != expected:
        raise ContractError(f"Claim 2/3 activation accounting mismatch: expected={expected}, actual={len(vectors)}")
    return vectors, {"terminal_index": terminal, "sequence_length": int(sequence.shape[1])}
