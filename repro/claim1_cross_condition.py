from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import torch

from repro.contracts import ContractError


COMMON_FIELDS = ("text", "text_len", "prompt_text", "prompt_text_len")
LLM_TOKEN_FIELDS = ("llm_prompt_speech_token", "llm_prompt_speech_token_len")
FLOW_FIELDS = (
    "flow_prompt_speech_token",
    "flow_prompt_speech_token_len",
    "prompt_speech_feat",
    "prompt_speech_feat_len",
    "flow_embedding",
)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and torch.equal(left, right)
    return left == right


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    payload = tensor.numpy().tobytes()
    metadata = f"{tensor.dtype}:{tuple(tensor.shape)}".encode("ascii")
    return hashlib.sha256(metadata + b"\0" + payload).hexdigest()


def field_digests(model_input: dict[str, Any]) -> dict[str, str]:
    return {
        key: tensor_digest(value)
        for key, value in sorted(model_input.items())
        if isinstance(value, torch.Tensor)
    }


def merge_cross_conditioned_inputs(
    neutral: dict[str, Any],
    emotional: dict[str, Any],
    *,
    condition: str,
    llm_embedding_mode: str = "neutral",
) -> dict[str, Any]:
    if condition not in {"neutral_both", "slm_driven", "flow_driven", "emotional_both"}:
        raise ContractError(f"unsupported cross-conditioning condition: {condition}")
    if llm_embedding_mode not in {"neutral", "native_bundle"}:
        raise ContractError(f"unsupported LLM embedding mode: {llm_embedding_mode}")

    required = set(COMMON_FIELDS + LLM_TOKEN_FIELDS + FLOW_FIELDS + ("llm_embedding",))
    for label, values in (("neutral", neutral), ("emotional", emotional)):
        missing = sorted(required.difference(values))
        if missing:
            raise ContractError(f"{label} frontend output is missing: {', '.join(missing)}")
    for field in COMMON_FIELDS:
        if not _same(neutral[field], emotional[field]):
            raise ContractError(f"common frontend field differs across references: {field}")

    merged = dict(neutral)
    if condition in {"slm_driven", "emotional_both"}:
        for field in LLM_TOKEN_FIELDS:
            merged[field] = emotional[field]
        if llm_embedding_mode == "native_bundle":
            merged["llm_embedding"] = emotional["llm_embedding"]
    if condition in {"flow_driven", "emotional_both"}:
        for field in FLOW_FIELDS:
            merged[field] = emotional[field]

    source = merged.get("source_speech_token")
    if source is not None and (not isinstance(source, torch.Tensor) or source.shape[-1] != 0):
        raise ContractError("source_speech_token must remain empty so the SLM path executes")
    return merged


def changed_tensor_fields(reference: dict[str, Any], candidate: dict[str, Any]) -> set[str]:
    keys = set(reference).union(candidate)
    return {
        key
        for key in keys
        if isinstance(reference.get(key), torch.Tensor)
        and isinstance(candidate.get(key), torch.Tensor)
        and not _same(reference[key], candidate[key])
    }


def synthesize_cross_conditioned(model: Any, model_input: dict[str, Any]) -> torch.Tensor:
    outputs: Iterator[dict[str, torch.Tensor]] = iter(model.tts(**model_input, stream=False))
    try:
        first = next(outputs)
    except StopIteration as exc:
        raise ContractError("cross-conditioned synthesis produced no output") from exc
    if "tts_speech" not in first or first["tts_speech"].numel() == 0:
        raise ContractError("cross-conditioned synthesis produced empty audio")
    try:
        next(outputs)
    except StopIteration:
        return first["tts_speech"]
    raise ContractError("non-streaming cross-conditioned synthesis produced multiple outputs")
