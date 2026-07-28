"""Native and steered synthesis adapters for the frozen Claim 5 arms."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from repro.claim1_cross_condition import merge_cross_conditioned_inputs, synthesize_cross_conditioned_with_llm_trace
from repro.claim1_gpu import _prepare_frontend_input, _write_wav_and_record
from repro.contracts import ContractError


def reset_claim5_generation_seed(seed: int) -> None:
    """Reset every stochastic backend before each matched arm render."""

    try:
        import torch
    except ImportError as exc:
        raise ContractError("Claim 5 generation requires torch") from exc
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _single_audio(outputs: Iterator[Mapping[str, Any]], *, label: str) -> Any:
    try:
        first = next(outputs)
    except StopIteration as exc:
        raise ContractError(f"Claim 5 {label} produced no audio") from exc
    audio = first.get("tts_speech") if isinstance(first, Mapping) else None
    if audio is None or getattr(audio, "numel", lambda: 0)() == 0:
        raise ContractError(f"Claim 5 {label} produced empty audio")
    try:
        next(outputs)
    except StopIteration:
        return audio
    raise ContractError(f"Claim 5 non-streaming {label} unexpectedly produced multiple outputs")


def native_no_steer(model: Any, *, text: str, reference_audio: Path, prompt_text: str) -> Any:
    """Call CosyVoice's native zero-shot API directly, with no steering hooks."""

    try:
        from cosyvoice.utils.file_utils import load_wav
    except ImportError as exc:
        raise ContractError("Claim 5 native zero-shot generation requires CosyVoice") from exc
    method = getattr(model, "inference_zero_shot", None)
    if not callable(method):
        raise ContractError("Claim 5 model has no native inference_zero_shot API")
    prompt_speech_16k = load_wav(str(reference_audio), 16000)
    return _single_audio(iter(method(
        tts_text=text, prompt_text=prompt_text, prompt_speech_16k=prompt_speech_16k,
        stream=False, speed=1.0,
    )), label="native no-steer")


def native_instruction(model: Any, *, text: str, reference_audio: Path, instruction: str) -> Any:
    """Call ``inference_instruct2`` directly with the frozen paper-style prompt."""

    try:
        from cosyvoice.utils.file_utils import load_wav
    except ImportError as exc:
        raise ContractError("Claim 5 native instruction generation requires CosyVoice") from exc
    if instruction not in {
        "Say it in an angry tone.", "Say it in a happy tone.", "Say it in a sad tone.", "Say it in a surprised tone.",
    }:
        raise ContractError("Claim 5 native instruction prompt drifted from the frozen illustrative form")
    method = getattr(model, "inference_instruct2", None)
    if not callable(method):
        raise ContractError("Claim 5 model has no native inference_instruct2 API")
    prompt_speech_16k = load_wav(str(reference_audio), 16000)
    return _single_audio(iter(method(
        tts_text=text, instruct_text=instruction, prompt_speech_16k=prompt_speech_16k,
        zero_shot_spk_id="", stream=False, speed=1.0, text_frontend=True,
    )), label="native instruction")


def flow_side_emotional_reference(
    model: Any, *, text: str, neutral_reference: Path, emotional_reference: Path, prompt_text: str,
) -> tuple[Any, int]:
    """Run Claim 1's frontend merge as a named flow-side cross-conditioning control."""

    neutral = _prepare_frontend_input(model, reference_audio=neutral_reference, prompt_text=prompt_text, tts_text=text)
    emotional = _prepare_frontend_input(model, reference_audio=emotional_reference, prompt_text=prompt_text, tts_text=text)
    merged = merge_cross_conditioned_inputs(neutral, emotional, condition="flow_driven")
    return synthesize_cross_conditioned_with_llm_trace(model.model, merged)


def render_claim5_arm(
    *, cosyvoice: Any, backbone: Any, arm_name: str, arm: Mapping[str, Any], text: str,
    neutral_reference: Path, emotional_reference: Path, prompt_text: str, output_path: Path,
    sample_rate: int, max_clipping_fraction: float,
) -> dict[str, Any]:
    """Execute exactly one arm and record audio integrity; never silently skip."""

    kind = arm.get("kind")
    trace_calls: int | None = None
    if kind == "native_no_steer":
        audio = native_no_steer(cosyvoice, text=text, reference_audio=neutral_reference, prompt_text=prompt_text)
    elif kind == "native_instruction":
        instruction = arm.get("instruction")
        if not isinstance(instruction, str):
            raise ContractError("Claim 5 instruction arm lacks its frozen prompt")
        audio = native_instruction(cosyvoice, text=text, reference_audio=neutral_reference, instruction=instruction)
    elif kind == "steered":
        try:
            import torch
        except ImportError as exc:
            raise ContractError("Claim 5 steering generation requires torch") from exc
        vectors = arm.get("vectors")
        alpha = arm.get("alpha")
        if not isinstance(vectors, Mapping) or not isinstance(alpha, (int, float)):
            raise ContractError("Claim 5 steered arm lacks vectors or alpha")
        result = backbone.generate_steered_speech(
            model=cosyvoice, text=text, reference_audio_path=str(neutral_reference), prompt_text=prompt_text,
            steering_vectors={"attn_output": {int(layer): torch.from_numpy(np.asarray(vector)).to(torch.float32) for layer, vector in vectors.items()}},
            layers=[17, 14], operations=["attn_output"], alpha=float(alpha), output_path=None,
        )
        if not isinstance(result, Mapping) or result.get("audio") is None:
            raise ContractError(f"Claim 5 steered generation returned no audio for {arm_name}")
        audio = result["audio"]
    elif kind == "flow_side_emotional_reference":
        audio, trace_calls = flow_side_emotional_reference(
            cosyvoice, text=text, neutral_reference=neutral_reference, emotional_reference=emotional_reference,
            prompt_text=prompt_text,
        )
    else:
        raise ContractError(f"unsupported Claim 5 arm kind: {kind!r}")
    record = _write_wav_and_record(
        output_path=output_path, audio=audio, sample_rate=sample_rate,
        max_clipping_fraction=max_clipping_fraction,
    )
    record["execution_kind"] = str(kind)
    record["llm_inference_calls"] = trace_calls
    return record
