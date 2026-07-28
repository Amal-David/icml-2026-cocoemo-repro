"""Frozen evaluator adapters for the public Claim 4 directional proxy.

The adapters intentionally have a small, explicit surface.  They load the
three revisions pinned by ``claim4_runtime.parse_claim4_settings`` and reject
any incomplete or malformed result instead of substituting a fallback score.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from cocoemo.evaluation.mixed_metrics import spearman_rank_corr
from repro.claim4_evaluation import NON_NEUTRAL_EMOTIONS
from repro.contracts import ContractError


class EmotionModel(Protocol):
    def generate(self, wav_path: str, **kwargs: Any) -> Any: ...


class SpeakerModel(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


class ASRModel(Protocol):
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...


_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def canonical_emotion_probabilities(
    result: Any, *, raw_label_map: Mapping[str, str], canonical_labels: tuple[str, ...]
) -> dict[str, float]:
    """Parse one Emotion2Vec utterance result without accepting partial scores."""

    if not isinstance(result, (list, tuple)) or len(result) != 1 or not isinstance(result[0], Mapping):
        raise ContractError("Emotion2Vec must return exactly one utterance result")
    payload = result[0]
    labels = payload.get("labels")
    scores = payload.get("scores")
    if not isinstance(labels, (list, tuple)) or not isinstance(scores, (list, tuple)) or len(labels) != len(scores):
        raise ContractError("Emotion2Vec result must provide aligned labels and scores")
    mapped: dict[str, float] = {}
    for raw_label, raw_score in zip(labels, scores):
        if raw_label not in raw_label_map:
            continue
        label = raw_label_map[raw_label]
        if label in mapped:
            raise ContractError(f"Emotion2Vec emitted duplicate canonical label: {label}")
        if not isinstance(raw_score, (int, float)) or not math.isfinite(float(raw_score)) or float(raw_score) < 0:
            raise ContractError(f"Emotion2Vec emitted an invalid score for {label}")
        mapped[label] = float(raw_score)
    if set(mapped) != set(canonical_labels):
        raise ContractError(
            "Emotion2Vec result is missing a pinned canonical label: "
            f"missing={sorted(set(canonical_labels) - set(mapped))}"
        )
    return {label: mapped[label] for label in canonical_labels}


def claim4_proportion_metrics(
    *, target_distribution: Mapping[str, float], arm_probabilities: Mapping[str, float],
    identity_probabilities: Mapping[str, float],
) -> tuple[float, float]:
    """Return the predeclared per-item rank and dominant-hit statistics.

    Both statistics use the identity-neutral arm as the same-item probability
    baseline.  Only positive non-neutral target masses participate; the frozen
    selection guarantees at least two and one unique maximum.
    """

    target = []
    increases = []
    for emotion in NON_NEUTRAL_EMOTIONS:
        key = f"p_{emotion}"
        value = target_distribution.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ContractError(f"Claim 4 target distribution has invalid {key}")
        if emotion not in arm_probabilities or emotion not in identity_probabilities:
            raise ContractError(f"Claim 4 Emotion2Vec probabilities lack {emotion}")
        if value > 0:
            target.append((emotion, float(value)))
            increase = float(arm_probabilities[emotion]) - float(identity_probabilities[emotion])
            if not math.isfinite(increase):
                raise ContractError(f"Claim 4 has non-finite probability increase for {emotion}")
            increases.append(increase)
    if len(target) < 2:
        raise ContractError("Claim 4 proportion metrics require two positive non-neutral target masses")
    target_values = [value for _, value in target]
    rho = spearman_rank_corr(target_values, increases)
    if rho is None or not math.isfinite(float(rho)):
        raise ContractError("Claim 4 Spearman rho is undefined for this evaluator output")
    target_max = max(target_values)
    dominant_indices = [index for index, value in enumerate(target_values) if value == target_max]
    if len(dominant_indices) != 1:
        raise ContractError("Claim 4 target must have one unique non-neutral dominant emotion")
    max_increase = max(increases)
    h_rate = 1.0 if increases[dominant_indices[0]] == max_increase else 0.0
    return float(rho), h_rate


def normalized_words(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ContractError("Whisper reference and hypothesis text must be non-empty strings")
    normal = unicodedata.normalize("NFKC", text).lower()
    normal = _WORD.sub(" ", normal)
    words = normal.split()
    if not words:
        raise ContractError("Whisper text normalization produced no words")
    return words


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Deterministic lowercase, Unicode-normalized word error rate."""

    reference_words = normalized_words(reference)
    hypothesis_words = normalized_words(hypothesis)
    previous = list(range(len(hypothesis_words) + 1))
    for index, expected in enumerate(reference_words, start=1):
        current = [index]
        for candidate_index, observed in enumerate(hypothesis_words, start=1):
            current.append(min(
                previous[candidate_index] + 1,
                current[candidate_index - 1] + 1,
                previous[candidate_index - 1] + (expected != observed),
            ))
        previous = current
    value = previous[-1] / len(reference_words)
    if not math.isfinite(value) or value < 0:
        raise ContractError("Whisper WER computation failed")
    return float(value)


def _load_pinned_emotion_model(
    emotion: Mapping[str, Any], *, device: str, snapshot_download: Any, auto_model: Any
) -> Any:
    """Materialize the exact Hub commit before handing a local path to FunASR.

    FunASR's HF route does not enforce ``model_revision``.  Passing its model
    loader a local snapshot prevents it from resolving a moving branch later.
    """

    snapshot = Path(snapshot_download(repo_id=emotion["id"], revision=emotion["revision"]))
    if not snapshot.is_dir():
        raise ContractError("Claim 4 Emotion2Vec snapshot download did not produce a directory")
    return auto_model(model=str(snapshot), device=device)


@dataclass
class FrozenClaim4Evaluators:
    """Loaded model handles and their pinned, failure-closed scoring adapters."""

    emotion_model: EmotionModel
    speaker_feature_extractor: Any
    speaker_model: SpeakerModel
    asr_processor: Any
    asr_model: ASRModel
    device: str
    raw_label_map: Mapping[str, str]
    canonical_labels: tuple[str, ...]
    language: str

    def emotion_probabilities(self, wav_path: Path) -> dict[str, float]:
        return canonical_emotion_probabilities(
            self.emotion_model.generate(
                str(wav_path), output_dir=None, granularity="utterance", extract_embedding=False,
            ),
            raw_label_map=self.raw_label_map,
            canonical_labels=self.canonical_labels,
        )

    def speaker_similarity(self, wav_path: Path, reference_wav: Path) -> float:
        try:
            import torch
            import torchaudio
        except ImportError as exc:
            raise ContractError("Claim 4 WavLM evaluation requires torch and torchaudio") from exc
        target, target_rate = torchaudio.load(wav_path)
        reference, reference_rate = torchaudio.load(reference_wav)
        target = target.float().mean(dim=0) if target.shape[0] > 1 else target.float().squeeze(0)
        reference = reference.float().mean(dim=0) if reference.shape[0] > 1 else reference.float().squeeze(0)
        sample_rate = getattr(self.speaker_feature_extractor, "sampling_rate", None)
        if not isinstance(sample_rate, int) or sample_rate < 1:
            raise ContractError("Claim 4 WavLM feature extractor has no valid sampling rate")
        if target_rate != sample_rate:
            target = torchaudio.functional.resample(target, target_rate, sample_rate)
        if reference_rate != sample_rate:
            reference = torchaudio.functional.resample(reference, reference_rate, sample_rate)
        inputs = self.speaker_feature_extractor(
            [target.cpu().numpy(), reference.cpu().numpy()],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        if not isinstance(inputs, Mapping):
            raise ContractError("Claim 4 WavLM feature extractor returned a non-mapping")
        prepared = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.speaker_model(**prepared)
            embeddings = getattr(output, "embeddings", None)
            if embeddings is None or len(embeddings) != 2:
                raise ContractError("Claim 4 WavLM returned an invalid embedding batch")
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            value = float(torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item())
        if not math.isfinite(value) or value < -1.000001 or value > 1.000001:
            raise ContractError("Claim 4 WavLM speaker similarity is invalid")
        return value

    def transcribe(self, wav_path: Path) -> str:
        try:
            import torch
            import torchaudio
        except ImportError as exc:
            raise ContractError("Claim 4 Whisper evaluation requires torch and torchaudio") from exc
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.float().mean(dim=0) if waveform.shape[0] > 1 else waveform.float().squeeze(0)
        feature_extractor = getattr(self.asr_processor, "feature_extractor", None)
        target_rate = getattr(feature_extractor, "sampling_rate", None)
        if not isinstance(target_rate, int) or target_rate < 1:
            raise ContractError("Claim 4 Whisper processor has no valid sampling rate")
        if sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        inputs = self.asr_processor(waveform.cpu().numpy(), sampling_rate=target_rate, return_tensors="pt")
        features = getattr(inputs, "input_features", None)
        if features is None:
            raise ContractError("Claim 4 Whisper processor returned no input features")
        prompt = self.asr_processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
        with torch.no_grad():
            generated = self.asr_model.generate(features.to(self.device), forced_decoder_ids=prompt)
        decoded = self.asr_processor.batch_decode(generated, skip_special_tokens=True)
        if not isinstance(decoded, (list, tuple)) or len(decoded) != 1 or not isinstance(decoded[0], str) or not decoded[0].strip():
            raise ContractError("Claim 4 Whisper returned no transcript")
        return decoded[0].strip()

    def evaluate(self, *, wav_path: Path, reference_wav: Path, reference_text: str) -> dict[str, Any]:
        probabilities = self.emotion_probabilities(wav_path)
        hypothesis = self.transcribe(wav_path)
        return {
            "emotion_probabilities": probabilities,
            "wavlm_speaker_similarity": self.speaker_similarity(wav_path, reference_wav),
            "whisper_hypothesis": hypothesis,
            "whisper_wer": word_error_rate(reference_text, hypothesis),
        }


def load_frozen_claim4_evaluators(settings: Any, *, device: str = "cuda") -> FrozenClaim4Evaluators:
    """Load only the three revisions frozen in the Claim 4 config."""

    if device != "cuda":
        raise ContractError("Claim 4 full evaluator requires CUDA")
    try:
        import torch
        from funasr import AutoModel
        from huggingface_hub import snapshot_download
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        raise ContractError("Claim 4 evaluator dependencies are unavailable") from exc
    if not torch.cuda.is_available():
        raise ContractError("Claim 4 full evaluator requires CUDA")
    emotion = settings.evaluators["emotion2vec"]
    wavlm = settings.evaluators["wavlm"]
    whisper = settings.evaluators["whisper"]
    try:
        emotion_model = _load_pinned_emotion_model(
            emotion, device=device, snapshot_download=snapshot_download, auto_model=AutoModel,
        )
        speaker_feature_extractor = AutoFeatureExtractor.from_pretrained(
            wavlm["id"], revision=wavlm["revision"], trust_remote_code=False,
        )
        speaker_model = AutoModelForAudioXVector.from_pretrained(
            wavlm["id"], revision=wavlm["revision"], trust_remote_code=False,
        ).to(device).eval()
        asr_processor = AutoProcessor.from_pretrained(
            whisper["id"], revision=whisper["revision"], trust_remote_code=False,
        )
        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper["id"], revision=whisper["revision"], trust_remote_code=False,
        ).to(device).eval()
    except Exception as exc:
        raise ContractError("Claim 4 failed to load a pinned evaluator revision") from exc
    return FrozenClaim4Evaluators(
        emotion_model=emotion_model,
        speaker_feature_extractor=speaker_feature_extractor,
        speaker_model=speaker_model,
        asr_processor=asr_processor,
        asr_model=asr_model,
        device=device,
        raw_label_map=emotion["raw_label_map"],
        canonical_labels=tuple(emotion["canonical_labels"]),
        language=whisper["language"],
    )
