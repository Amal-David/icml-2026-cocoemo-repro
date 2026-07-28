from __future__ import annotations

import pytest

from repro.claim4_evaluators import (
    FrozenClaim4Evaluators,
    _load_pinned_emotion_model,
    canonical_emotion_embedding,
    canonical_emotion_probabilities,
    claim4_proportion_metrics,
    word_error_rate,
)
from repro.contracts import ContractError


LABELS = ("angry", "happy", "neutral", "sad", "surprise")
RAW_MAP = {
    "生气/angry": "angry", "开心/happy": "happy", "中立/neutral": "neutral",
    "难过/sad": "sad", "吃惊/surprised": "surprise",
}


class _FakeEmotionModel:
    def generate(self, _: str, **__: object) -> list[dict[str, object]]:
        return [{"labels": list(RAW_MAP), "scores": [0.4, 0.2, 0.1, 0.2, 0.1]}]


def test_emotion2vec_parser_requires_all_frozen_labels_and_keeps_scores() -> None:
    output = [{"labels": list(RAW_MAP), "scores": [0.4, 0.2, 0.1, 0.2, 0.1]}]
    assert canonical_emotion_probabilities(output, raw_label_map=RAW_MAP, canonical_labels=LABELS) == {
        "angry": 0.4, "happy": 0.2, "neutral": 0.1, "sad": 0.2, "surprise": 0.1,
    }
    with pytest.raises(ContractError, match="missing a pinned canonical"):
        canonical_emotion_probabilities(
            [{"labels": list(RAW_MAP)[:-1], "scores": [0.4, 0.2, 0.1, 0.2]}],
            raw_label_map=RAW_MAP, canonical_labels=LABELS,
        )


def test_frozen_evaluator_emotion_adapter_uses_the_exact_fake_model_output(tmp_path) -> None:
    evaluators = FrozenClaim4Evaluators(
        emotion_model=_FakeEmotionModel(), speaker_feature_extractor=None, speaker_model=None,
        asr_processor=None, asr_model=None, device="cuda", raw_label_map=RAW_MAP,
        canonical_labels=LABELS, language="en",
    )
    assert evaluators.emotion_probabilities(tmp_path / "sample.wav")["angry"] == 0.4


def test_emotion_embedding_adapter_rejects_ambiguous_or_nonfinite_payloads() -> None:
    embedding = canonical_emotion_embedding([{"embedding": [[1.0, 2.0, 3.0]]}])
    assert embedding.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ContractError, match="exactly one"):
        canonical_emotion_embedding([{"embedding": [1.0, 2.0], "feats": [1.0, 2.0]}])
    with pytest.raises(ContractError, match="finite"):
        canonical_emotion_embedding([{"embedding": [1.0, float("nan")]}])


def test_funasr_receives_a_local_snapshot_of_the_exact_pinned_revision(tmp_path) -> None:
    snapshot = tmp_path / "emotion2vec-pinned"
    snapshot.mkdir()
    calls: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        calls["snapshot"] = kwargs
        return str(snapshot)

    def fake_auto_model(**kwargs: object) -> object:
        calls["auto_model"] = kwargs
        return object()

    model = _load_pinned_emotion_model(
        {"id": "emotion2vec/emotion2vec_plus_large", "revision": "exact-revision"},
        device="cuda", snapshot_download=fake_snapshot_download, auto_model=fake_auto_model,
    )
    assert model is not None
    assert calls["snapshot"] == {
        "repo_id": "emotion2vec/emotion2vec_plus_large", "revision": "exact-revision",
    }
    assert calls["auto_model"] == {"model": str(snapshot), "device": "cuda"}


def test_claim4_metrics_are_relative_to_identity_and_require_defined_rank() -> None:
    identity = {"angry": 0.1, "happy": 0.1, "neutral": 0.6, "sad": 0.1, "surprise": 0.1}
    arm = {"angry": 0.5, "happy": 0.2, "neutral": 0.1, "sad": 0.1, "surprise": 0.1}
    rho, hit = claim4_proportion_metrics(
        target_distribution={"p_angry": 0.5, "p_happy": 0.2, "p_sad": 0.0, "p_surprise": 0.0, "p_neutral": 0.3},
        arm_probabilities=arm, identity_probabilities=identity,
    )
    assert rho == pytest.approx(1.0)
    assert hit == 1.0
    with pytest.raises(ContractError, match="Spearman rho is undefined"):
        claim4_proportion_metrics(
            target_distribution={"p_angry": 0.5, "p_happy": 0.2, "p_sad": 0.0, "p_surprise": 0.0, "p_neutral": 0.3},
            arm_probabilities=identity, identity_probabilities=identity,
        )


def test_word_error_rate_is_deterministic_and_normalizes_case_and_punctuation() -> None:
    assert word_error_rate("Don't forget a jacket.", "don't forget a jacket") == 0.0
    assert word_error_rate("one two", "one three four") == 1.0
