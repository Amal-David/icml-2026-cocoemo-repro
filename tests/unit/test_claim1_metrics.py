from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from repro.claim1_metrics import (
    AcousticMetricSettings,
    Claim1MetricError,
    Claim1PairMetrics,
    Claim1WavPair,
    compare_acoustic_features,
    evaluate_claim1_wav_pairs,
    extract_acoustic_features,
    normalized_progress_alignment,
    _preregistered_assessment,
    summarize_claim1_metrics,
)


def _tone(
    *,
    frequency: float = 220.0,
    duration_seconds: float = 1.0,
    sample_rate: int = 16_000,
    silence_seconds: float = 0.0,
) -> np.ndarray:
    samples = int(round(duration_seconds * sample_rate))
    time = np.arange(samples, dtype=np.float64) / sample_rate
    envelope = 0.30 + 0.65 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * time))
    tone = envelope * np.sin(2.0 * np.pi * frequency * time)
    if silence_seconds:
        start = samples // 3
        end = start + int(round(silence_seconds * sample_rate))
        tone[start:end] = 0.0
    return tone


def _pair_metrics(*, condition: str, index: int, f0: float, energy: float, rate: float) -> Claim1PairMetrics:
    return Claim1PairMetrics(
        group_id=f"g{index}",
        target_emotion="angry",
        condition=condition,
        generated_wav=f"{condition}-{index}.wav",
        emotional_reference_wav=f"reference-{index}.wav",
        f0_ccc=f0,
        energy_ccc=energy,
        speaking_rate_proxy_hz=rate,
        generated_duration_seconds=1.0,
        reference_duration_seconds=1.0,
        generated_voiced_fraction=1.0,
        reference_voiced_fraction=1.0,
    )


def test_extract_acoustic_features_recovers_synthetic_tone_and_energy_nuclei() -> None:
    features = extract_acoustic_features(_tone(), 16_000)

    assert np.median(features.f0_hz) == pytest.approx(220.0, rel=0.04)
    assert features.voiced_fraction > 0.95
    assert features.speaking_rate_proxy_hz > 1.0


def test_duration_and_unvoiced_handling_are_explicit_and_deterministic() -> None:
    reference = extract_acoustic_features(_tone(duration_seconds=1.0), 16_000)
    generated = extract_acoustic_features(_tone(duration_seconds=1.5, silence_seconds=0.20), 16_000)
    f0_ccc, energy_ccc = compare_acoustic_features(generated, reference)

    assert generated.duration_seconds > reference.duration_seconds
    assert generated.voiced_fraction < reference.voiced_fraction
    assert f0_ccc > 0.95
    assert np.isfinite(energy_ccc)
    assert normalized_progress_alignment([1.0, 3.0, 5.0], points=7)[0] == pytest.approx(1.0)


def test_feature_extraction_fails_loudly_for_silence() -> None:
    with pytest.raises(Claim1MetricError, match="silence floor"):
        extract_acoustic_features(np.zeros(16_000), 16_000)


def test_wav_evaluator_requires_complete_conditions_and_reads_synthetic_audio(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    torchaudio.save(reference, torch.from_numpy(_tone()).float().unsqueeze(0), 16_000)
    generated: list[Path] = []
    for condition, frequency in (("slm_driven", 220.0), ("flow_driven", 230.0), ("emotional_both", 210.0)):
        path = tmp_path / f"{condition}.wav"
        torchaudio.save(path, torch.from_numpy(_tone(frequency=frequency)).float().unsqueeze(0), 16_000)
        generated.append(path)
    pairs = [
        Claim1WavPair("g1", "angry", condition, path, reference)
        for condition, path in zip(("slm_driven", "flow_driven", "emotional_both"), generated)
    ]

    evaluated = evaluate_claim1_wav_pairs(pairs)
    assert len(evaluated) == 3
    assert all(np.isfinite(item.f0_ccc) for item in evaluated)
    with pytest.raises(Claim1MetricError, match="incomplete acoustic conditions"):
        evaluate_claim1_wav_pairs(pairs[:2])


def test_paired_summary_is_deterministic_and_preserves_rate_pairing() -> None:
    metrics: list[Claim1PairMetrics] = []
    for index, (slm_rate, flow_rate) in enumerate(((2.0, 1.0), (6.0, 5.0), (10.0, 9.0)), start=1):
        metrics.extend(
            [
                _pair_metrics(condition="slm_driven", index=index, f0=0.1 * index, energy=0.2 * index, rate=slm_rate),
                _pair_metrics(condition="flow_driven", index=index, f0=0.3 * index, energy=0.4 * index, rate=flow_rate),
                _pair_metrics(condition="emotional_both", index=index, f0=0.5, energy=0.6, rate=3.0),
            ]
        )
    settings = replace(AcousticMetricSettings(), bootstrap_replicates=500, bootstrap_seed=17)

    first = summarize_claim1_metrics(metrics, settings=settings)
    second = summarize_claim1_metrics(metrics, settings=settings)

    assert first == second
    comparison = first["paired_comparisons"]["slm_driven_minus_flow_driven"]
    assert comparison["f0_ccc_mean_difference"]["estimate"] == pytest.approx(-0.4)
    assert comparison["energy_ccc_mean_difference"]["estimate"] == pytest.approx(-0.4)
    assert comparison["speaking_rate_proxy_population_sd_difference_hz"]["estimate"] == pytest.approx(0.0)


def test_paired_summary_rejects_nonindependent_or_incomplete_denominators() -> None:
    one_group = [
        _pair_metrics(condition=condition, index=1, f0=0.1, energy=0.2, rate=1.0)
        for condition in ("slm_driven", "flow_driven", "emotional_both")
    ]
    with pytest.raises(Claim1MetricError, match="at least two independent"):
        summarize_claim1_metrics(one_group)

    incomplete = one_group + [
        _pair_metrics(condition="slm_driven", index=2, f0=0.1, energy=0.2, rate=1.0)
    ]
    with pytest.raises(Claim1MetricError, match="incomplete metric conditions"):
        summarize_claim1_metrics(incomplete)


def test_scaled_condition_summary_keeps_controls_and_clusters_by_actor() -> None:
    conditions = (
        "neutral_both",
        "slm_driven",
        "flow_driven",
        "emotional_both",
        "permuted_emotion_both",
    )
    metrics: list[Claim1PairMetrics] = []
    for index in range(1, 3):
        for condition in conditions:
            metrics.append(
                _pair_metrics(
                    condition=condition,
                    index=index,
                    f0=0.1 if condition == "slm_driven" else 0.3,
                    energy=0.2 if condition == "slm_driven" else 0.4,
                    rate=float(index),
                )
            )
    settings = replace(AcousticMetricSettings(), bootstrap_replicates=100, bootstrap_seed=5)

    summary = summarize_claim1_metrics(metrics, conditions=conditions, settings=settings)

    assert summary["accounting"]["conditions"] == list(conditions)
    assert summary["accounting"]["actor_clusters"] == 2
    assert summary["conditions"]["permuted_emotion_both"]["n"] == 2
    assert summary["preregistered_directional_assessment"]["all_reported_directions_match"] is False
    assert summary["preregistered_directional_assessment"]["directional_outcome"] == "directional_inconclusive"
    assert summary["preregistered_directional_assessment"]["directional_inconclusive"] is True


def test_directional_assessment_uses_signed_intervals_not_paper_coverage() -> None:
    comparison = {
        "f0_ccc_mean_difference": {"estimate": -0.1, "lower": -0.2, "upper": -0.01},
        "energy_ccc_mean_difference": {"estimate": -0.1, "lower": -0.2, "upper": -0.01},
        "speaking_rate_proxy_population_sd_difference_hz": {
            "estimate": 0.1,
            "lower": 0.01,
            "upper": 0.2,
        },
    }
    supported = _preregistered_assessment(comparison)
    assert supported["directional_outcome"] == "directional_supported"
    assert supported["directional_supported"] is True

    failed_comparison = dict(comparison)
    failed_comparison["f0_ccc_mean_difference"] = {
        "estimate": 0.1,
        "lower": 0.01,
        "upper": 0.2,
    }
    failed = _preregistered_assessment(failed_comparison)
    assert failed["directional_outcome"] == "directional_failed"
    assert failed["directional_failed"] is True

    inconclusive_comparison = dict(comparison)
    inconclusive_comparison["f0_ccc_mean_difference"] = {
        "estimate": -0.01,
        "lower": -0.1,
        "upper": 0.1,
    }
    inconclusive = _preregistered_assessment(inconclusive_comparison)
    assert inconclusive["directional_outcome"] == "directional_inconclusive"
    assert inconclusive["directional_inconclusive"] is True
