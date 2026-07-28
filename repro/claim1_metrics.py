"""Strict acoustic evaluator for CoCoEmo Claim 1 cross-conditioning runs.

The paper reports F0 CCC, energy CCC, and speaking-rate standard deviation,
but does not specify a feature extractor, an alignment rule, or its exact
definition of speaking rate. This module fixes those choices before any
evaluation and records the resulting estimates as a directional public-data
test rather than as an exact Table 1 reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torchaudio

from repro.contracts import ContractError
from repro.stats import (
    BootstrapResult,
    concordance_correlation_coefficient,
)


CLAIM1_CONDITIONS = ("slm_driven", "flow_driven", "emotional_both")


class Claim1MetricError(ContractError):
    pass


@dataclass(frozen=True)
class AcousticMetricSettings:
    """Frozen feature choices for the public RAVDESS directional evaluation."""

    analysis_sample_rate: int = 16_000
    frame_seconds: float = 0.025
    hop_seconds: float = 0.010
    min_f0_hz: float = 50.0
    max_f0_hz: float = 500.0
    min_periodicity: float = 0.35
    activity_dynamic_range_db: float = 40.0
    silence_floor_db: float = -70.0
    alignment_points: int = 100
    minimum_voiced_frames: int = 3
    minimum_active_frames: int = 3
    syllable_peak_min_separation_seconds: float = 0.120
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20260728

    def validate(self) -> None:
        if self.analysis_sample_rate < 8_000:
            raise Claim1MetricError("analysis_sample_rate must be at least 8000 Hz")
        if not 0.005 <= self.hop_seconds < self.frame_seconds <= 0.100:
            raise Claim1MetricError("frame_seconds and hop_seconds must satisfy 0.005 <= hop < frame <= 0.100")
        if not 0 < self.min_f0_hz < self.max_f0_hz:
            raise Claim1MetricError("F0 bounds must be positive and ordered")
        if not 0.0 < self.min_periodicity <= 1.0:
            raise Claim1MetricError("min_periodicity must be in (0, 1]")
        if self.alignment_points < 2:
            raise Claim1MetricError("alignment_points must be at least two")
        if self.minimum_voiced_frames < 2 or self.minimum_active_frames < 2:
            raise Claim1MetricError("minimum frame counts must be at least two")
        if self.bootstrap_replicates < 100:
            raise Claim1MetricError("at least 100 bootstrap replicates are required")


def parse_acoustic_metric_settings(values: object) -> AcousticMetricSettings:
    """Read a fully pinned acoustic evaluator configuration.

    A scaled run may not inherit silently changing defaults.  The smoke path
    deliberately does not evaluate acoustics and therefore does not use this
    parser.
    """

    if not isinstance(values, dict):
        raise Claim1MetricError("claim1.acoustic_metrics must be a mapping")
    names = {field.name for field in fields(AcousticMetricSettings)}
    unknown = sorted(set(values).difference(names))
    missing = sorted(names.difference(values))
    if unknown or missing:
        details: list[str] = []
        if unknown:
            details.append("unknown=" + ", ".join(unknown))
        if missing:
            details.append("missing=" + ", ".join(missing))
        raise Claim1MetricError("claim1.acoustic_metrics must pin every setting: " + "; ".join(details))
    try:
        result = AcousticMetricSettings(**values)
    except TypeError as exc:
        raise Claim1MetricError("invalid claim1.acoustic_metrics values") from exc
    result.validate()
    return result


@dataclass(frozen=True)
class AcousticFeatures:
    duration_seconds: float
    frame_count: int
    active_frame_count: int
    voiced_frame_count: int
    voiced_fraction: float
    f0_hz: tuple[float, ...]
    log_energy_db: tuple[float, ...]
    speaking_rate_proxy_hz: float


@dataclass(frozen=True)
class Claim1WavPair:
    group_id: str
    target_emotion: str
    condition: str
    generated_wav: Path
    emotional_reference_wav: Path


@dataclass(frozen=True)
class Claim1PairMetrics:
    group_id: str
    target_emotion: str
    condition: str
    generated_wav: str
    emotional_reference_wav: str
    f0_ccc: float
    energy_ccc: float
    speaking_rate_proxy_hz: float
    generated_duration_seconds: float
    reference_duration_seconds: float
    generated_voiced_fraction: float
    reference_voiced_fraction: float


def _as_mono_float(waveform: np.ndarray) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim == 1:
        mono = values
    elif values.ndim == 2 and values.shape[0] >= 1:
        mono = np.mean(values, axis=0)
    else:
        raise Claim1MetricError("waveform must have shape (frames,) or (channels, frames)")
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise Claim1MetricError("waveform must be non-empty and finite")
    return mono


def _resample_waveform(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate < 1:
        raise Claim1MetricError("sample rate must be positive")
    if source_rate == target_rate:
        return waveform
    source = np.asarray(waveform, dtype=np.float32)
    tensor = torchaudio.functional.resample(
        # Torchaudio expects channels first. Conversion happens once at the boundary.
        torch.from_numpy(source).unsqueeze(0), source_rate, target_rate
    )
    return tensor.squeeze(0).numpy().astype(np.float64, copy=False)


def _frame_signal(waveform: np.ndarray, *, frame_samples: int, hop_samples: int) -> np.ndarray:
    if waveform.size < frame_samples:
        raise Claim1MetricError(
            f"waveform is too short for one analysis frame: samples={waveform.size}, frame={frame_samples}"
        )
    frame_count = 1 + (waveform.size - frame_samples) // hop_samples
    frames = np.empty((frame_count, frame_samples), dtype=np.float64)
    for index in range(frame_count):
        start = index * hop_samples
        frames[index] = waveform[start : start + frame_samples]
    return frames


def _normalized_autocorrelation_f0(
    frame: np.ndarray,
    *,
    sample_rate: int,
    min_f0_hz: float,
    max_f0_hz: float,
    min_periodicity: float,
) -> float | None:
    centered = (frame - np.mean(frame)) * np.hamming(frame.size)
    energy = float(np.dot(centered, centered))
    if energy <= np.finfo(np.float64).eps:
        return None
    correlation = np.correlate(centered, centered, mode="full")[frame.size - 1 :]
    minimum_lag = max(1, int(np.floor(sample_rate / max_f0_hz)))
    maximum_lag = min(correlation.size - 1, int(np.ceil(sample_rate / min_f0_hz)))
    if maximum_lag <= minimum_lag:
        raise Claim1MetricError("analysis frame does not support configured F0 bounds")
    candidates = correlation[minimum_lag : maximum_lag + 1] / correlation[0]
    lag = minimum_lag + int(np.argmax(candidates))
    periodicity = float(candidates[lag - minimum_lag])
    if periodicity < min_periodicity:
        return None
    return float(sample_rate / lag)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, np.full(width, 1.0 / width), mode="valid")


def _speaking_rate_proxy(
    log_energy_db: np.ndarray,
    active: np.ndarray,
    *,
    hop_seconds: float,
    min_separation_seconds: float,
) -> float:
    active_count = int(np.count_nonzero(active))
    if active_count == 0:
        raise Claim1MetricError("cannot estimate speaking-rate proxy without active frames")
    smoothed = _smooth(log_energy_db, max(1, int(round(0.080 / hop_seconds))))
    active_values = smoothed[active]
    prominence_floor = float(np.min(active_values) + 0.35 * (np.max(active_values) - np.min(active_values)))
    minimum_distance = max(1, int(round(min_separation_seconds / hop_seconds)))
    peaks: list[int] = []
    for index in range(1, smoothed.size - 1):
        if not active[index] or smoothed[index] < prominence_floor:
            continue
        if smoothed[index] >= smoothed[index - 1] and smoothed[index] > smoothed[index + 1]:
            if peaks and index - peaks[-1] < minimum_distance:
                if smoothed[index] > smoothed[peaks[-1]]:
                    peaks[-1] = index
            else:
                peaks.append(index)
    active_seconds = active_count * hop_seconds
    return float(len(peaks) / active_seconds)


def extract_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    settings: AcousticMetricSettings = AcousticMetricSettings(),
) -> AcousticFeatures:
    """Extract F0, log-energy, and a deterministic vocalic-nuclei rate proxy.

    F0 uses a normalized-autocorrelation estimator on voiced frames. The rate
    proxy is the count of separated log-energy nuclei divided by active speech
    time; it is deliberately named a proxy because the paper provides no
    implementation or syllabification definition for speaking rate.
    """

    settings.validate()
    mono = _as_mono_float(waveform)
    signal = _resample_waveform(mono, sample_rate, settings.analysis_sample_rate)
    frame_samples = int(round(settings.frame_seconds * settings.analysis_sample_rate))
    hop_samples = int(round(settings.hop_seconds * settings.analysis_sample_rate))
    frames = _frame_signal(signal, frame_samples=frame_samples, hop_samples=hop_samples)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    log_energy = 20.0 * np.log10(np.maximum(rms, np.finfo(np.float64).tiny))
    peak_db = float(np.max(log_energy))
    if peak_db < settings.silence_floor_db:
        raise Claim1MetricError(
            f"waveform is below silence floor: peak_db={peak_db:.2f}, floor_db={settings.silence_floor_db:.2f}"
        )
    active_floor = max(settings.silence_floor_db, peak_db - settings.activity_dynamic_range_db)
    active = log_energy >= active_floor
    active_count = int(np.count_nonzero(active))
    if active_count < settings.minimum_active_frames:
        raise Claim1MetricError(
            f"insufficient active frames: required={settings.minimum_active_frames}, actual={active_count}"
        )
    f0_values: list[float] = []
    for frame, is_active in zip(frames, active):
        if not is_active:
            continue
        estimate = _normalized_autocorrelation_f0(
            frame,
            sample_rate=settings.analysis_sample_rate,
            min_f0_hz=settings.min_f0_hz,
            max_f0_hz=settings.max_f0_hz,
            min_periodicity=settings.min_periodicity,
        )
        if estimate is not None:
            f0_values.append(estimate)
    if len(f0_values) < settings.minimum_voiced_frames:
        raise Claim1MetricError(
            f"insufficient voiced frames: required={settings.minimum_voiced_frames}, actual={len(f0_values)}"
        )
    return AcousticFeatures(
        duration_seconds=float(signal.size / settings.analysis_sample_rate),
        frame_count=int(frames.shape[0]),
        active_frame_count=active_count,
        voiced_frame_count=len(f0_values),
        voiced_fraction=float(len(f0_values) / frames.shape[0]),
        f0_hz=tuple(f0_values),
        log_energy_db=tuple(float(value) for value in log_energy),
        speaking_rate_proxy_hz=_speaking_rate_proxy(
            log_energy,
            active,
            hop_seconds=settings.hop_seconds,
            min_separation_seconds=settings.syllable_peak_min_separation_seconds,
        ),
    )


def normalized_progress_alignment(values: Sequence[float], *, points: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size < 2 or not np.all(np.isfinite(source)):
        raise Claim1MetricError("aligned profile must contain at least two finite values")
    if points < 2:
        raise Claim1MetricError("alignment points must be at least two")
    source_progress = np.linspace(0.0, 1.0, source.size)
    target_progress = np.linspace(0.0, 1.0, points)
    return np.interp(target_progress, source_progress, source)


def compare_acoustic_features(
    generated: AcousticFeatures,
    reference: AcousticFeatures,
    *,
    settings: AcousticMetricSettings = AcousticMetricSettings(),
) -> tuple[float, float]:
    """Compute CCCs after deterministic duration and voicing handling.

    Energy is aligned over complete frame trajectories by normalized duration.
    F0 first removes unvoiced frames in each waveform, then aligns the retained
    trajectories by normalized voiced-frame rank. This prevents an unmatched
    silence or duration from being treated as an F0 value; voicing fractions
    remain in each per-sample record for audit.
    """

    settings.validate()
    generated_f0 = normalized_progress_alignment(generated.f0_hz, points=settings.alignment_points)
    reference_f0 = normalized_progress_alignment(reference.f0_hz, points=settings.alignment_points)
    generated_energy = normalized_progress_alignment(
        generated.log_energy_db, points=settings.alignment_points
    )
    reference_energy = normalized_progress_alignment(
        reference.log_energy_db, points=settings.alignment_points
    )
    return (
        concordance_correlation_coefficient(generated_f0, reference_f0),
        concordance_correlation_coefficient(generated_energy, reference_energy),
    )


def _load_wav(path: Path, *, settings: AcousticMetricSettings) -> AcousticFeatures:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise Claim1MetricError(f"audio file does not exist: {source}")
    waveform, sample_rate = torchaudio.load(source)
    return extract_acoustic_features(
        waveform.detach().cpu().numpy(), sample_rate, settings=settings
    )


def _validate_pair_accounting(
    pairs: Iterable[Claim1WavPair], *, conditions: tuple[str, ...]
) -> list[Claim1WavPair]:
    expected_conditions = set(conditions)
    if not expected_conditions or len(expected_conditions) != len(conditions):
        raise Claim1MetricError("expected conditions must be non-empty and unique")
    by_key: dict[tuple[str, str], dict[str, Claim1WavPair]] = {}
    for pair in pairs:
        if not pair.group_id or not pair.target_emotion:
            raise Claim1MetricError("each acoustic pair needs group_id and target_emotion")
        key = (pair.group_id, pair.target_emotion)
        condition_records = by_key.setdefault(key, {})
        if pair.condition not in expected_conditions:
            raise Claim1MetricError(f"unsupported acoustic condition: {pair.condition}")
        if pair.condition in condition_records:
            raise Claim1MetricError(
                f"duplicate acoustic pair: group={pair.group_id}, emotion={pair.target_emotion}, condition={pair.condition}"
            )
        condition_records[pair.condition] = pair
    if not by_key:
        raise Claim1MetricError("no Claim 1 acoustic pairs were provided")
    result: list[Claim1WavPair] = []
    for key, condition_records in sorted(by_key.items()):
        actual_conditions = set(condition_records)
        if actual_conditions != expected_conditions:
            raise Claim1MetricError(
                f"incomplete acoustic conditions for group={key[0]}, emotion={key[1]}: "
                f"expected={sorted(expected_conditions)}, actual={sorted(actual_conditions)}"
            )
        references = {pair.emotional_reference_wav.expanduser().resolve() for pair in condition_records.values()}
        if len(references) != 1:
            raise Claim1MetricError(
                f"condition references differ for group={key[0]}, emotion={key[1]}"
            )
        result.extend(condition_records[condition] for condition in conditions)
    return result


def evaluate_claim1_wav_pairs(
    pairs: Iterable[Claim1WavPair],
    *,
    conditions: tuple[str, ...] = CLAIM1_CONDITIONS,
    settings: AcousticMetricSettings = AcousticMetricSettings(),
) -> list[Claim1PairMetrics]:
    """Evaluate complete cross-conditioned output sets without filling gaps."""

    validated = _validate_pair_accounting(pairs, conditions=conditions)
    cache: dict[Path, AcousticFeatures] = {}

    def features(path: Path) -> AcousticFeatures:
        resolved = path.expanduser().resolve()
        if resolved not in cache:
            cache[resolved] = _load_wav(resolved, settings=settings)
        return cache[resolved]

    results: list[Claim1PairMetrics] = []
    for pair in validated:
        generated = features(pair.generated_wav)
        reference = features(pair.emotional_reference_wav)
        f0_ccc, energy_ccc = compare_acoustic_features(generated, reference, settings=settings)
        results.append(
            Claim1PairMetrics(
                group_id=pair.group_id,
                target_emotion=pair.target_emotion,
                condition=pair.condition,
                generated_wav=str(pair.generated_wav),
                emotional_reference_wav=str(pair.emotional_reference_wav),
                f0_ccc=f0_ccc,
                energy_ccc=energy_ccc,
                speaking_rate_proxy_hz=generated.speaking_rate_proxy_hz,
                generated_duration_seconds=generated.duration_seconds,
                reference_duration_seconds=reference.duration_seconds,
                generated_voiced_fraction=generated.voiced_fraction,
                reference_voiced_fraction=reference.voiced_fraction,
            )
        )
    return results


def _bootstrap_payload(result: BootstrapResult) -> dict[str, float | int]:
    return {
        "estimate": result.estimate,
        "lower": result.lower,
        "upper": result.upper,
        "replicates": result.replicates,
        "seed": result.seed,
    }


def _population_standard_deviation(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise Claim1MetricError("condition standard deviation needs at least two finite observations")
    return float(np.std(array, ddof=0))


def _actor_cluster_key(group_id: str) -> str:
    """Return the RAVDESS actor cluster while retaining unit-test portability."""

    parts = group_id.split("_")
    if len(parts) >= 2 and parts[0] == "actor" and parts[1].isdigit():
        return f"actor_{parts[1]}"
    return f"group_{group_id}"


def _clustered_paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    clusters: Sequence[str],
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """Resample RAVDESS actors, retaining every nested group/emotion pair.

    The four target emotions within a group share one neutral source and the
    four groups for an actor share a speaker.  Treating all group-emotion rows
    as independent would materially understate uncertainty.
    """

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise Claim1MetricError("clustered paired bootstrap needs matched vectors with at least two rows")
    if len(clusters) != x.size or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise Claim1MetricError("clustered paired bootstrap inputs are invalid")
    if replicates < 100:
        raise Claim1MetricError("at least 100 bootstrap replicates are required")

    index_by_cluster: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        index_by_cluster.setdefault(cluster, []).append(index)
    cluster_ids = sorted(index_by_cluster)
    if len(cluster_ids) < 2:
        raise Claim1MetricError("clustered paired bootstrap requires at least two actor clusters")

    def evaluate(indices: np.ndarray) -> float:
        value = float(statistic(x[indices], y[indices]))
        if not np.isfinite(value):
            raise Claim1MetricError("clustered paired bootstrap statistic must be finite")
        return value

    all_indices = np.arange(x.size)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        sampled_clusters = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        indices = np.concatenate(
            [np.asarray(index_by_cluster[cluster_ids[index]], dtype=np.int64) for index in sampled_clusters]
        )
        draws[draw] = evaluate(indices)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return BootstrapResult(
        estimate=evaluate(all_indices),
        lower=float(lower),
        upper=float(upper),
        replicates=replicates,
        seed=seed,
    )


def _interval_contains(interval: dict[str, float | int], target: float) -> bool:
    return float(interval["lower"]) <= target <= float(interval["upper"])


def _direction_matches(estimate: float, expected_sign: int) -> bool:
    return estimate < 0.0 if expected_sign < 0 else estimate > 0.0


def _signed_directional_outcome(
    interval: dict[str, float | int], *, expected_sign: int
) -> str:
    """Classify a preregistered direction using only its signed CI."""

    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if expected_sign < 0:
        if upper < 0.0:
            return "directional_supported"
        if lower > 0.0:
            return "directional_failed"
    else:
        if lower > 0.0:
            return "directional_supported"
        if upper < 0.0:
            return "directional_failed"
    return "directional_inconclusive"


def _preregistered_assessment(comparison: dict[str, dict[str, float | int]]) -> dict[str, object]:
    """Assess the signed Table 1 contrasts without overclaiming exact replication."""

    specifications = {
        "f0_ccc_mean_difference": {
            "paper_difference": 0.109 - 0.305,
            "expected_sign": -1,
        },
        "energy_ccc_mean_difference": {
            "paper_difference": 0.308 - 0.737,
            "expected_sign": -1,
        },
        "speaking_rate_proxy_population_sd_difference_hz": {
            "paper_difference": 0.691 - 0.518,
            "expected_sign": 1,
        },
    }
    checks: dict[str, dict[str, object]] = {}
    all_directions = True
    all_paper_contrasts_covered = True
    outcomes: list[str] = []
    for name, specification in specifications.items():
        interval = comparison[name]
        estimate = float(interval["estimate"])
        direction_matches = _direction_matches(estimate, int(specification["expected_sign"]))
        covers_paper_contrast = _interval_contains(interval, float(specification["paper_difference"]))
        directional_outcome = _signed_directional_outcome(
            interval, expected_sign=int(specification["expected_sign"])
        )
        outcomes.append(directional_outcome)
        all_directions = all_directions and direction_matches
        all_paper_contrasts_covered = all_paper_contrasts_covered and covers_paper_contrast
        checks[name] = {
            "estimate": estimate,
            "paper_difference": specification["paper_difference"],
            "direction_matches_paper": direction_matches,
            "directional_outcome": directional_outcome,
            "paper_difference_within_cluster_bootstrap_interval_diagnostic": covers_paper_contrast,
        }
    if any(outcome == "directional_failed" for outcome in outcomes):
        directional_outcome = "directional_failed"
    elif all(outcome == "directional_supported" for outcome in outcomes):
        directional_outcome = "directional_supported"
    else:
        directional_outcome = "directional_inconclusive"
    return {
        "scope": "directional public-RAVDESS proxy; not an exact Table 1 reproduction",
        "directional_outcome": directional_outcome,
        "directional_supported": directional_outcome == "directional_supported",
        "directional_failed": directional_outcome == "directional_failed",
        "directional_inconclusive": directional_outcome == "directional_inconclusive",
        "all_reported_directions_match": all_directions,
        "all_paper_contrasts_within_cluster_bootstrap_intervals_diagnostic": all_paper_contrasts_covered,
        "checks": checks,
    }


def summarize_claim1_metrics(
    metrics: Iterable[Claim1PairMetrics],
    *,
    conditions: tuple[str, ...] = CLAIM1_CONDITIONS,
    settings: AcousticMetricSettings = AcousticMetricSettings(),
) -> dict[str, object]:
    """Return auditable condition summaries and paired bootstrap comparisons."""

    settings.validate()
    items = list(metrics)
    expected_conditions = set(conditions)
    required_comparison_conditions = {"slm_driven", "flow_driven"}
    if not required_comparison_conditions.issubset(expected_conditions):
        raise Claim1MetricError(
            "Claim 1 summary requires slm_driven and flow_driven conditions"
        )
    by_key: dict[tuple[str, str], dict[str, Claim1PairMetrics]] = {}
    for item in items:
        if item.condition not in expected_conditions:
            raise Claim1MetricError(f"unsupported metric condition: {item.condition}")
        key = (item.group_id, item.target_emotion)
        existing = by_key.setdefault(key, {})
        if item.condition in existing:
            raise Claim1MetricError(f"duplicate metric record: group={key[0]}, emotion={key[1]}, condition={item.condition}")
        existing[item.condition] = item
    if len(by_key) < 2:
        raise Claim1MetricError("paired Claim 1 summaries require at least two independent group-emotion pairs")
    for key, records in sorted(by_key.items()):
        if set(records) != expected_conditions:
            raise Claim1MetricError(
                f"incomplete metric conditions for group={key[0]}, emotion={key[1]}"
            )

    ordered_keys = sorted(by_key)
    by_condition = {
        condition: [by_key[key][condition] for key in ordered_keys] for condition in conditions
    }
    condition_summary: dict[str, dict[str, float | int]] = {}
    for condition, records in by_condition.items():
        f0 = np.asarray([record.f0_ccc for record in records], dtype=np.float64)
        energy = np.asarray([record.energy_ccc for record in records], dtype=np.float64)
        rate = np.asarray([record.speaking_rate_proxy_hz for record in records], dtype=np.float64)
        if not np.all(np.isfinite(f0)) or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(rate)):
            raise Claim1MetricError(f"non-finite metric found for condition: {condition}")
        condition_summary[condition] = {
            "n": int(len(records)),
            "f0_ccc_mean": float(np.mean(f0)),
            "energy_ccc_mean": float(np.mean(energy)),
            "speaking_rate_proxy_mean_hz": float(np.mean(rate)),
            "speaking_rate_proxy_population_sd_hz": _population_standard_deviation(rate),
        }

    left = by_condition["slm_driven"]
    right = by_condition["flow_driven"]
    def values(records: Sequence[Claim1PairMetrics], attribute: str) -> np.ndarray:
        return np.asarray([getattr(record, attribute) for record in records], dtype=np.float64)

    clusters = [_actor_cluster_key(record.group_id) for record in left]
    f0_comparison = _clustered_paired_bootstrap(
        values(left, "f0_ccc"),
        values(right, "f0_ccc"),
        clusters,
        statistic=lambda slm, flow: float(np.mean(slm) - np.mean(flow)),
        replicates=settings.bootstrap_replicates,
        seed=settings.bootstrap_seed,
    )
    energy_comparison = _clustered_paired_bootstrap(
        values(left, "energy_ccc"),
        values(right, "energy_ccc"),
        clusters,
        statistic=lambda slm, flow: float(np.mean(slm) - np.mean(flow)),
        replicates=settings.bootstrap_replicates,
        seed=settings.bootstrap_seed,
    )
    rate_comparison = _clustered_paired_bootstrap(
        values(left, "speaking_rate_proxy_hz"),
        values(right, "speaking_rate_proxy_hz"),
        clusters,
        statistic=lambda slm, flow: _population_standard_deviation(slm)
        - _population_standard_deviation(flow),
        replicates=settings.bootstrap_replicates,
        seed=settings.bootstrap_seed,
    )
    comparison = {
        "f0_ccc_mean_difference": _bootstrap_payload(f0_comparison),
        "energy_ccc_mean_difference": _bootstrap_payload(energy_comparison),
        "speaking_rate_proxy_population_sd_difference_hz": _bootstrap_payload(rate_comparison),
    }
    return {
        "accounting": {
            "group_emotion_pairs": len(ordered_keys),
            "actor_clusters": len(set(clusters)),
            "uncertainty_unit": "actor cluster (all nested group-emotion rows retained per draw)",
            "conditions": list(conditions),
            "complete_per_condition": {condition: len(records) for condition, records in by_condition.items()},
        },
        "settings": asdict(settings),
        "alignment": {
            "f0": "remove unvoiced frames per waveform, then interpolate the voiced trajectories to normalized rank",
            "energy": "interpolate complete log-RMS frame trajectories to normalized duration",
            "speaking_rate": "separated log-energy nuclei per active-speech second; reported as a proxy",
        },
        "conditions": condition_summary,
        "paired_comparisons": {
            "slm_driven_minus_flow_driven": comparison
        },
        "preregistered_directional_assessment": _preregistered_assessment(comparison),
    }
