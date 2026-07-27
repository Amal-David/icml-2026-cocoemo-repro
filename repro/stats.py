from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


class StatisticsError(ValueError):
    pass


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    replicates: int
    seed: int


def _finite_pair(left: Sequence[float], right: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise StatisticsError("paired inputs must be one-dimensional and have equal shape")
    if x.size < 2:
        raise StatisticsError("paired inputs require at least two observations")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise StatisticsError("paired inputs must contain only finite values")
    return x, y


def concordance_correlation_coefficient(left: Sequence[float], right: Sequence[float]) -> float:
    """Lin CCC using population moments, matching the descriptive paper metric."""
    x, y = _finite_pair(left, right)
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    if denominator == 0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return 2.0 * covariance / denominator


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    replicates: int = 10_000,
    seed: int = 20260728,
    confidence: float = 0.95,
) -> BootstrapResult:
    x, y = _finite_pair(left, right)
    if replicates < 100:
        raise StatisticsError("at least 100 bootstrap replicates are required")
    if not 0.0 < confidence < 1.0:
        raise StatisticsError("confidence must be between zero and one")

    differences = x - y
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, differences.size, size=differences.size)
        draws[index] = float(statistic(differences[sample]))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapResult(
        estimate=float(statistic(differences)),
        lower=float(lower),
        upper=float(upper),
        replicates=replicates,
        seed=seed,
    )
