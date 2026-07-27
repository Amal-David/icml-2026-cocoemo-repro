import numpy as np
import pytest

from repro.stats import StatisticsError, concordance_correlation_coefficient, paired_bootstrap


def test_ccc_is_one_for_identical_values() -> None:
    values = [1.0, 2.0, 4.0, 8.0]

    assert concordance_correlation_coefficient(values, values) == pytest.approx(1.0)


def test_ccc_penalizes_mean_shift_even_with_perfect_correlation() -> None:
    left = [1.0, 2.0, 3.0, 4.0]
    shifted = [2.0, 3.0, 4.0, 5.0]

    assert np.corrcoef(left, shifted)[0, 1] == pytest.approx(1.0)
    assert concordance_correlation_coefficient(left, shifted) == pytest.approx(5.0 / 7.0)


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    left = [5.0, 6.0, 7.0, 8.0]
    right = [1.0, 2.0, 3.0, 4.0]

    first = paired_bootstrap(left, right, replicates=500, seed=17)
    second = paired_bootstrap(left, right, replicates=500, seed=17)

    assert first == second
    assert first.estimate == pytest.approx(4.0)
    assert first.lower == pytest.approx(4.0)
    assert first.upper == pytest.approx(4.0)


def test_statistics_reject_missing_values() -> None:
    with pytest.raises(StatisticsError, match="finite"):
        concordance_correlation_coefficient([1.0, float("nan")], [1.0, 2.0])
