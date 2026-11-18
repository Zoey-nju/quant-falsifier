"""Tests for Sharpe ratio corrections."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_falsifier import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_stats,
    expected_max_sharpe,
    min_track_record_length,
    per_period_sharpes,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from quant_falsifier._stats import kurtosis, skewness


def test_sharpe_ratio_matches_manual_computation() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 500)
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)
    assert sharpe_ratio(returns, annualize=False) == pytest.approx(expected / math.sqrt(252))


def test_sharpe_ratio_subtracts_risk_free_rate() -> None:
    returns = np.full(100, 0.001)
    returns[::2] = 0.0
    risky = sharpe_ratio(returns, risk_free=0.0, annualize=False)
    cash = sharpe_ratio(returns, risk_free=0.252, annualize=False)
    assert cash < risky


def test_sharpe_ratio_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        sharpe_ratio(np.ones(10))
    with pytest.raises(ValueError, match="non-finite"):
        sharpe_ratio([0.1, np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        sharpe_ratio(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="at least 2"):
        sharpe_ratio([0.01])


def test_psr_is_one_half_at_the_benchmark() -> None:
    assert probabilistic_sharpe_ratio(0.05, 100, benchmark_sr=0.05) == pytest.approx(0.5)


def test_psr_increases_with_sample_size() -> None:
    short = probabilistic_sharpe_ratio(0.05, 100)
    long = probabilistic_sharpe_ratio(0.05, 10_000)
    assert long > short


def test_psr_penalises_negative_skew() -> None:
    symmetric = probabilistic_sharpe_ratio(0.08, 500, skew=0.0, kurtosis=3.0)
    skewed = probabilistic_sharpe_ratio(0.08, 500, skew=-1.5, kurtosis=3.0)
    assert skewed < symmetric


def test_expected_max_sharpe_grows_with_trials() -> None:
    values = [expected_max_sharpe(n, 0.05) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert 0.05 < expected_max_sharpe(100, 0.05) < 0.25


def test_expected_max_sharpe_requires_dispersion() -> None:
    with pytest.raises(ValueError, match="two trials"):
        expected_max_sharpe(1, 0.05)
    with pytest.raises(ValueError, match="positive"):
        expected_max_sharpe(10, 0.0)


def test_deflated_sharpe_is_below_raw_significance() -> None:
    rng = np.random.default_rng(4)
    trials = rng.normal(0.0, 0.01, size=(600, 25))
    sharpes = per_period_sharpes(trials)
    result = deflated_sharpe_ratio(trials[:, 0], sharpes)
    assert result.dsr < result.psr_vs_zero
    assert result.n_trials == 25
    assert result.expected_max_sr > 0.0


def test_deflated_sharpe_falls_as_the_search_widens() -> None:
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0006, 0.01, 800)
    narrow = np.linspace(-0.05, 0.05, 5)
    wide = np.linspace(-0.05, 0.05, 500)
    assert deflated_sharpe_ratio(returns, wide).dsr < deflated_sharpe_ratio(returns, narrow).dsr


def test_deflated_sharpe_rejects_single_trial_and_zero_dispersion() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, 100)
    with pytest.raises(ValueError, match="two trials"):
        deflated_sharpe_ratio(returns, [0.1])
    with pytest.raises(ValueError, match="dispersion"):
        deflated_sharpe_ratio(returns, [0.1, 0.1, 0.1])


def test_from_stats_matches_series_version() -> None:
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0008, 0.012, 750)
    sharpes = np.linspace(-0.04, 0.06, 30)
    from_series = deflated_sharpe_ratio(returns, sharpes)
    from_stats = deflated_sharpe_ratio_from_stats(
        sharpe_ratio(returns, annualize=False),
        returns.size,
        skew=skewness(returns),
        kurtosis=kurtosis(returns),
        trial_sharpes=sharpes,
    )
    assert from_stats.dsr == pytest.approx(from_series.dsr, rel=1e-9)
    assert from_stats.expected_max_sr == pytest.approx(from_series.expected_max_sr, rel=1e-9)


def test_from_stats_uses_periods_per_year() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0008, 0.012, 750)
    sharpes = np.linspace(-0.04, 0.06, 30)
    sr = sharpe_ratio(returns, annualize=False)
    daily = deflated_sharpe_ratio_from_stats(
        sr,
        returns.size,
        skew=skewness(returns),
        kurtosis=kurtosis(returns),
        trial_sharpes=sharpes,
        periods_per_year=252,
    )
    monthly = deflated_sharpe_ratio_from_stats(
        sr,
        returns.size,
        skew=skewness(returns),
        kurtosis=kurtosis(returns),
        trial_sharpes=sharpes,
        periods_per_year=12,
    )
    # DSR and PSR are scale-invariant; only annualised fields differ.
    assert monthly.dsr == pytest.approx(daily.dsr)
    assert monthly.psr_vs_zero == pytest.approx(daily.psr_vs_zero)
    scale = (12 / 252) ** 0.5
    assert monthly.observed_sr == pytest.approx(daily.observed_sr * scale)
    assert monthly.expected_max_sr == pytest.approx(daily.expected_max_sr * scale)
    assert monthly.periods_per_year == 12


def test_from_stats_rejects_non_positive_periods() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        deflated_sharpe_ratio_from_stats(
            0.1, 100, skew=0.0, kurtosis=3.0, trial_sharpes=[0.0, 0.1], periods_per_year=0
        )


def test_min_track_record_length_behaves() -> None:
    short = min_track_record_length(0.1, confidence=0.95)
    strict = min_track_record_length(0.1, confidence=0.99)
    assert strict > short > 1.0
    assert min_track_record_length(0.2) < min_track_record_length(0.1)
    with pytest.raises(ValueError, match="exceed the benchmark"):
        min_track_record_length(0.0, benchmark_sr=0.1)


def test_per_period_sharpes_rejects_zero_variance_column() -> None:
    matrix = np.column_stack([np.linspace(-1, 1, 50), np.ones(50)])
    with pytest.raises(ValueError, match="zero variance"):
        per_period_sharpes(matrix)


def test_per_period_sharpes_accepts_single_column() -> None:
    values = np.linspace(-1.0, 1.0, 40)
    assert per_period_sharpes(values).shape == (1,)
