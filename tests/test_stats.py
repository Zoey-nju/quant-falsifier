"""Tests for the statistical primitives in ``_stats``."""

from __future__ import annotations

import numpy as np
import pytest

from quant_falsifier._stats import (
    EULER_GAMMA,
    default_block_length,
    kurtosis,
    norm_cdf,
    norm_ppf,
    skewness,
)


def test_norm_cdf_is_monotone_and_bounded() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(-1.0) < 0.5 < norm_cdf(1.0)
    assert 0.0 <= norm_cdf(-10.0) < 1e-5
    assert 0.999 < norm_cdf(10.0) <= 1.0


def test_norm_ppf_inverts_norm_cdf() -> None:
    for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-12)


def test_norm_ppf_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        norm_ppf(0.0)
    with pytest.raises(ValueError):
        norm_ppf(1.0)
    with pytest.raises(ValueError):
        norm_ppf(-0.1)


def test_skewness_of_normal_sample_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 1.0, size=10_000)
    assert abs(skewness(sample)) < 0.1


def test_skewness_of_right_skewed_sample_is_positive() -> None:
    rng = np.random.default_rng(1)
    sample = rng.exponential(1.0, size=10_000)
    assert skewness(sample) > 1.0


def test_skewness_handles_degenerate_input() -> None:
    assert skewness([1.0]) == 0.0
    assert skewness([5.0, 5.0, 5.0]) == 0.0


def test_kurtosis_of_normal_sample_is_near_three() -> None:
    rng = np.random.default_rng(2)
    sample = rng.normal(0.0, 1.0, size=10_000)
    assert kurtosis(sample) == pytest.approx(3.0, abs=0.1)


def test_kurtosis_handles_degenerate_input() -> None:
    assert kurtosis([1.0]) == 3.0
    assert kurtosis([5.0, 5.0, 5.0]) == 3.0


def test_default_block_length_returns_sensible_values() -> None:
    assert default_block_length(2) == 1
    assert default_block_length(1000) >= 2
    # Clamped to at most n_obs // 2.
    assert default_block_length(4) <= 2


def test_default_block_length_rejects_small_input() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        default_block_length(1)


def test_euler_gamma_constant() -> None:
    assert pytest.approx(0.5772156649015329, abs=1e-12) == EULER_GAMMA
