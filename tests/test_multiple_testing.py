"""Tests for bootstrap-based multiple-testing corrections."""

from __future__ import annotations

import numpy as np
import pytest

from quant_falsifier import (
    reality_check,
    romano_wolf,
    spa_test,
    stationary_bootstrap_indices,
)


def test_bootstrap_paths_have_expected_shape_and_range() -> None:
    indices = stationary_bootstrap_indices(120, 40, mean_block=6, seed=0)
    assert indices.shape == (40, 120)
    assert indices.min() >= 0
    assert indices.max() < 120
    # A new block starts with probability 1/6, so most adjacent indices stay consecutive.
    continuation = np.mean(np.diff(indices, axis=1) == 1)
    assert continuation > 0.6


def test_bootstrap_chunks_cover_every_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_falsifier import multiple_testing as mt

    monkeypatch.setattr(mt, "_BOOTSTRAP_CHUNK_ELEMENTS", 500)
    chunks = list(mt._bootstrap_chunks(100, 40, 6.0, 0))
    assert len(chunks) > 1
    assert sum(stop - start for start, stop, _ in chunks) == 40
    assert all(indices.shape == (stop - start, 100) for start, stop, indices in chunks)
    assert all(indices.min() >= 0 and indices.max() < 100 for _, _, indices in chunks)


def test_chunked_result_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=(200, 4))
    first = reality_check(returns, n_boot=300, seed=5)
    second = reality_check(returns, n_boot=300, seed=5)
    assert first.pvalue == second.pvalue


def test_bootstrap_with_short_blocks_is_nearly_iid() -> None:
    indices = stationary_bootstrap_indices(500, 10, mean_block=1, seed=1)
    continuation = np.mean(np.diff(indices, axis=1) == 1)
    assert continuation < 0.1


def test_bootstrap_validates_arguments() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        stationary_bootstrap_indices(1, 10)
    with pytest.raises(ValueError, match="n_boot"):
        stationary_bootstrap_indices(10, 0)
    with pytest.raises(ValueError, match="mean_block"):
        stationary_bootstrap_indices(10, 5, mean_block=0)


def test_reality_check_does_not_manufacture_significance() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, size=(400, 20))
    result = reality_check(returns, n_boot=400, seed=2)
    assert result.pvalue > 0.05
    assert 0.0 <= result.pvalue <= 1.0


def test_reality_check_finds_a_genuine_edge() -> None:
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0, 0.01, size=(400, 10))
    returns[:, 3] += 0.006
    result = reality_check(returns, n_boot=400, seed=2)
    assert result.best_index == 3
    assert result.pvalue < 0.05


def test_reality_check_benchmark_is_subtracted() -> None:
    rng = np.random.default_rng(8)
    column = rng.normal(0.0, 0.01, size=300)
    returns = np.repeat(column[:, None], 3, axis=1)
    result = reality_check(returns, benchmark=column, n_boot=100, seed=1)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.pvalue > 0.5


def test_spa_detects_edge_and_rejects_noise() -> None:
    rng = np.random.default_rng(12)
    noise = rng.normal(0.0, 0.01, size=(400, 15))
    noise_result = spa_test(noise, n_boot=400, seed=3)
    assert noise_result.pvalue > 0.01

    signal = noise.copy()
    signal[:, 7] += 0.008
    result = spa_test(signal, n_boot=400, seed=3)
    assert result.best_index == 7
    assert result.pvalue < 0.01
    assert result.pvalue < noise_result.pvalue


def test_spa_rejects_zero_variance_differential() -> None:
    returns = np.column_stack([np.linspace(-0.01, 0.01, 50), np.zeros(50)])
    with pytest.raises(ValueError, match="zero variance"):
        spa_test(returns, n_boot=50)


def test_romano_wolf_isolates_the_real_strategies() -> None:
    rng = np.random.default_rng(21)
    returns = rng.normal(0.0, 0.01, size=(500, 8))
    returns[:, 2] += 0.007
    returns[:, 5] += 0.006
    result = romano_wolf(returns, n_boot=400, seed=4)
    assert set(result.significant_indices()) == {2, 5}
    assert result.adjusted_pvalues.shape == (8,)
    assert np.all(result.adjusted_pvalues >= 0.0)
    assert np.all(result.adjusted_pvalues <= 1.0)


def test_romano_wolf_adjusted_pvalues_are_monotone_in_the_ordering() -> None:
    rng = np.random.default_rng(22)
    returns = rng.normal(0.0, 0.01, size=(300, 6))
    result = romano_wolf(returns, n_boot=300, seed=5)
    ordered = result.adjusted_pvalues[result.order]
    assert np.all(np.diff(ordered) >= -1e-12)


def test_romano_wolf_validates_alpha() -> None:
    rng = np.random.default_rng(23)
    returns = rng.normal(0.0, 0.01, size=(60, 2))
    with pytest.raises(ValueError, match="alpha"):
        romano_wolf(returns, n_boot=50, alpha=1.5)


def test_multiple_testing_requires_enough_observations() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        reality_check(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="same length"):
        reality_check(np.zeros((20, 2)), benchmark=np.zeros(10))
