"""Tests for the synthetic data generators."""

from __future__ import annotations

import numpy as np
import pytest

from quant_falsifier import DemoData, demo_dataset, make_skill_signal, make_strategy_returns


def test_make_strategy_returns_shape_and_skill() -> None:
    returns = make_strategy_returns(n_obs=100, n_trials=5, skill=0.001, seed=0)
    assert returns.shape == (100, 5)
    # Skill is added to the first strategy only.
    assert returns[:, 0].mean() > returns[:, 1].mean()


def test_make_strategy_returns_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        make_strategy_returns(n_obs=5, n_trials=3)
    with pytest.raises(ValueError, match="n_trials"):
        make_strategy_returns(n_obs=20, n_trials=0)
    with pytest.raises(ValueError, match="common"):
        make_strategy_returns(n_obs=20, n_trials=3, common=1.5)


def test_make_strategy_returns_is_reproducible() -> None:
    a = make_strategy_returns(n_obs=50, n_trials=4, seed=123)
    b = make_strategy_returns(n_obs=50, n_trials=4, seed=123)
    assert np.array_equal(a, b)


def test_make_skill_signal_returns_demo_data() -> None:
    data = make_skill_signal(n_obs=200, seed=1)
    assert isinstance(data, DemoData)
    assert data.trial_returns.shape == (200, 1)
    assert data.forward_returns.shape == (200,)
    assert data.clean_signal.shape == (200,)
    assert data.leaky_signal.shape == (200,)


def test_make_skill_signal_leak_is_stronger_than_clean() -> None:
    data = make_skill_signal(n_obs=500, leak=2.0, seed=2)
    clean_corr = float(np.corrcoef(data.clean_signal, data.forward_returns)[0, 1])
    leaky_corr = float(np.corrcoef(data.leaky_signal, data.forward_returns)[0, 1])
    assert leaky_corr > clean_corr


def test_make_skill_signal_validates_persistence() -> None:
    with pytest.raises(ValueError, match="persistence"):
        make_skill_signal(n_obs=50, persistence=1.5)


def test_demo_dataset_shape() -> None:
    data = demo_dataset(n_obs=100, n_trials=8, seed=3)
    assert data.trial_returns.shape == (100, 8)
    assert data.forward_returns.shape == (100,)
    assert data.clean_signal.shape == (100,)
    assert data.leaky_signal.shape == (100,)


def test_demo_dataset_requires_minimum_observations() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        demo_dataset(n_obs=10, n_trials=3)


def test_demo_dataset_is_reproducible() -> None:
    a = demo_dataset(n_obs=80, n_trials=6, seed=42)
    b = demo_dataset(n_obs=80, n_trials=6, seed=42)
    assert np.array_equal(a.trial_returns, b.trial_returns)
