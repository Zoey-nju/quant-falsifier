"""Tests for the leakage screens."""

from __future__ import annotations

import numpy as np
import pytest

from quant_falsifier import (
    PurgedKFold,
    audit_splits,
    check_point_in_time,
    label_overlap,
    permutation_pvalue,
    sharpe_ratio,
    signal_delay_curve,
)


def test_purged_kfold_produces_disjoint_folds() -> None:
    times = np.arange(120, dtype=np.float64)
    cv = PurgedKFold(5, embargo_pct=0.02)
    folds = list(cv.split(times))
    assert len(folds) == 5
    for train, test in folds:
        assert train.size > 0
        assert test.size > 0
        assert np.intersect1d(train, test).size == 0


def test_purged_kfold_removes_overlapping_labels() -> None:
    times = np.arange(150, dtype=np.float64)
    ends = times + 7.0
    cv = PurgedKFold(5, label_end_times=ends)
    for train, test in cv.split(times):
        assert np.all((ends[train] < times[test].min()) | (times[train] > times[test].max()))


def test_purged_kfold_validates_inputs() -> None:
    cv = PurgedKFold(5)
    with pytest.raises(ValueError, match="ascending"):
        list(cv.split(np.array([3.0, 2.0, 1.0, 4.0, 5.0, 6.0])))
    with pytest.raises(ValueError, match="folds"):
        list(cv.split(np.arange(3, dtype=np.float64)))
    with pytest.raises(ValueError, match="embargo_pct"):
        PurgedKFold(3, embargo_pct=1.5)


def test_naive_kfold_fails_the_audit_while_purged_kfold_passes() -> None:
    times = np.arange(100, dtype=np.float64)
    ends = times + 5.0
    naive = [
        (np.setdiff1d(np.arange(100), test), test) for test in np.array_split(np.arange(100), 5)
    ]
    naive_audit = audit_splits(naive, times, label_end_times=ends)
    assert not naive_audit.is_clean
    assert naive_audit.n_label_overlaps > 0
    assert naive_audit.n_index_collisions == 0

    purged = audit_splits(
        PurgedKFold(5, label_end_times=ends).split(times), times, label_end_times=ends
    )
    assert purged.is_clean


def test_audit_detects_index_collisions_and_embargo_violations() -> None:
    times = np.arange(20, dtype=np.float64)
    splits = [(np.array([0, 1, 2]), np.array([2, 3, 4]))]
    audit = audit_splits(splits, times, label_end_times=times + 1.0)
    assert audit.n_index_collisions == 1

    splits = [(np.arange(5, 20), np.arange(0, 5))]
    embargo_audit = audit_splits(splits, times, label_end_times=times + 0.5, embargo=3)
    assert embargo_audit.n_embargo_violations == 3


def test_audit_requires_sorted_times() -> None:
    with pytest.raises(ValueError, match="ascending"):
        audit_splits([(np.array([0]), np.array([1]))], np.array([2.0, 1.0]))


def test_point_in_time_detects_lookahead() -> None:
    times = np.arange(10, dtype=np.float64)
    clean = check_point_in_time(times - 1.0, times)
    assert clean.passed
    assert clean.n_violations == 0

    leaky = check_point_in_time(times + 1.0, times)
    assert not leaky.passed
    assert leaky.n_violations == 10
    assert leaky.worst_violation == pytest.approx(1.0)
    assert leaky.n_checks == 10


def test_publication_lag_and_tolerance_are_applied() -> None:
    times = np.arange(6, dtype=np.float64)
    audit = check_point_in_time(times - 1.0, times, publication_lag=2.0)
    assert audit.n_violations == 6
    assert audit.worst_violation == pytest.approx(1.0)
    tolerant = check_point_in_time(times - 1.0, times, publication_lag=2.0, tolerance=1.0)
    assert tolerant.passed


def test_point_in_time_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        check_point_in_time([1.0, 2.0], [1.0])


def test_label_overlap_quantifies_effective_sample_size() -> None:
    times = np.arange(400, dtype=np.float64)
    ends = times + 5.0
    overlap = label_overlap(times, ends)
    assert overlap.overlap_fraction > 0.9
    assert overlap.mean_cluster_size > 4.0
    assert overlap.effective_n < 0.3 * overlap.n_obs

    instant = label_overlap(times, times + 0.0)
    assert instant.overlap_fraction == 0.0
    assert instant.effective_n == pytest.approx(instant.n_obs)


def test_label_overlap_rejects_unsorted_input() -> None:
    with pytest.raises(ValueError, match="ascending"):
        label_overlap(np.array([2.0, 1.0, 3.0]))


def test_delay_curve_collapses_when_the_signal_is_foresight() -> None:
    rng = np.random.default_rng(0)
    forward = rng.normal(0.0, 0.01, size=600)
    signal = forward + 0.2 * rng.normal(0.0, 0.01, size=600)
    curve = signal_delay_curve(signal, forward, delays=(0, 1))
    assert curve.sharpes[0] > 5.0
    assert abs(curve.sharpes[1]) < 3.0
    assert curve.retention()[1] < 0.25


def test_delay_curve_keeps_a_persistent_signal() -> None:
    rng = np.random.default_rng(1)
    state = np.zeros(600)
    for t in range(1, 600):
        state[t] = 0.95 * state[t - 1] + rng.normal()
    forward = 0.01 * state + rng.normal(0.0, 0.01, size=600)
    curve = signal_delay_curve(state, forward, delays=(0, 1))
    assert curve.sharpes[1] > 0.7 * curve.sharpes[0]


def test_delay_curve_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        signal_delay_curve(np.arange(10.0), np.arange(10.0), delays=(-1,))
    with pytest.raises(ValueError, match="same length"):
        signal_delay_curve(np.arange(10.0), np.arange(9.0))
    with pytest.raises(ValueError, match="zero variance"):
        signal_delay_curve(np.ones(10), np.ones(10))


def test_permutation_test_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(3)
    forward = rng.normal(0.0, 0.01, size=500)
    predictive = forward + 0.3 * rng.normal(0.0, 0.01, size=500)
    clean = permutation_pvalue(predictive, forward, n_perm=300, seed=1)
    noise = permutation_pvalue(rng.normal(size=500), forward, n_perm=300, seed=1)
    assert clean.pvalue < 0.01
    assert noise.pvalue > 0.05
    assert clean.observed > 0.0


def test_permutation_test_rejects_bad_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        permutation_pvalue(np.arange(20.0), np.arange(20.0), metric="r2")


def test_delay_curve_uses_annualisation_factor() -> None:
    rng = np.random.default_rng(4)
    forward = rng.normal(0.0, 0.01, size=300)
    weights = (forward - forward.mean()) / forward.std(ddof=1)
    curve = signal_delay_curve(forward, forward, delays=(0,), periods_per_year=12)
    manual = sharpe_ratio(weights * forward, periods_per_year=12)
    assert curve.sharpes[0] == pytest.approx(manual)


def test_embargo_violations_in_gaps_between_non_contiguous_test_blocks() -> None:
    times = np.arange(20, dtype=np.float64)
    # Test set has two blocks: [15] and [19]. Training fills the gap [16, 18].
    train = np.concatenate([np.arange(15), np.array([16, 17, 18])])
    test = np.array([15, 19])
    audit = audit_splits([(train, test)], times, label_end_times=times + 0.5, embargo=3)
    # The embargo zone after test block [15] covers indices 16, 17, 18, all in train.
    assert audit.n_embargo_violations == 3


def test_delay_curve_retention_returns_nan_without_delay_zero() -> None:
    rng = np.random.default_rng(5)
    forward = rng.normal(0.0, 0.01, size=400)
    signal = forward + 0.1 * rng.normal(0.0, 0.01, size=400)
    curve = signal_delay_curve(signal, forward, delays=(1, 2, 3))
    retention = curve.retention()
    assert np.all(np.isnan(retention))
