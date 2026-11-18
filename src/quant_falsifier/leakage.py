"""Screens for look-ahead bias and evaluation-protocol leakage.

No test can prove that a backtest is free of look-ahead. What can be checked is
the structure of the evaluation protocol, and that is where the expensive
mistakes usually live:

``PurgedKFold``
    Time-ordered cross-validation that removes training samples whose label
    window overlaps the test window (purging), plus an optional embargo after
    each test block.

``audit_splits``
    Independent audit of *any* collection of train/test index sets, including
    ones produced by a library this project does not control. It counts
    overlapping labels, train/test index collisions and embargo violations.

``check_point_in_time``
    Compares, per observation, the timestamp of the newest input with the
    timestamp of the trading decision. A feature built from data that did not
    exist yet is a violation, full stop.

``label_overlap``
    Overlapping labels (for example, overlapping ``h``-bar forward returns)
    make observations dependent, so the nominal sample size overstates the
    information content. The function reports the implied effective sample size.

``signal_delay_curve``
    Re-runs a signal delayed by one, two, ... bars. An edge that only exists at
    delay zero is either genuinely fast or computed from the bar it trades; the
    curve makes that visible instead of hiding it.

``permutation_pvalue``
    Builds a null distribution by block-shuffling the signal, preserving its
    autocorrelation while destroying its pairing with returns.

Conventions:
    Time is a numeric array (bar index or epoch seconds) sorted ascending, and
    ``returns[t]`` is the return realised *after* the decision made at ``times[t]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ._stats import default_block_length
from ._validation import FloatArray, IntArray, as_1d, require_min_length
from .metrics import sharpe_ratio

__all__ = [
    "DelayCurve",
    "LabelOverlap",
    "PermutationResult",
    "PointInTimeAudit",
    "PurgedKFold",
    "SplitAudit",
    "audit_splits",
    "check_point_in_time",
    "label_overlap",
    "permutation_pvalue",
    "signal_delay_curve",
]


@dataclass(frozen=True)
class SplitAudit:
    """Structural audit of a cross-validation protocol.

    Attributes
    ----------
        n_splits: Number of splits inspected.
        n_obs: Number of observations in the sample.
        n_label_overlaps: Training samples whose label window intersects the
            test window, summed over splits.
        n_index_collisions: Indices that appear in both the training and the
            test set of the same split, summed over splits.
        n_embargo_violations: Training samples that fall inside the embargo
            window immediately after the test block, summed over splits.
        leaking_splits: Ordinal positions of splits with at least one leak.
        examples: Up to five ``(split, train_index, test_start)`` tuples showing
            concrete overlaps.
    """

    n_splits: int
    n_obs: int
    n_label_overlaps: int
    n_index_collisions: int
    n_embargo_violations: int
    leaking_splits: tuple[int, ...]
    examples: tuple[tuple[int, int, int], ...]
    embargo: int

    @property
    def is_clean(self) -> bool:
        """Whether the protocol has no label overlap, collision or embargo violation."""
        return (
            self.n_label_overlaps == 0
            and self.n_index_collisions == 0
            and self.n_embargo_violations == 0
        )


@dataclass(frozen=True)
class PointInTimeAudit:
    """Result of a point-in-time availability check.

    Attributes
    ----------
        n_checks: Number of decisions inspected.
        n_violations: Number of decisions that used data not yet available.
        worst_violation: Largest ``feature_time + lag - decision_time`` among the
            violations, in the units of the supplied timestamps.
        violation_indices: Up to ten offending observation indices.
        publication_lag: Lag applied to feature timestamps before comparison.
        tolerance: Slack allowed before a comparison counts as a violation.
    """

    n_checks: int
    n_violations: int
    worst_violation: float
    violation_indices: tuple[int, ...]
    publication_lag: float
    tolerance: float

    @property
    def passed(self) -> bool:
        """Whether every decision only used available data."""
        return self.n_violations == 0


@dataclass(frozen=True)
class LabelOverlap:
    """Dependence induced by overlapping label windows.

    Attributes
    ----------
        n_obs: Number of observations.
        mean_cluster_size: Average number of observations that share at least one
            label window, including the observation itself.
        overlap_fraction: Share of observations whose label window contains the
            next observation's timestamp.
        effective_n: ``n_obs / mean_cluster_size``, a first-order estimate of the
            independent sample size.
    """

    n_obs: int
    mean_cluster_size: float
    overlap_fraction: float
    effective_n: float


@dataclass(frozen=True)
class DelayCurve:
    """Annualised performance of a signal as it is delayed before trading.

    Attributes
    ----------
        delays: Delay values, in bars.
        sharpes: Annualised Sharpe ratio at each delay.
        periods_per_year: Annualisation factor.
    """

    delays: IntArray
    sharpes: FloatArray
    periods_per_year: int

    def retention(self) -> FloatArray:
        """Sharpe at each delay divided by the Sharpe at delay zero.

        Values close to zero mean the edge disappears as soon as the signal is
        acted on one bar later; that deserves an explanation in any write-up.

        Returns
        -------
            An array of the same length as ``delays``. When delay zero is not
            part of the curve, or the delay-zero Sharpe ratio is zero, every
            entry is ``nan`` because no meaningful baseline exists.
        """
        zero_positions = np.flatnonzero(self.delays == 0)
        if zero_positions.size == 0:
            return np.full_like(self.sharpes, np.nan)
        base = float(self.sharpes[int(zero_positions[0])])
        if base == 0.0:
            return np.full_like(self.sharpes, np.nan)
        return np.asarray(self.sharpes / base, dtype=np.float64)


@dataclass(frozen=True)
class PermutationResult:
    """Block-permutation test of the relationship between signal and returns.

    Attributes
    ----------
        observed: Metric computed on the original pairing.
        pvalue: One-sided bootstrap-style p-value with ``(1 + exceedances) /
            (n_perm + 1)``.
        metric: ``"sharpe"`` or ``"spearman"``.
        null_mean: Mean of the permutation distribution.
        null_std: Standard deviation of the permutation distribution.
        n_perm: Number of permutations.
        mean_block: Mean block length used for shuffling, in bars.
    """

    observed: float
    pvalue: float
    metric: str
    null_mean: float
    null_std: float
    n_perm: int
    mean_block: int


class PurgedKFold:
    """Time-ordered K-fold splits with purging and an optional embargo.

    The sample is cut into ``n_splits`` contiguous test blocks. Training data are
    every other observation whose label window does not intersect the test
    window, minus an embargo of ``embargo_pct * n_obs`` observations immediately
    after each test block.

    The object follows the scikit-learn splitter protocol: pass the timestamp
    array as ``X`` to :meth:`split` and iterate over ``(train, test)`` index
    arrays.

    Args:
        n_splits: Number of folds, at least two.
        embargo_pct: Fraction of the sample to drop after each test block, in
            ``[0, 1)``.
        label_end_times: Optional end timestamp of each observation's label. When
            omitted, each label is assumed to end at the next observation.

    Raises
    ------
        ValueError: If ``n_splits < 2`` or ``embargo_pct`` is out of range.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        embargo_pct: float = 0.0,
        label_end_times: Sequence[float] | FloatArray | None = None,
    ) -> None:
        if int(n_splits) < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if not 0.0 <= float(embargo_pct) < 1.0:
            raise ValueError(f"embargo_pct must lie in [0, 1), got {embargo_pct}")
        self.n_splits = int(n_splits)
        self.embargo_pct = float(embargo_pct)
        self._label_end_times = label_end_times

    def get_n_splits(
        self,
        X: object | None = None,
        y: object | None = None,
        groups: object | None = None,
    ) -> int:
        """Return the number of splits, for scikit-learn compatibility."""
        return self.n_splits

    def split(
        self,
        X: object,
        y: object | None = None,
        groups: object | None = None,
    ) -> Iterator[tuple[IntArray, IntArray]]:
        """Yield ``(train_index, test_index)`` pairs.

        Args:
            X: One-dimensional array-like of ascending numeric timestamps.
            y: Ignored, present for API compatibility.
            groups: Ignored, present for API compatibility.

        Yields
        ------
            Integer index arrays for training and testing.

        Raises
        ------
            ValueError: If ``X`` is not sorted, or the sample is shorter than the
                number of splits.
        """
        del y, groups
        times = as_1d(X, "times")
        n_obs = times.size
        if n_obs < self.n_splits:
            raise ValueError(f"cannot build {self.n_splits} folds from {n_obs} observations")
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be sorted in ascending order")
        ends = _resolve_label_ends(times, self._label_end_times)
        embargo = math.floor(self.embargo_pct * n_obs)
        bounds = np.linspace(0, n_obs, self.n_splits + 1).astype(int)

        for fold in range(self.n_splits):
            start, stop = int(bounds[fold]), int(bounds[fold + 1])
            if stop <= start:
                continue
            test = np.arange(start, stop, dtype=np.int64)
            low, high = times[start], times[stop - 1]
            overlaps = (ends >= low) & (times <= high)
            keep = np.ones(n_obs, dtype=bool)
            keep[test] = False
            keep &= ~overlaps
            if embargo:
                keep[stop : min(n_obs, stop + embargo)] = False
            yield np.flatnonzero(keep).astype(np.int64), test

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return f"PurgedKFold(n_splits={self.n_splits}, embargo_pct={self.embargo_pct})"


def audit_splits(
    splits: Iterable[tuple[Sequence[int] | IntArray, Sequence[int] | IntArray]],
    times: object,
    *,
    label_end_times: object | None = None,
    embargo: int = 0,
) -> SplitAudit:
    """Audit an existing collection of train/test index sets for leakage.

    Args:
        splits: Iterable of ``(train_index, test_index)`` pairs.
        times: Ascending numeric timestamps, one per observation.
        label_end_times: End timestamp of each label. Defaults to the next
            observation's timestamp.
        embargo: Number of observations after each test block in which training
            samples are not allowed.

    Returns
    -------
        A :class:`SplitAudit`.

    Raises
    ------
        ValueError: If timestamps are unsorted, label ends precede their start,
            or an index falls outside the sample.
    """
    stamps = as_1d(times, "times")
    require_min_length(stamps, 2, "times")
    if np.any(np.diff(stamps) < 0):
        raise ValueError("times must be sorted in ascending order")
    ends = _resolve_label_ends(stamps, label_end_times)
    n_obs = stamps.size
    if int(embargo) < 0:
        raise ValueError(f"embargo must be non-negative, got {embargo}")

    n_splits = 0
    overlaps_total = 0
    collisions_total = 0
    embargo_total = 0
    leaking: list[int] = []
    examples: list[tuple[int, int, int]] = []

    for split_id, (raw_train, raw_test) in enumerate(splits):
        train = _as_indices(raw_train, n_obs, "train")
        test = _as_indices(raw_test, n_obs, "test")
        if test.size == 0:
            continue
        n_splits += 1
        collisions_total += int(np.intersect1d(train, test, assume_unique=False).size)
        low = np.min(stamps[test])
        high = np.max(stamps[test])
        overlapping = (ends[train] >= low) & (stamps[train] <= high)
        count = int(np.count_nonzero(overlapping))
        overlaps_total += count
        if count:
            leaking.append(split_id)
            for position in np.flatnonzero(overlapping)[: max(0, 5 - len(examples))]:
                test_start = int(test[0])
                examples.append((split_id, int(train[position]), test_start))
        if embargo:
            embargo_total += int(_count_embargo_violations(train, test, n_obs, int(embargo)))

    return SplitAudit(
        n_splits=n_splits,
        n_obs=n_obs,
        n_label_overlaps=overlaps_total,
        n_index_collisions=collisions_total,
        n_embargo_violations=embargo_total,
        leaking_splits=tuple(leaking),
        examples=tuple(examples),
        embargo=int(embargo),
    )


def check_point_in_time(
    feature_times: object,
    decision_times: object,
    *,
    publication_lag: float = 0.0,
    tolerance: float = 0.0,
) -> PointInTimeAudit:
    """Check that every decision only used data that was already available.

    Args:
        feature_times: Timestamp of the newest input used at each decision.
        decision_times: Timestamp at which each decision is made.
        publication_lag: Delay between an input's timestamp and its actual
            availability, in the units of the timestamps.
        tolerance: Slack allowed before a comparison counts as a violation.

    Returns
    -------
        A :class:`PointInTimeAudit`; ``passed`` is true when nothing leaked.

    Raises
    ------
        ValueError: If the two arrays differ in length.
    """
    features = as_1d(feature_times, "feature_times")
    decisions = as_1d(decision_times, "decision_times")
    if features.size != decisions.size:
        raise ValueError(
            f"feature_times and decision_times must have the same length, got "
            f"{features.size} and {decisions.size}"
        )
    lag = float(publication_lag)
    slack = float(tolerance)
    violations = features + lag - decisions
    flagged = violations > slack
    count = int(np.count_nonzero(flagged))
    worst = float(np.max(violations[flagged])) if count else 0.0
    indices = tuple(int(i) for i in np.flatnonzero(flagged)[:10])
    return PointInTimeAudit(
        n_checks=int(features.size),
        n_violations=count,
        worst_violation=worst,
        violation_indices=indices,
        publication_lag=lag,
        tolerance=slack,
    )


def label_overlap(sample_times: object, label_end_times: object | None = None) -> LabelOverlap:
    """Quantify dependence caused by overlapping label windows.

    Args:
        sample_times: Ascending numeric timestamps of the samples.
        label_end_times: End timestamp of each label. Defaults to the next
            observation's timestamp.

    Returns
    -------
        A :class:`LabelOverlap`.

    Raises
    ------
        ValueError: If timestamps are unsorted or a label ends before it starts.
    """
    stamps = as_1d(sample_times, "sample_times")
    require_min_length(stamps, 2, "sample_times")
    if np.any(np.diff(stamps) < 0):
        raise ValueError("sample_times must be sorted in ascending order")
    ends = _resolve_label_ends(stamps, label_end_times)
    n_obs = stamps.size
    # Number of later observations whose timestamp is strictly before this
    # label's end time (the overlapping cluster ahead of each sample).
    later_inside = np.searchsorted(stamps, ends, side="left") - np.arange(n_obs) - 1
    later_inside = np.clip(later_inside, 0, None)
    cluster = 1.0 + later_inside.astype(np.float64)
    mean_cluster = float(cluster.mean())
    return LabelOverlap(
        n_obs=n_obs,
        mean_cluster_size=mean_cluster,
        overlap_fraction=float(np.count_nonzero(later_inside) / n_obs),
        effective_n=float(n_obs / mean_cluster),
    )


def signal_delay_curve(
    signal: object,
    forward_returns: object,
    *,
    delays: Sequence[int] = (0, 1, 2, 3, 5),
    periods_per_year: int = 252,
) -> DelayCurve:
    """Re-evaluate a signal when it is acted on later than intended.

    The strategy return at ``t`` is the standardised signal at ``t - delay``
    multiplied by the forward return at ``t``.

    Args:
        signal: One-dimensional signal, aligned so that ``signal[t]`` is known at
            the decision time of ``forward_returns[t]``.
        forward_returns: One-dimensional returns realised after each decision.
        delays: Non-negative integer delays to test, in bars.
        periods_per_year: Annualisation factor for the reported Sharpe ratios.

    Returns
    -------
        A :class:`DelayCurve`.

    Raises
    ------
        ValueError: If lengths differ, a delay is negative, or the series is too
            short for the largest delay.
    """
    sig = as_1d(signal, "signal")
    fwd = as_1d(forward_returns, "forward_returns")
    if sig.size != fwd.size:
        raise ValueError(
            f"signal and forward_returns must have the same length, got {sig.size} and {fwd.size}"
        )
    delay_values = np.asarray(delays, dtype=np.int64)
    if delay_values.ndim != 1 or delay_values.size == 0:
        raise ValueError("delays must be a non-empty one-dimensional sequence")
    if np.any(delay_values < 0):
        raise ValueError("delays must be non-negative")
    max_delay = int(np.max(delay_values))
    if sig.size <= max_delay + 2:
        raise ValueError(
            f"series of length {sig.size} is too short for a maximum delay of {max_delay}"
        )
    std = float(sig.std(ddof=1))
    if std == 0.0:
        raise ValueError("signal has zero variance, so the delay curve is undefined")
    weights = (sig - sig.mean()) / std

    sharpes = []
    for delay in delay_values:
        active = weights[: sig.size - delay] * fwd[delay:]
        sharpes.append(sharpe_ratio(active, periods_per_year=periods_per_year, annualize=True))
    return DelayCurve(
        delays=delay_values,
        sharpes=np.asarray(sharpes, dtype=np.float64),
        periods_per_year=int(periods_per_year),
    )


def permutation_pvalue(
    signal: object,
    forward_returns: object,
    *,
    metric: str = "sharpe",
    n_perm: int = 2000,
    mean_block: float | None = None,
    seed: int | None = None,
) -> PermutationResult:
    """One-sided permutation test that the signal is paired with returns.

    The signal is shuffled in circular blocks, which keeps its autocorrelation
    structure while destroying any genuine alignment with returns.

    Args:
        signal: One-dimensional signal aligned to ``forward_returns``.
        forward_returns: One-dimensional returns realised after each decision.
        metric: ``"sharpe"`` for the Sharpe ratio of a signal-weighted portfolio,
            or ``"spearman"`` for rank correlation.
        n_perm: Number of permutations.
        mean_block: Mean block length; defaults to ``round(n ** (1/3))``.
        seed: Seed for the random number generator.

    Returns
    -------
        A :class:`PermutationResult`.

    Raises
    ------
        ValueError: If lengths differ, the metric is unknown, or the signal is
            degenerate.
    """
    if metric not in {"sharpe", "spearman"}:
        raise ValueError(f"metric must be 'sharpe' or 'spearman', got {metric!r}")
    if int(n_perm) < 1:
        raise ValueError(f"n_perm must be at least 1, got {n_perm}")
    sig = as_1d(signal, "signal")
    fwd = as_1d(forward_returns, "forward_returns")
    if sig.size != fwd.size:
        raise ValueError(
            f"signal and forward_returns must have the same length, got {sig.size} and {fwd.size}"
        )
    require_min_length(sig, 10, "signal")
    block = int(mean_block) if mean_block is not None else default_block_length(sig.size)
    block = max(1, block)

    evaluate = _sharpe_metric if metric == "sharpe" else _spearman_metric
    observed = evaluate(sig, fwd)

    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype=np.float64)
    for index in range(int(n_perm)):
        null[index] = evaluate(sig[block_shuffle_indices(sig.size, block, rng)], fwd)
    pvalue = float((1 + np.count_nonzero(null >= observed)) / (int(n_perm) + 1))
    return PermutationResult(
        observed=float(observed),
        pvalue=pvalue,
        metric=metric,
        null_mean=float(null.mean()),
        null_std=float(null.std(ddof=1)) if null.size > 1 else 0.0,
        n_perm=int(n_perm),
        mean_block=block,
    )


def block_shuffle_indices(n_obs: int, block: int, rng: np.random.Generator) -> IntArray:
    """Circular block permutation of ``n_obs`` indices.

    Args:
        n_obs: Number of indices.
        block: Block length in observations.
        rng: Random generator.

    Returns
    -------
        A permutation-like index array of length ``n_obs``.
    """
    block = max(1, int(block))
    n_blocks = math.ceil(n_obs / block)
    order = rng.permutation(n_blocks)
    indices = (order[:, None] * block + np.arange(block)[None, :]).ravel() % n_obs
    return np.asarray(indices[:n_obs], dtype=np.int64)


def _sharpe_metric(signal: FloatArray, forward_returns: FloatArray) -> float:
    """Per-period Sharpe ratio of a zero-mean signal-weighted portfolio."""
    std = float(signal.std(ddof=1))
    if std == 0.0:
        return 0.0
    weights = (signal - signal.mean()) / std
    return float(sharpe_ratio(weights * forward_returns, annualize=False))


def _spearman_metric(signal: FloatArray, forward_returns: FloatArray) -> float:
    """Rank correlation with ordinal tie handling."""
    a = np.argsort(np.argsort(signal, kind="stable"), kind="stable").astype(np.float64)
    b = np.argsort(np.argsort(forward_returns, kind="stable"), kind="stable").astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def _as_indices(values: Sequence[int] | IntArray, n_obs: int, name: str) -> IntArray:
    """Validate an index array against the sample size."""
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{name} indices must be one-dimensional")
    if indices.size and (np.min(indices) < 0 or np.max(indices) >= n_obs):
        raise ValueError(f"{name} indices must lie in [0, {n_obs})")
    return indices


def _count_embargo_violations(train: IntArray, test: IntArray, n_obs: int, embargo: int) -> int:
    """Count training samples inside the embargo window after each test block.

    The embargo applies to every contiguous run of test indices, not just the
    last one, so non-contiguous test sets (for example from purged K-fold) are
    audited correctly.
    """
    if embargo <= 0 or test.size == 0 or train.size == 0:
        return 0
    sorted_test = np.sort(test)
    # Contiguous runs: positions where the gap to the next index exceeds one.
    breaks = np.flatnonzero(np.diff(sorted_test) > 1)
    block_starts = np.concatenate(([sorted_test[0]], sorted_test[breaks + 1]))
    block_ends = np.concatenate((sorted_test[breaks], [sorted_test[-1]]))
    embargo_mask = np.zeros(n_obs, dtype=bool)
    for _start, end in zip(block_starts, block_ends, strict=True):
        zone_start = int(end) + 1
        zone_stop = min(n_obs, zone_start + embargo)
        if zone_start < zone_stop:
            embargo_mask[zone_start:zone_stop] = True
    return int(np.count_nonzero(embargo_mask[train]))


def _resolve_label_ends(times: FloatArray, label_end_times: object | None) -> FloatArray:
    """Return label end timestamps, validating user-supplied values."""
    if label_end_times is None:
        if times.size < 2:
            raise ValueError("at least two timestamps are required to infer label ends")
        step = float(np.median(np.diff(times)))
        ends = np.empty_like(times)
        ends[:-1] = times[1:]
        ends[-1] = times[-1] + step
        return ends
    ends = as_1d(label_end_times, "label_end_times")
    if ends.size != times.size:
        raise ValueError(
            f"label_end_times must have the same length as times, got {ends.size} and {times.size}"
        )
    if np.any(ends < times):
        raise ValueError("label_end_times must not precede the corresponding timestamps")
    return ends
