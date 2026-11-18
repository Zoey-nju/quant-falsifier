"""Multiple-testing corrections for strategies discovered by search.

Three procedures are provided, in increasing order of sophistication:

``reality_check``
    White (2000). Resamples the performance differentials of every strategy and
    compares the best observed statistic with the distribution of the *best*
    statistic under the null. Answers: "is the winner better than luck?".

``spa_test``
    Hansen (2005). Studentises each strategy and recentres the bootstrap
    distribution, which fixes the reality check's tendency to be conservative
    when many strategies are poor.

``romano_wolf``
    Romano and Wolf (2005). Stepdown procedure that reports an adjusted p-value
    per strategy, so you can say *which* strategies survive rather than only
    whether at least one does.

All three rely on a stationary bootstrap (Politis and Romano, 1994) with
geometric block lengths, which preserves short-range dependence in return
series. p-values use the ``(1 + #{bootstrap exceeds observed}) / (B + 1)``
convention, so they are never exactly zero.

References
----------
    White, H. (2000). "A Reality Check for Data Snooping". *Econometrica*,
    68(5), 1097-1126.

    Hansen, P. R. (2005). "A Test for Superior Predictive Ability".
    *Journal of Business & Economic Statistics*, 23(4), 365-380.

    Romano, J. P. and Wolf, M. (2005). "Stepwise Multiple Testing as Formalized
    Data Snooping". *Econometrica*, 73(4), 1237-1282.

    Politis, D. N. and Romano, J. P. (1994). "The Stationary Bootstrap".
    *Journal of the American Statistical Association*, 89(428), 1303-1313.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._stats import default_block_length
from ._validation import FloatArray, IntArray, as_1d, as_2d, require_positive

__all__ = [
    "RealityCheckResult",
    "RomanoWolfResult",
    "SpaResult",
    "reality_check",
    "romano_wolf",
    "spa_test",
    "stationary_bootstrap_indices",
]


@dataclass(frozen=True)
class RealityCheckResult:
    """Result of White's reality check.

    Attributes
    ----------
        statistic: ``max_k sqrt(T) * mean(d_k)`` for the best strategy.
        pvalue: Bootstrap p-value for :math:`H_0`: no strategy beats the benchmark.
        best_index: Column index of the best strategy.
        means: Mean performance differential per strategy.
        n_obs: Number of return observations.
        n_trials: Number of strategies tested.
        n_boot: Number of bootstrap replications.
        mean_block: Mean block length of the stationary bootstrap, in observations.
    """

    statistic: float
    pvalue: float
    best_index: int
    means: FloatArray
    n_obs: int
    n_trials: int
    n_boot: int
    mean_block: float


@dataclass(frozen=True)
class SpaResult:
    """Result of Hansen's superior predictive ability test.

    Attributes
    ----------
        statistic: Studentised maximum performance ``max_k sqrt(T) * mean(d_k)/sd_k``.
        pvalue: Bootstrap p-value for :math:`H_0`: no strategy beats the benchmark.
        best_index: Column index of the strategy with the largest studentised mean.
        studentised_means: Studentised mean differential per strategy.
        standard_errors: Standard deviation of the differential per strategy.
        n_obs: Number of return observations.
        n_trials: Number of strategies tested.
        n_boot: Number of bootstrap replications.
        mean_block: Mean block length of the stationary bootstrap, in observations.
    """

    statistic: float
    pvalue: float
    best_index: int
    studentised_means: FloatArray
    standard_errors: FloatArray
    n_obs: int
    n_trials: int
    n_boot: int
    mean_block: float


@dataclass(frozen=True)
class RomanoWolfResult:
    """Result of the Romano-Wolf stepdown procedure.

    Attributes
    ----------
        adjusted_pvalues: Adjusted p-value per strategy, aligned to column order.
        order: Column indices sorted from most to least significant.
        n_obs: Number of return observations.
        n_trials: Number of strategies tested.
        n_boot: Number of bootstrap replications.
        mean_block: Mean block length of the stationary bootstrap, in observations.
        alpha: Confidence level used by :meth:`significant_indices`.
    """

    adjusted_pvalues: FloatArray
    order: IntArray
    n_obs: int
    n_trials: int
    n_boot: int
    mean_block: float
    alpha: float

    def significant_indices(self) -> list[int]:
        """Return column indices whose family-wise adjusted p-value is at most ``alpha``."""
        return [int(i) for i in np.flatnonzero(self.adjusted_pvalues <= self.alpha)]


def stationary_bootstrap_indices(
    n_obs: int,
    n_boot: int,
    *,
    mean_block: float | None = None,
    seed: int | None = None,
) -> IntArray:
    """Draw stationary-bootstrap index paths of shape ``(n_boot, n_obs)``.

    Each path is a resample of observation indices where new blocks start with
    probability ``1 / mean_block`` at every step. Indices wrap around, so block
    lengths are geometric with mean ``mean_block``.

    Args:
        n_obs: Length of the original series.
        n_boot: Number of bootstrap replications.
        mean_block: Mean block length. Defaults to ``round(n_obs ** (1/3))``.
        seed: Seed for :class:`numpy.random.default_rng`.

    Returns
    -------
        Integer array of shape ``(n_boot, n_obs)`` with values in
        ``[0, n_obs)``.

    Raises
    ------
        ValueError: If ``n_obs < 2``, ``n_boot < 1`` or ``mean_block`` is not
            positive.
    """
    n, b, block = _validate_bootstrap_args(n_obs, n_boot, mean_block)
    rng = np.random.default_rng(seed)
    return _draw_stationary_paths(rng, n, b, block)


# Target number of integers per chunk when bootstrapping large samples. Keeping
# the working set near 2 million elements caps peak memory at roughly 100 MB
# regardless of ``n_obs``; without chunking the intermediate arrays grow with
# ``n_boot * n_obs`` and a million-row series needs tens of gigabytes.
_BOOTSTRAP_CHUNK_ELEMENTS = 2_000_000


def _validate_bootstrap_args(
    n_obs: int, n_boot: int, mean_block: float | None
) -> tuple[int, int, float]:
    """Validate bootstrap arguments and return ``(n_obs, n_boot, block)``."""
    if int(n_obs) < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs}")
    if int(n_boot) < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    block = float(mean_block) if mean_block is not None else float(default_block_length(int(n_obs)))
    require_positive(block, "mean_block")
    return int(n_obs), int(n_boot), block


def _draw_stationary_paths(rng: np.random.Generator, n: int, b: int, block: float) -> IntArray:
    """Draw ``b`` stationary-bootstrap paths of length ``n`` from ``rng``."""
    restart = rng.random((b, n)) < (1.0 / block)
    restart[:, 0] = True
    positions = np.arange(n, dtype=np.int64)
    owner = np.maximum.accumulate(np.where(restart, positions[None, :], -1), axis=1)
    starts = rng.integers(0, n, size=(b, n), dtype=np.int64)
    block_start = np.asarray(np.take_along_axis(starts, owner, axis=1), dtype=np.int64)
    offset = positions[None, :] - owner
    return np.asarray((block_start + offset) % n, dtype=np.int64)


def _bootstrap_chunks(
    n_obs: int, n_boot: int, block: float, seed: int | None
) -> Iterator[tuple[int, int, IntArray]]:
    """Yield ``(start, stop, indices)`` chunks of bootstrap index paths.

    The index matrices are produced in chunks so that peak memory stays bounded
    even when ``n_obs`` is large, while the concatenated result is identical in
    distribution to :func:`stationary_bootstrap_indices`.
    """
    n, b, block = _validate_bootstrap_args(n_obs, n_boot, block)
    rng = np.random.default_rng(seed)
    chunk = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // max(1, n))
    start = 0
    while start < b:
        stop = min(b, start + chunk)
        yield start, stop, _draw_stationary_paths(rng, n, stop - start, block)
        start = stop


def reality_check(
    returns: npt.ArrayLike,
    *,
    benchmark: npt.ArrayLike | None = None,
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int | None = None,
) -> RealityCheckResult:
    """White's reality check for data snooping.

    Args:
        returns: Two-dimensional array-like of shape ``(n_obs, n_trials)``. A
            one-dimensional input is treated as a single strategy. Higher values
            must mean better performance, so pass returns, not losses.
        benchmark: Optional one-dimensional benchmark series of length
            ``n_obs``. Defaults to a flat zero benchmark (cash).
        n_boot: Number of bootstrap replications.
        mean_block: Mean block length of the stationary bootstrap.
        seed: Seed for the random number generator.

    Returns
    -------
        A :class:`RealityCheckResult`.

    Raises
    ------
        ValueError: If the input is non-finite, too short, or a benchmark has the
            wrong length.
    """
    differentials, n_obs, n_trials = _prepare(returns, benchmark)
    block = _resolve_block(mean_block, n_obs)
    means = differentials.mean(axis=0)

    root_t = math.sqrt(n_obs)
    statistic = float(np.max(root_t * means))
    n_boot = int(n_boot)
    boot_statistics = np.empty(n_boot, dtype=np.float64)
    for start, stop, indices in _bootstrap_chunks(n_obs, n_boot, block, seed):
        chunk_means = _bootstrap_means(differentials, indices)
        boot_statistics[start:stop] = np.max(root_t * (chunk_means - means[None, :]), axis=1)
    pvalue = float((1 + np.count_nonzero(boot_statistics >= statistic)) / (n_boot + 1))
    return RealityCheckResult(
        statistic=statistic,
        pvalue=pvalue,
        best_index=int(np.argmax(means)),
        means=means,
        n_obs=n_obs,
        n_trials=n_trials,
        n_boot=n_boot,
        mean_block=block,
    )


def spa_test(
    returns: npt.ArrayLike,
    *,
    benchmark: npt.ArrayLike | None = None,
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int | None = None,
) -> SpaResult:
    """Hansen's test for superior predictive ability.

    Strategies whose studentised mean falls below the ``-sqrt(2 log log T)``
    threshold are recentred at zero, following Hansen (2005), which keeps the
    test from becoming overly conservative when the candidate set contains many
    clearly bad strategies.

    Args:
        returns: Two-dimensional array-like of shape ``(n_obs, n_trials)``.
        benchmark: Optional benchmark series of length ``n_obs``.
        n_boot: Number of bootstrap replications.
        mean_block: Mean block length of the stationary bootstrap.
        seed: Seed for the random number generator.

    Returns
    -------
        A :class:`SpaResult`.

    Raises
    ------
        ValueError: If any strategy has zero variance in its performance
            differential, which would make the studentised statistic undefined.
    """
    differentials, n_obs, n_trials = _prepare(returns, benchmark)
    block = _resolve_block(mean_block, n_obs)
    standard_errors = differentials.std(axis=0, ddof=1)
    if np.any(standard_errors <= 0.0):
        degenerate = np.flatnonzero(standard_errors <= 0.0).tolist()
        raise ValueError(
            f"strategies {degenerate} have zero variance in their performance differential, "
            "so the studentised statistic is undefined"
        )
    means = differentials.mean(axis=0)

    root_t = math.sqrt(n_obs)
    studentised = root_t * means / standard_errors
    statistic = float(np.max(studentised))
    cutoff = -math.sqrt(2.0 * math.log(math.log(n_obs))) if n_obs > 3 else float("-inf")
    recentring = np.where(studentised >= cutoff, means, 0.0)
    n_boot = int(n_boot)
    boot_statistics = np.empty(n_boot, dtype=np.float64)
    for start, stop, indices in _bootstrap_chunks(n_obs, n_boot, block, seed):
        chunk_means = _bootstrap_means(differentials, indices)
        chunk_stats = np.max(
            root_t * (chunk_means - recentring[None, :]) / standard_errors[None, :], axis=1
        )
        boot_statistics[start:stop] = np.maximum(chunk_stats, 0.0)
    pvalue = float((1 + np.count_nonzero(boot_statistics >= statistic)) / (n_boot + 1))
    return SpaResult(
        statistic=statistic,
        pvalue=pvalue,
        best_index=int(np.argmax(studentised)),
        studentised_means=np.asarray(studentised, dtype=np.float64),
        standard_errors=np.asarray(standard_errors, dtype=np.float64),
        n_obs=n_obs,
        n_trials=n_trials,
        n_boot=n_boot,
        mean_block=block,
    )


def romano_wolf(
    returns: npt.ArrayLike,
    *,
    benchmark: npt.ArrayLike | None = None,
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
) -> RomanoWolfResult:
    """Romano-Wolf stepdown adjusted p-values, one per strategy.

    Args:
        returns: Two-dimensional array-like of shape ``(n_obs, n_trials)``.
        benchmark: Optional benchmark series of length ``n_obs``.
        n_boot: Number of bootstrap replications.
        mean_block: Mean block length of the stationary bootstrap.
        seed: Seed for the random number generator.
        alpha: Family-wise significance level stored on the result and used by
            :meth:`RomanoWolfResult.significant_indices`.

    Returns
    -------
        A :class:`RomanoWolfResult`.

    Raises
    ------
        ValueError: If the input is degenerate or ``alpha`` is outside ``(0, 1)``.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")
    differentials, n_obs, n_trials = _prepare(returns, benchmark)
    block = _resolve_block(mean_block, n_obs)
    standard_errors = differentials.std(axis=0, ddof=1)
    if np.any(standard_errors <= 0.0):
        degenerate = np.flatnonzero(standard_errors <= 0.0).tolist()
        raise ValueError(
            f"strategies {degenerate} have zero variance in their performance differential"
        )
    n_boot = int(n_boot)
    means = differentials.mean(axis=0)
    boot_means = np.empty((n_boot, n_trials), dtype=np.float64)
    for start, stop, indices in _bootstrap_chunks(n_obs, n_boot, block, seed):
        boot_means[start:stop] = _bootstrap_means(differentials, indices)

    root_t = math.sqrt(n_obs)
    studentised = root_t * means / standard_errors
    order = np.argsort(-studentised, kind="stable")
    adjusted = np.empty(n_trials, dtype=np.float64)
    previous = 0.0
    for position, column in enumerate(order):
        remaining = order[position:]
        boot_statistics = np.max(
            root_t
            * (boot_means[:, remaining] - means[None, remaining])
            / standard_errors[None, remaining],
            axis=1,
        )
        pvalue = float(
            (1 + np.count_nonzero(boot_statistics >= studentised[column])) / (n_boot + 1)
        )
        previous = max(previous, pvalue)
        adjusted[column] = previous

    return RomanoWolfResult(
        adjusted_pvalues=adjusted,
        order=np.asarray(order, dtype=np.int64),
        n_obs=n_obs,
        n_trials=n_trials,
        n_boot=int(n_boot),
        mean_block=block,
        alpha=float(alpha),
    )


def _prepare(
    returns: npt.ArrayLike, benchmark: npt.ArrayLike | None
) -> tuple[FloatArray, int, int]:
    """Validate inputs and return ``(differentials, n_obs, n_trials)``."""
    matrix = as_2d(returns, "returns")
    n_obs, n_trials = matrix.shape
    # The stationary bootstrap and the log-log cutoff in spa_test need a
    # non-trivial series length; 10 is a conservative floor that keeps the
    # default block length (n ** (1/3)) at or above 2.
    if n_obs < 10:
        raise ValueError(f"returns needs at least 10 observations, got {n_obs}")
    if benchmark is None:
        return matrix, n_obs, n_trials
    bench = as_1d(benchmark, "benchmark")
    if bench.size != n_obs:
        raise ValueError(
            f"benchmark must have the same length as returns ({n_obs}), got {bench.size}"
        )
    return matrix - bench[:, None], n_obs, n_trials


def _resolve_block(mean_block: float | None, n_obs: int) -> float:
    """Resolve the bootstrap block length, defaulting to the rule of thumb."""
    if mean_block is None:
        return float(default_block_length(n_obs))
    return require_positive(float(mean_block), "mean_block")


def _bootstrap_means(differentials: FloatArray, indices: IntArray) -> FloatArray:
    """Bootstrap means of every strategy, shape ``(n_boot, n_trials)``."""
    n_boot = indices.shape[0]
    n_trials = differentials.shape[1]
    means = np.empty((n_boot, n_trials), dtype=np.float64)
    for column in range(n_trials):
        means[:, column] = differentials[indices, column].mean(axis=1)
    return means
