"""Performance statistics that account for how many hypotheses were tested.

The central claim of this module is simple: a Sharpe ratio is meaningless without
the number of trials that produced it. A researcher who tries 200 variants and
reports the best one is running a maximum over 200 noisy statistics, and the
expected value of that maximum is strictly positive even when every variant is
worthless.

All Sharpe ratios are computed on the native sampling frequency of ``returns``.
Annualisation is opt-in at the reporting layer so that the correction formulas,
which are defined in per-observation units, are never accidentally applied to
annualised numbers.

References
----------
    Bailey, D. H. and Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
    Frontier". *Journal of Risk*, 15(2), 3-44.

    Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality".
    *Journal of Portfolio Management*, 40(5), 94-107.

    Harvey, C. R. and Liu, Y. (2015). "Backtesting". *Journal of Portfolio
    Management*, 42(1), 13-28.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from ._stats import EULER_GAMMA, norm_cdf, norm_ppf
from ._stats import kurtosis as _kurtosis
from ._stats import skewness as _skewness
from ._validation import FloatArray, as_1d, as_2d, require_min_length

__all__ = [
    "DeflatedSharpeResult",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_stats",
    "expected_max_sharpe",
    "min_track_record_length",
    "per_period_sharpes",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
]


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Outcome of a deflated Sharpe ratio test.

    Attributes
    ----------
        dsr: Probability that the true Sharpe ratio exceeds the expected maximum
            of ``n_trials`` worthless strategies, i.e. the deflated Sharpe ratio.
            A strategy should be rejected when this is below roughly ``0.95``.
        psr_vs_zero: Probabilistic Sharpe ratio against a benchmark of zero, the
            uncorrected significance level.
        observed_sr: Annualised Sharpe ratio of the selected strategy.
        expected_max_sr: Annualised expected maximum Sharpe ratio under the null
            hypothesis of zero skill, given the observed dispersion across trials.
        trial_sr_std: Annualised standard deviation of Sharpe ratios across trials.
        n_obs: Number of return observations.
        n_trials: Number of trials used in the correction.
        skew: Skewness of the selected strategy's returns.
        kurtosis: Pearson kurtosis of the selected strategy's returns.
        periods_per_year: Annualisation factor used for the annualised fields.
    """

    dsr: float
    psr_vs_zero: float
    observed_sr: float
    expected_max_sr: float
    trial_sr_std: float
    n_obs: int
    n_trials: int
    skew: float
    kurtosis: float
    periods_per_year: int

    def as_dict(self) -> dict[str, float | int]:
        """Return the result as a JSON-serialisable dictionary."""
        return asdict(self)


def sharpe_ratio(
    returns: npt.ArrayLike,
    *,
    periods_per_year: int = 252,
    annualize: bool = True,
    risk_free: float = 0.0,
) -> float:
    """Compute the Sharpe ratio of a return series.

    Args:
        returns: One-dimensional array-like of per-period simple or log returns.
        periods_per_year: Number of return periods in a year (``252`` for daily
            bars, ``12`` for monthly bars). Only used when ``annualize`` is true.
        annualize: Multiply the per-period ratio by ``sqrt(periods_per_year)``.
        risk_free: Annualised risk-free rate, converted to a per-period rate
            before being subtracted from each observation.

    Returns
    -------
        The Sharpe ratio.

    Raises
    ------
        ValueError: If the series is empty, non-finite, shorter than two
            observations, or has zero variance.
    """
    arr = as_1d(returns, "returns")
    require_min_length(arr, 2, "returns")
    per_period_rf = float(risk_free) / _validate_periods(periods_per_year)
    excess = arr - per_period_rf
    std = float(excess.std(ddof=1))
    if std == 0.0:
        raise ValueError("returns have zero variance, so the Sharpe ratio is undefined")
    ratio = float(excess.mean()) / std
    return ratio * math.sqrt(periods_per_year) if annualize else ratio


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """Probabilistic Sharpe ratio of Bailey and Lopez de Prado (2012).

    Answers: given the sampling variability of a Sharpe ratio estimated from
    ``n_obs`` observations with the supplied higher moments, what is the chance
    that the true Sharpe ratio exceeds ``benchmark_sr``?

    Args:
        sr: Observed Sharpe ratio on the native sampling frequency.
        n_obs: Number of return observations used to estimate ``sr``.
        skew: Skewness of the return series (``0`` for a symmetric sample).
        kurtosis: Pearson kurtosis of the return series (``3`` for a normal sample).
        benchmark_sr: Per-period Sharpe ratio under the null hypothesis.

    Returns
    -------
        A probability in ``(0, 1)``.

    Raises
    ------
        ValueError: If ``n_obs < 2`` or the variance adjustment is non-positive.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs}")
    variance_term = 1.0 - float(skew) * float(sr) + ((float(kurtosis) - 1.0) / 4.0) * float(sr) ** 2
    if variance_term <= 0.0:
        raise ValueError(
            "the higher-moment adjustment is non-positive; check that kurtosis >= 1 "
            "and that the supplied Sharpe ratio is in per-period units"
        )
    statistic = (float(sr) - float(benchmark_sr)) * math.sqrt(n_obs - 1) / math.sqrt(variance_term)
    return norm_cdf(statistic)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    r"""Expected maximum Sharpe ratio of ``n_trials`` independent worthless strategies.

    Uses the extreme-value approximation of Bailey and Lopez de Prado (2014):

    .. math::

        E[\\max SR] \\approx \\sqrt{V[SR]}\\left((1-\\gamma)\\Phi^{-1}\\!\\left(1-\\tfrac1N\\right)
        + \\gamma\\,\\Phi^{-1}\\!\\left(1-\\tfrac{1}{Ne}\\right)\\right)

    where :math:`\\gamma \\approx 0.5772` is the Euler-Mascheroni constant,
    :math:`N` is the number of trials and :math:`V[SR]` the variance of Sharpe
    ratios across those trials.

    Args:
        n_trials: Number of independent strategy variants that were evaluated.
        sr_std: Standard deviation of Sharpe ratios across trials, in the same
            per-period units as the ratio being deflated.

    Returns
    -------
        The expected maximum per-period Sharpe ratio under the null.

    Raises
    ------
        ValueError: If ``n_trials < 2`` or ``sr_std <= 0``.
    """
    if n_trials < 2:
        raise ValueError(
            "at least two trials are required to estimate the dispersion of Sharpe "
            "ratios; with a single trial there is nothing to deflate against"
        )
    std = float(sr_std)
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(f"sr_std must be a finite positive number, got {sr_std!r}")
    upper = norm_ppf(1.0 - 1.0 / n_trials)
    tail = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return std * ((1.0 - EULER_GAMMA) * upper + EULER_GAMMA * tail)


def deflated_sharpe_ratio_from_stats(
    sr: float,
    n_obs: int,
    *,
    skew: float,
    kurtosis: float,
    trial_sharpes: npt.ArrayLike,
    periods_per_year: int = 252,
) -> DeflatedSharpeResult:
    """Deflated Sharpe ratio from summary statistics rather than a return series.

    Args:
        sr: Observed per-period Sharpe ratio of the selected strategy.
        n_obs: Number of return observations behind ``sr``.
        skew: Skewness of the selected strategy's returns.
        kurtosis: Pearson kurtosis of the selected strategy's returns.
        trial_sharpes: Per-period Sharpe ratios of every trial that was evaluated,
            including the selected one. At least two distinct values are required.
        periods_per_year: Annualisation factor used for the reported annualised
            fields (``252`` for daily bars, ``12`` for monthly bars).

    Returns
    -------
        A :class:`DeflatedSharpeResult`; the annualised fields use
        ``periods_per_year``.

    Raises
    ------
        ValueError: If fewer than two trials are supplied, their dispersion is
            zero, or ``periods_per_year`` is not positive.
    """
    periods = _validate_periods(periods_per_year)
    trials = as_1d(trial_sharpes, "trial_sharpes")
    if trials.size < 2:
        raise ValueError(
            "at least two trials are required to estimate the dispersion of Sharpe "
            "ratios across trials"
        )
    trial_std = _trial_dispersion(trials)
    threshold = expected_max_sharpe(trials.size, trial_std)
    dsr = probabilistic_sharpe_ratio(
        sr, n_obs, skew=skew, kurtosis=kurtosis, benchmark_sr=threshold
    )
    scale = math.sqrt(periods)
    return DeflatedSharpeResult(
        dsr=dsr,
        psr_vs_zero=probabilistic_sharpe_ratio(sr, n_obs, skew=skew, kurtosis=kurtosis),
        observed_sr=float(sr) * scale,
        expected_max_sr=threshold * scale,
        trial_sr_std=trial_std * scale,
        n_obs=int(n_obs),
        n_trials=int(trials.size),
        skew=float(skew),
        kurtosis=float(kurtosis),
        periods_per_year=periods,
    )


def deflated_sharpe_ratio(
    returns: npt.ArrayLike,
    trial_sharpes: npt.ArrayLike,
    *,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> DeflatedSharpeResult:
    """Deflated Sharpe ratio for the best strategy out of many attempts.

    Args:
        returns: Return series of the *selected* strategy.
        trial_sharpes: Per-period (non-annualised) Sharpe ratios of all trials,
            including the selected one. Compute them with
            ``sharpe_ratio(r, annualize=False)`` on each trial's returns.
        periods_per_year: Annualisation factor used for the reported fields.
        risk_free: Annualised risk-free rate applied to the selected strategy.

    Returns
    -------
        A :class:`DeflatedSharpeResult`.

    Raises
    ------
        ValueError: If the return series is degenerate or the trial dispersion is
            zero.
    """
    arr = as_1d(returns, "returns")
    require_min_length(arr, 3, "returns")
    periods = _validate_periods(periods_per_year)
    per_period_rf = float(risk_free) / periods
    sr_per_period = sharpe_ratio(
        arr, periods_per_year=periods, annualize=False, risk_free=risk_free
    )

    trials = as_1d(trial_sharpes, "trial_sharpes")
    if trials.size < 2:
        raise ValueError(
            "at least two trials are required to estimate the dispersion of Sharpe "
            "ratios across trials"
        )
    trial_std = _trial_dispersion(trials)
    threshold = expected_max_sharpe(trials.size, trial_std)
    shape = (
        _skewness(arr - per_period_rf),
        _kurtosis(arr - per_period_rf),
    )
    dsr = probabilistic_sharpe_ratio(
        sr_per_period, arr.size, skew=shape[0], kurtosis=shape[1], benchmark_sr=threshold
    )
    scale = math.sqrt(periods)
    return DeflatedSharpeResult(
        dsr=dsr,
        psr_vs_zero=probabilistic_sharpe_ratio(
            sr_per_period, arr.size, skew=shape[0], kurtosis=shape[1]
        ),
        observed_sr=sr_per_period * scale,
        expected_max_sr=threshold * scale,
        trial_sr_std=trial_std * scale,
        n_obs=int(arr.size),
        n_trials=int(trials.size),
        skew=float(shape[0]),
        kurtosis=float(shape[1]),
        periods_per_year=periods,
    )


def min_track_record_length(
    sr: float,
    *,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    r"""Minimum number of observations needed to trust a Sharpe ratio.

    Implements the minimum track record length of Bailey and Lopez de Prado
    (2012):

    .. math::

        MinTRL = 1 + \\left[1 - \\gamma_3 SR + \\frac{\\gamma_4 - 1}{4} SR^2\\right]
        \\left(\\frac{\\Phi^{-1}(\\text{confidence})}{SR - SR^*}\\right)^2

    Args:
        sr: Observed per-period Sharpe ratio.
        benchmark_sr: Per-period Sharpe ratio to beat.
        skew: Skewness of the return series.
        kurtosis: Pearson kurtosis of the return series.
        confidence: Required confidence level in the open interval ``(0, 1)``.

    Returns
    -------
        The required number of observations, in the same units as ``sr``.

    Raises
    ------
        ValueError: If ``sr <= benchmark_sr`` or ``confidence`` is out of range.
    """
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence!r}")
    excess = float(sr) - float(benchmark_sr)
    if excess <= 0.0:
        raise ValueError(
            "the observed Sharpe ratio must exceed the benchmark, otherwise no track "
            "record is long enough"
        )
    variance_term = 1.0 - float(skew) * float(sr) + ((float(kurtosis) - 1.0) / 4.0) * float(sr) ** 2
    if variance_term <= 0.0:
        raise ValueError("the higher-moment adjustment is non-positive")
    return 1.0 + variance_term * (norm_ppf(float(confidence)) / excess) ** 2


def _validate_periods(periods_per_year: int) -> int:
    """Validate the annualisation factor and return it as an ``int``."""
    value = int(periods_per_year)
    if value <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year!r}")
    return value


def _trial_dispersion(trials: FloatArray) -> float:
    """Standard deviation of trial Sharpe ratios, rejecting numerical noise.

    Identical inputs can still produce a tiny non-zero standard deviation through
    floating-point summation, and a near-zero dispersion would make the expected
    maximum Sharpe ratio vanish, silently disabling the correction.
    """
    std = float(trials.std(ddof=1))
    scale = max(float(np.abs(trials).max()), 1.0)
    if not math.isfinite(std) or std <= 1e-12 * scale:
        raise ValueError(
            "all trials share the same Sharpe ratio, so the across-trial dispersion "
            "is zero and the deflated Sharpe ratio is undefined"
        )
    return std


def per_period_sharpes(trial_returns: npt.ArrayLike) -> FloatArray:
    """Per-period Sharpe ratios for every column of a trial matrix.

    This is the input :func:`deflated_sharpe_ratio` expects: the correction is
    defined on non-annualised ratios, and mixing the two units is the single most
    common way to make a deflated Sharpe ratio look better than it is.

    Args:
        trial_returns: Two-dimensional array-like of shape ``(n_obs, n_trials)``.

    Returns
    -------
        One Sharpe ratio per column.

    Raises
    ------
        ValueError: If a column has zero variance.
    """
    matrix = as_2d(trial_returns, "trial_returns")
    if matrix.shape[0] < 2:
        raise ValueError("trial_returns needs at least two observations per trial")
    stds = matrix.std(axis=0, ddof=1)
    if np.any(stds == 0.0):
        degenerate = np.flatnonzero(stds == 0.0).tolist()
        raise ValueError(f"trials {degenerate} have zero variance and no Sharpe ratio")
    return np.asarray(matrix.mean(axis=0) / stds, dtype=np.float64)
