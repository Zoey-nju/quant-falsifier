"""Small statistical primitives implemented on the standard library.

Only scalar normal-distribution helpers are needed, so the runtime dependency
footprint stays at ``numpy`` alone: ``statistics.NormalDist`` already ships a
high-quality normal CDF and quantile function.
"""

from __future__ import annotations

from statistics import NormalDist

_NORMAL = NormalDist()

#: Euler-Mascheroni constant used by the expected-maximum-Sharpe approximation.
EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    """Standard normal CDF, evaluated in scalar form."""
    return _NORMAL.cdf(float(x))


def norm_ppf(p: float) -> float:
    """Standard normal quantile function for ``0 < p < 1``.

    Raises
    ------
        ValueError: If ``p`` lies outside the open unit interval.
    """
    value = float(p)
    if not 0.0 < value < 1.0:
        raise ValueError(f"p must lie strictly between 0 and 1, got {p!r}")
    return _NORMAL.inv_cdf(value)


def skewness(x: object) -> float:
    """Population skewness of a numeric sequence.

    Args:
        x: One-dimensional array-like of observations.

    Returns
    -------
        The third standardized moment. Returns ``0.0`` for a degenerate series.
    """
    import numpy as np

    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    centred = arr - arr.mean()
    m2 = float(np.mean(centred**2))
    if m2 == 0.0:
        return 0.0
    m3 = float(np.mean(centred**3))
    return float(m3 / m2**1.5)


def kurtosis(x: object) -> float:
    """Population Pearson kurtosis (``3.0`` for a normal sample).

    Args:
        x: One-dimensional array-like of observations.

    Returns
    -------
        The fourth standardized moment. Returns ``3.0`` for a degenerate series.
    """
    import numpy as np

    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        return 3.0
    centred = arr - arr.mean()
    m2 = float(np.mean(centred**2))
    if m2 == 0.0:
        return 3.0
    m4 = float(np.mean(centred**4))
    return m4 / m2**2


def default_block_length(n_obs: int) -> int:
    """Rule-of-thumb mean block length for stationary bootstraps.

    Uses ``n ** (1/3)``, the order recommended by Politis & White (2004) for
    weakly dependent data, clamped to ``[1, n_obs // 2]``.

    Args:
        n_obs: Number of observations in the series.

    Returns
    -------
        A positive integer block length.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs}")
    cube_root: int = round(float(n_obs) ** (1.0 / 3.0))
    return max(1, min(n_obs // 2, cube_root))
