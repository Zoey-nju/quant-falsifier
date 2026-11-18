"""Numeric and validation helpers shared across the package.

The public API deliberately does **no** silent data cleaning: non-finite values and
degenerate series raise instead of being dropped, because a validation tool that
quietly repairs its input cannot be trusted.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def as_1d(values: npt.ArrayLike, name: str) -> FloatArray:
    """Coerce ``values`` to a one-dimensional float array, rejecting bad input.

    Args:
        values: Array-like of numeric values.
        name: Argument name used in error messages.

    Returns
    -------
        A contiguous one-dimensional ``float64`` array.

    Raises
    ------
        ValueError: If the input is empty, not one-dimensional, or contains
            ``NaN``/``inf``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains non-finite values; clean or fix the data before auditing it"
        )
    return cast(FloatArray, np.ascontiguousarray(arr))


def as_2d(values: npt.ArrayLike, name: str) -> FloatArray:
    """Coerce ``values`` to a two-dimensional float array of shape ``(n_obs, n_trials)``.

    A one-dimensional input is treated as a single trial.

    Args:
        values: Array-like of numeric values.
        name: Argument name used in error messages.

    Returns
    -------
        A contiguous two-dimensional ``float64`` array.

    Raises
    ------
        ValueError: If the input is empty, more than two-dimensional, or not finite.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains non-finite values; clean or fix the data before auditing it"
        )
    return cast(FloatArray, np.ascontiguousarray(arr))


def require_min_length(arr: FloatArray, minimum: int, name: str) -> None:
    """Raise ``ValueError`` when ``arr`` has fewer than ``minimum`` observations."""
    if arr.size < minimum:
        raise ValueError(f"{name} needs at least {minimum} observations, got {arr.size}")


def require_positive(value: float, name: str) -> float:
    """Return ``value`` as a float, raising when it is not strictly positive."""
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return out
