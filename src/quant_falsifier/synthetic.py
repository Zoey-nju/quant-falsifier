"""Synthetic data generators for demos, tests and documentation.

Real market data cannot be shipped with an open-source repository, and shipping a
"sample" extract invites exactly the kind of silent-selection problem this package
exists to catch. Everything here is generated from a seeded
:class:`numpy.random.Generator`, so a demo is reproducible from the code alone.

These are not market simulators. They produce series with the minimum structure
needed to exercise each audit: a persistent latent factor that a signal can
legitimately observe, cross-strategy correlation, tunable skill, and signals with
an explicit, controllable amount of look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import FloatArray

__all__ = ["DemoData", "demo_dataset", "make_skill_signal", "make_strategy_returns"]


@dataclass(frozen=True)
class DemoData:
    """Data prepared for an end-to-end demo.

    Attributes
    ----------
        trial_returns: ``(n_obs, n_trials)`` matrix of strategy returns.
        forward_returns: Return realised after each decision.
        clean_signal: Signal that only uses information available at decision time.
        leaky_signal: The same signal with future returns mixed in.
    """

    trial_returns: FloatArray
    forward_returns: FloatArray
    clean_signal: FloatArray
    leaky_signal: FloatArray


def make_strategy_returns(
    n_obs: int = 1200,
    n_trials: int = 24,
    *,
    skill: float = 0.0,
    common: float = 0.35,
    seed: int | None = 0,
) -> FloatArray:
    """Generate correlated strategy return series with optional skill.

    Returns are a one-factor mixture: ``common`` controls how much of each series
    is shared market variation, so strategies look correlated like real candidates
    from a parameter sweep. ``skill`` is added to the first strategy only.

    Args:
        n_obs: Number of observations per strategy.
        n_trials: Number of strategies.
        skill: Per-bar mean return added to strategy zero.
        common: Share of variance explained by the common factor, in ``[0, 1)``.
        seed: Seed for the random generator.

    Returns
    -------
        Array of shape ``(n_obs, n_trials)``.

    Raises
    ------
        ValueError: If dimensions are too small or ``common`` is out of range.
    """
    if int(n_obs) < 10:
        raise ValueError(f"n_obs must be at least 10, got {n_obs}")
    if int(n_trials) < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if not 0.0 <= float(common) < 1.0:
        raise ValueError(f"common must lie in [0, 1), got {common}")
    rng = np.random.default_rng(seed)
    n = int(n_obs)
    k = int(n_trials)
    market = rng.normal(0.0, 0.01, size=n)
    idiosyncratic = rng.normal(0.0, 0.01, size=(n, k))
    weight = float(common)
    returns = weight * market[:, None] + np.sqrt(1.0 - weight**2) * idiosyncratic
    if skill:
        returns[:, 0] += float(skill)
    return np.asarray(returns, dtype=np.float64)


def make_skill_signal(
    n_obs: int = 1200,
    *,
    persistence: float = 0.8,
    predictive: float = 0.3,
    leak: float = 0.0,
    seed: int | None = 1,
) -> DemoData:
    """Generate a signal with a genuine edge and an explicit amount of leakage.

    A persistent latent state drives next-bar returns; the signal observes the
    state at decision time, which is legitimate. ``leak`` mixes the realised
    return itself into the signal, which is not.

    Args:
        n_obs: Number of observations.
        persistence: Autocorrelation of the latent state, in ``[0, 1)``.
        predictive: Strength of the state's effect on next-bar returns.
        leak: Weight of the realised return mixed into the reported signal.
        seed: Seed for the random generator.

    Returns
    -------
        A :class:`DemoData` whose trial matrix contains a single strategy.
    """
    if not 0.0 <= float(persistence) < 1.0:
        raise ValueError(f"persistence must lie in [0, 1), got {persistence}")
    rng = np.random.default_rng(seed)
    n = int(n_obs)
    state = np.zeros(n)
    shocks = rng.normal(0.0, 1.0, size=n)
    for t in range(1, n):
        state[t] = float(persistence) * state[t - 1] + shocks[t]
    forward = 0.0002 + 0.01 * (float(predictive) * state + rng.normal(0.0, 1.0, size=n))
    clean = state / float(state.std(ddof=1))
    leaky = clean + float(leak) * (forward - forward.mean()) / float(forward.std(ddof=1))
    weights = (clean - clean.mean()) / clean.std(ddof=1)
    trial = weights * forward
    return DemoData(
        trial_returns=trial[:, None],
        forward_returns=forward,
        clean_signal=clean,
        leaky_signal=leaky,
    )


def demo_dataset(
    n_obs: int = 900,
    n_trials: int = 30,
    *,
    skill: float = 0.0013,
    seed: int | None = 7,
) -> DemoData:
    """Build a reproducible dataset that exercises every audit in the package.

    Args:
        n_obs: Number of observations.
        n_trials: Number of strategies in the search.
        skill: Per-bar mean return of the best strategy.
        seed: Seed for the random generator.

    Returns
    -------
        A :class:`DemoData`.
    """
    if int(n_obs) < 50:
        raise ValueError(f"n_obs must be at least 50, got {n_obs}")
    rng = np.random.default_rng(seed)
    n = int(n_obs)
    k = int(n_trials)
    state = np.zeros(n)
    shocks = rng.normal(0.0, 1.0, size=n)
    for t in range(1, n):
        state[t] = 0.8 * state[t - 1] + shocks[t]
    market = rng.normal(0.0, 0.01, size=n)
    idio = rng.normal(0.0, 0.01, size=(n, k))
    trials = 0.35 * market[:, None] + np.sqrt(1.0 - 0.35**2) * idio
    trials[:, :3] += 0.00018 * state[:, None]
    trials[:, 0] += float(skill)
    forward = trials[:, 0]
    clean = state / float(state.std(ddof=1))
    clean = clean + 0.3 * rng.normal(0.0, 1.0, size=n)
    clean = (clean - clean.mean()) / clean.std(ddof=1)
    leaky = clean + 4.0 * (forward - forward.mean()) / float(forward.std(ddof=1))
    return DemoData(
        trial_returns=trials,
        forward_returns=forward,
        clean_signal=clean,
        leaky_signal=leaky,
    )
