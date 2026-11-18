"""Falsification tooling for quantitative research.

``quant_falsifier`` is a small library for the part of a backtest that decides
whether a result deserves to be believed:

* a hash-chained :class:`TrialLedger` so the number of trials is a fact, not a
  memory;
* deflated Sharpe ratios and probabilistic Sharpe ratios that correct for that
  number;
* White's reality check, Hansen's SPA test and Romano-Wolf stepdown p-values
  built on a stationary bootstrap;
* leakage screens: purged cross-validation, split audits, point-in-time
  availability checks, overlapping-label accounting, signal-delay curves and
  block-permutation tests;
* Markdown and self-contained HTML reports that state what the checks do *not*
  prove.

Example:
    >>> import numpy as np
    >>> from quant_falsifier import deflated_sharpe_ratio, per_period_sharpes
    >>> rng = np.random.default_rng(0)
    >>> trials = rng.normal(0.0, 0.01, size=(500, 20))
    >>> sharpes = per_period_sharpes(trials)
    >>> result = deflated_sharpe_ratio(trials[:, 0], sharpes)
    >>> round(result.observed_sr, 2) == round(float(sharpes[0]) * 252 ** 0.5, 2)
    True
"""

from ._version import __version__
from .leakage import (
    DelayCurve,
    LabelOverlap,
    PermutationResult,
    PointInTimeAudit,
    PurgedKFold,
    SplitAudit,
    audit_splits,
    check_point_in_time,
    label_overlap,
    permutation_pvalue,
    signal_delay_curve,
)
from .ledger import (
    GENESIS_HASH,
    LedgerVerification,
    TrialLedger,
    TrialRecord,
    fingerprint_data,
)
from .metrics import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_stats,
    expected_max_sharpe,
    min_track_record_length,
    per_period_sharpes,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .multiple_testing import (
    RealityCheckResult,
    RomanoWolfResult,
    SpaResult,
    reality_check,
    romano_wolf,
    spa_test,
    stationary_bootstrap_indices,
)
from .report import (
    AuditReport,
    Section,
    Verdict,
    deflated_sharpe_section,
    delay_section,
    label_overlap_section,
    ledger_section,
    permutation_section,
    point_in_time_section,
    reality_check_section,
    romano_wolf_section,
    spa_section,
    split_audit_section,
)
from .synthetic import DemoData, demo_dataset, make_skill_signal, make_strategy_returns

__all__ = [
    "GENESIS_HASH",
    "AuditReport",
    "DeflatedSharpeResult",
    "DelayCurve",
    "DemoData",
    "LabelOverlap",
    "LedgerVerification",
    "PermutationResult",
    "PointInTimeAudit",
    "PurgedKFold",
    "RealityCheckResult",
    "RomanoWolfResult",
    "Section",
    "SpaResult",
    "SplitAudit",
    "TrialLedger",
    "TrialRecord",
    "Verdict",
    "__version__",
    "audit_splits",
    "check_point_in_time",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_stats",
    "deflated_sharpe_section",
    "delay_section",
    "demo_dataset",
    "expected_max_sharpe",
    "fingerprint_data",
    "label_overlap",
    "label_overlap_section",
    "ledger_section",
    "make_skill_signal",
    "make_strategy_returns",
    "min_track_record_length",
    "per_period_sharpes",
    "permutation_pvalue",
    "permutation_section",
    "point_in_time_section",
    "probabilistic_sharpe_ratio",
    "reality_check",
    "reality_check_section",
    "romano_wolf",
    "romano_wolf_section",
    "sharpe_ratio",
    "signal_delay_curve",
    "spa_section",
    "spa_test",
    "split_audit_section",
    "stationary_bootstrap_indices",
]
