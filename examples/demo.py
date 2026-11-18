"""Audit a synthetic parameter sweep without using the command line.

Run with::

    python examples/demo.py

The script builds 30 variants of a strategy on synthetic data, records every trial in a
hash-chained ledger, deflates the best Sharpe ratio, tests the whole family for data snooping,
audits a purged cross-validation split, and prints a Markdown report. It never touches market
data and writes nothing to disk unless you pass an output directory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from quant_falsifier import (
    AuditReport,
    PurgedKFold,
    TrialLedger,
    audit_splits,
    deflated_sharpe_ratio,
    deflated_sharpe_section,
    demo_dataset,
    per_period_sharpes,
    reality_check,
    reality_check_section,
    split_audit_section,
)


def main(out_dir: Path | None = None) -> AuditReport:
    """Run the audit and return the report.

    Args:
        out_dir: Optional directory. When given, the report and ledger are written there.

    Returns
    -------
        The in-memory report.
    """
    data = demo_dataset(seed=7)
    n_obs = data.trial_returns.shape[0]
    sharpes = per_period_sharpes(data.trial_returns)

    with tempfile.TemporaryDirectory() as scratch:
        ledger = TrialLedger((out_dir or Path(scratch)) / "example_trials.jsonl")
        if out_dir is not None and ledger.path.exists():
            ledger.path.unlink()
        for trial, sharpe in enumerate(sharpes):
            ledger.append(f"variant_{trial:02d}", metrics={"sharpe_per_period": float(sharpe)})
        trial_sharpes = ledger.sharpes()

    best = int(np.argmax(sharpes))
    dsr = deflated_sharpe_ratio(data.trial_returns[:, best], trial_sharpes)
    rc = reality_check(data.trial_returns, n_boot=800, seed=7)

    times = np.arange(1, n_obs + 1, dtype=np.float64)
    label_ends = times + 5.0
    kfold = PurgedKFold(5, embargo_pct=0.01, label_end_times=label_ends)
    splits = audit_splits(kfold.split(times), times, label_end_times=label_ends, embargo=9)

    report = AuditReport(title="Example audit", subject="30 synthetic variants, 900 bars")
    report.add(deflated_sharpe_section(dsr))
    report.add(reality_check_section(rc))
    report.add(split_audit_section(splits))
    return report


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rendered = main(output)
    print(rendered.to_markdown())
    if output is not None:
        print(f"written to {rendered.write(output / 'example_report.md')}")
