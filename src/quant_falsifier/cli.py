"""Command-line interface for quant-falsifier."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ._version import __version__
from .leakage import (
    PurgedKFold,
    audit_splits,
    check_point_in_time,
    label_overlap,
    permutation_pvalue,
    signal_delay_curve,
)
from .ledger import TrialLedger, fingerprint_data
from .metrics import deflated_sharpe_ratio, per_period_sharpes
from .multiple_testing import reality_check, romano_wolf, spa_test
from .report import (
    AuditReport,
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
from .synthetic import demo_dataset

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``quant-falsifier`` command.

    Returns
    -------
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="quant-falsifier",
        description=(
            "Audit a quantitative backtest for multiple-testing inflation and "
            "evaluation-protocol leakage."
        ),
    )
    parser.add_argument("--version", action="version", version=f"quant-falsifier {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo",
        help="run every audit on synthetic data and write a report",
        description="Run every audit on synthetic data and write Markdown and HTML reports.",
    )
    demo.add_argument(
        "--out",
        type=Path,
        default=Path("demo_out"),
        help="output directory (default: demo_out)",
    )
    demo.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing trials.jsonl that was not produced by the demo",
    )
    demo.add_argument(
        "--seed",
        type=int,
        default=7,
        help="seed for the synthetic data generator (default: 7)",
    )
    demo.set_defaults(handler=_run_demo)

    ledger = subparsers.add_parser(
        "ledger",
        help="verify a trial ledger",
        description="Verify the hash chain of a trial ledger.",
    )
    ledger_subparsers = ledger.add_subparsers(dest="ledger_command", required=True)
    verify = ledger_subparsers.add_parser(
        "verify",
        help="verify the hash chain of a ledger file",
        description="Verify sequence numbers, previous-hash links and record hashes.",
    )
    verify.add_argument("path", type=Path, help="path to the JSONL ledger")
    verify.add_argument("--json", action="store_true", help="emit machine-readable output")
    verify.set_defaults(handler=_run_ledger_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
        Process exit code: ``0`` on success, ``1`` when a ledger fails verification.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _run_ledger_verify(args: argparse.Namespace) -> int:
    """Verify a ledger and report the outcome."""
    verification = TrialLedger(args.path).verify()
    if args.json:
        payload = asdict(verification)
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        sys.stdout.write(verification.describe() + "\n")
    return 0 if verification.valid else 1


def _is_demo_ledger(ledger: TrialLedger) -> bool:
    """Return whether an existing ledger looks like output of ``demo``.

    Only files whose records all use the demo's naming and note are treated as
    disposable, so a user's own ledger in the same directory is never deleted.
    """
    try:
        records = ledger.records()
    except (OSError, ValueError):
        return False
    if not records or not ledger.verify().valid:
        return False
    prefix = "variant_"
    return all(
        record.hypothesis_id.startswith(prefix)
        and record.hypothesis_id[len(prefix) :].isdigit()
        and record.params.get("note") == "synthetic parameter sweep"
        for record in records
    )


def _run_demo(args: argparse.Namespace) -> int:
    """Run every audit on synthetic data and write reports."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)
    data = demo_dataset(seed=seed)
    n_obs = int(data.trial_returns.shape[0])
    times = np.arange(1, n_obs + 1, dtype=np.float64)
    label_ends = times + 5.0

    ledger = TrialLedger(out_dir / "trials.jsonl")
    # Rebuild from scratch so the demo can be re-run deterministically, but never
    # delete a ledger that the user created for their own work.
    if ledger.path.exists() and not args.force and not _is_demo_ledger(ledger):
        raise SystemExit(
            f"{ledger.path} already exists and does not look like demo output; "
            "pass --force to overwrite it or choose another --out directory"
        )
    if ledger.path.exists():
        ledger.path.unlink()
    sharpes = per_period_sharpes(data.trial_returns)
    fingerprint = fingerprint_data(data.trial_returns)
    for trial, sharpe in enumerate(sharpes):
        ledger.append(
            f"variant_{trial:02d}",
            params={"trial": trial, "note": "synthetic parameter sweep"},
            metrics={"sharpe_per_period": float(sharpe)},
            data_fingerprint=fingerprint,
        )
    verification = ledger.verify()
    best = int(np.argmax(sharpes))
    dsr = deflated_sharpe_ratio(data.trial_returns[:, best], sharpes)

    n_boot = 800
    rc = reality_check(data.trial_returns, n_boot=n_boot, seed=seed)
    spa = spa_test(data.trial_returns, n_boot=n_boot, seed=seed)
    rw = romano_wolf(data.trial_returns, n_boot=n_boot, seed=seed)

    naive_splits = [
        (np.setdiff1d(np.arange(n_obs), test), test) for test in np.array_split(np.arange(n_obs), 5)
    ]
    naive = audit_splits(naive_splits, times, label_end_times=label_ends)
    purged_kfold = PurgedKFold(5, embargo_pct=0.01, label_end_times=label_ends)
    purged = audit_splits(purged_kfold.split(times), times, label_end_times=label_ends, embargo=9)

    clean_pit = check_point_in_time(times - 1, times)
    leaky_pit = check_point_in_time(times + 1, times)
    overlap = label_overlap(times, label_ends)
    curve = signal_delay_curve(data.clean_signal, data.forward_returns)
    permutation = permutation_pvalue(data.clean_signal, data.forward_returns, n_perm=500, seed=seed)

    report = AuditReport(
        title="Quant-falsifier demo audit",
        subject="Synthetic parameter sweep: 30 variants over 900 bars",
        metadata={"Seed": str(seed), "Data": "synthetic, no market data included"},
    )
    report.add(ledger_section(verification, records=ledger.records()))
    report.add(deflated_sharpe_section(dsr))
    report.add(reality_check_section(rc))
    report.add(spa_section(spa))
    report.add(romano_wolf_section(rw))
    report.add(
        replace(split_audit_section(purged), title="Cross-validation leakage (purged K-fold)")
    )
    report.add(
        replace(point_in_time_section(clean_pit), title="Point-in-time availability (clean signal)")
    )
    report.add(label_overlap_section(overlap))
    report.add(delay_section(curve))
    report.add(permutation_section(permutation))

    controls = AuditReport(
        title="Negative controls: protocols that must fail",
        subject="The same data audited with two deliberately broken evaluation protocols",
        metadata={
            "Seed": str(seed),
            "Purpose": "demonstrate that each screen fires when the protocol is wrong",
        },
    )
    controls.add(
        replace(split_audit_section(naive), title="Cross-validation leakage (naive K-fold)")
    )
    controls.add(
        replace(point_in_time_section(leaky_pit), title="Point-in-time availability (leaky signal)")
    )

    markdown_path = report.write(out_dir / "report.md")
    html_path = report.write(out_dir / "report.html")
    controls_markdown_path = controls.write(out_dir / "negative_controls.md")
    controls_html_path = controls.write(out_dir / "negative_controls.html")

    for name, rendered in (("Honest audit", report), ("Negative controls", controls)):
        sys.stdout.write(f"{name}: overall verdict {rendered.overall_verdict.value}\n")
        for section in rendered.sections:
            sys.stdout.write(f"  {section.verdict.value:4s}  {section.title}\n")
    sys.stdout.write("Wrote:\n")
    for path in (
        html_path,
        markdown_path,
        controls_html_path,
        controls_markdown_path,
        ledger.path,
    ):
        sys.stdout.write(f"  {path}\n")
    return 0
