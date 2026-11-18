"""Human-readable audit reports in Markdown and self-contained HTML.

The report is deliberately plain: tables, verdicts and numbers. Every section
carries its own verdict so a reader can see *which* check failed, and every
report ends with an explicit statement of what the checks cannot prove. A tool
that claims to certify a backtest is lying; a tool that shows its evidence is
useful.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from ._version import __version__
from .leakage import (
    DelayCurve,
    LabelOverlap,
    PermutationResult,
    PointInTimeAudit,
    SplitAudit,
)
from .ledger import LedgerVerification, TrialRecord
from .metrics import DeflatedSharpeResult
from .multiple_testing import RealityCheckResult, RomanoWolfResult, SpaResult

__all__ = [
    "AuditReport",
    "Section",
    "Verdict",
    "deflated_sharpe_section",
    "delay_section",
    "label_overlap_section",
    "ledger_section",
    "permutation_section",
    "point_in_time_section",
    "reality_check_section",
    "romano_wolf_section",
    "spa_section",
    "split_audit_section",
]


class Verdict(str, Enum):
    """Outcome of a single audit check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"

    @property
    def rank(self) -> int:
        """Severity rank used to aggregate an overall verdict."""
        return {"PASS": 0, "INFO": 0, "WARN": 1, "FAIL": 2}[self.value]

    def badge(self) -> str:
        """Return a compact textual badge."""
        return f"[{self.value}]"


@dataclass(frozen=True)
class Section:
    """One check inside an :class:`AuditReport`.

    Attributes
    ----------
        title: Section heading.
        verdict: Outcome of the check.
        summary: One or two sentences a reader can quote.
        metrics: Label/value pairs rendered as a table.
        notes: Additional caveats or next steps.
    """

    title: str
    verdict: Verdict
    summary: str
    metrics: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()


class AuditReport:
    """A collection of audit sections rendered as Markdown or HTML.

    Args:
        title: Report title.
        subject: Optional description of what is being audited, for example a
            strategy name or a file path.
        metadata: Extra header key/value pairs.
    """

    def __init__(
        self,
        *,
        title: str = "Backtest audit",
        subject: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.title = title
        self.subject = subject
        self.metadata: dict[str, str] = dict(metadata or {})
        self.sections: list[Section] = []
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add(self, section: Section) -> AuditReport:
        """Append a section and return the report for chaining."""
        self.sections.append(section)
        return self

    def extend(self, sections: Iterable[Section]) -> AuditReport:
        """Append several sections and return the report for chaining."""
        for section in sections:
            self.add(section)
        return self

    @property
    def overall_verdict(self) -> Verdict:
        """The worst verdict among the sections, or ``INFO`` when empty."""
        if not self.sections:
            return Verdict.INFO
        worst = max(self.sections, key=lambda section: section.verdict.rank)
        return worst.verdict

    def to_markdown(self) -> str:
        """Render the report as GitHub-flavoured Markdown."""
        lines: list[str] = [
            f"# {_md_text(self.title)}",
            "",
            f"- **Overall verdict:** `{self.overall_verdict.value}`",
            f"- **Generated:** {self.created_at}",
            f"- **Tool:** quant-falsifier {__version__}",
        ]
        if self.subject:
            lines.append(f"- **Subject:** {_md_text(self.subject)}")
        for key, value in self.metadata.items():
            lines.append(f"- **{_md_text(key)}:** {_md_text(value)}")
        lines.append("")

        if self.sections:
            lines.extend(["## Summary", "", "| Check | Verdict | Finding |", "| --- | --- | --- |"])
            for section in self.sections:
                lines.append(
                    f"| {_md_cell(section.title)} | `{section.verdict.value}` | "
                    f"{_md_cell(section.summary)} |"
                )
            lines.append("")

        for index, section in enumerate(self.sections, start=1):
            lines.extend(
                [
                    f"## {index}. {_md_text(section.title)}",
                    "",
                    f"**Verdict:** `{section.verdict.value}`",
                    "",
                    _md_text(section.summary),
                    "",
                ]
            )
            if section.metrics:
                lines.extend(["| Metric | Value |", "| --- | --- |"])
                for label, value in section.metrics:
                    lines.append(f"| {_md_cell(label)} | {_md_cell(value)} |")
                lines.append("")
            for note in section.notes:
                lines.append(f"- {_md_text(note)}")
            if section.notes:
                lines.append("")

        lines.extend(_limitations_markdown())
        return "\n".join(lines).rstrip() + "\n"

    def to_html(self) -> str:
        """Render the report as a single self-contained HTML document."""
        rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(section.title)}</td>"
            f'<td><span class="badge {section.verdict.value.lower()}">'
            f"{section.verdict.value}</span></td>"
            f"<td>{html.escape(section.summary)}</td>"
            "</tr>"
            for section in self.sections
        )
        blocks: list[str] = []
        for index, section in enumerate(self.sections, start=1):
            metric_rows = "\n".join(
                f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
                for label, value in section.metrics
            )
            notes = "".join(f"<li>{html.escape(note)}</li>" for note in section.notes)
            blocks.append(
                f"<section>\n<h2>{index}. {html.escape(section.title)}</h2>\n"
                f'<p><span class="badge {section.verdict.value.lower()}">'
                f"{section.verdict.value}</span></p>\n"
                f"<p>{html.escape(section.summary)}</p>\n"
                + (f'<table class="metrics">{metric_rows}</table>\n' if metric_rows else "")
                + (f'<ul class="notes">{notes}</ul>\n' if notes else "")
                + "</section>"
            )
        meta_items = "".join(
            f"<li><strong>{html.escape(key)}:</strong> {html.escape(value)}</li>"
            for key, value in self._html_meta().items()
        )
        limitations = "".join(f"<li>{html.escape(item)}</li>" for item in _LIMITATIONS)

        return "\n".join(
            [
                "<!DOCTYPE html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f"<title>{html.escape(self.title)}</title>",
                f"<style>{_CSS}</style>",
                "</head>",
                "<body>",
                '<main class="page">',
                f"<h1>{html.escape(self.title)}</h1>",
                f'<p class="overall">Overall verdict: '
                f'<span class="badge {self.overall_verdict.value.lower()}">'
                f"{self.overall_verdict.value}</span></p>",
                f'<ul class="meta">{meta_items}</ul>',
                '<table class="summary"><thead><tr><th>Check</th><th>Verdict</th>'
                "<th>Finding</th></tr></thead><tbody>",
                rows,
                "</tbody></table>",
                *blocks,
                '<section class="limits">',
                "<h2>What this report does not prove</h2>",
                f"<ul>{limitations}</ul>",
                "</section>",
                f"<footer>Generated by quant-falsifier {__version__} at {self.created_at}</footer>",
                "</main>",
                "</body>",
                "</html>",
            ]
        )

    def write(self, path: str | Path) -> Path:
        """Write the report and return the path.

        Args:
            path: Destination ending in ``.md`` or ``.html``.

        Returns
        -------
            The path written.

        Raises
        ------
            ValueError: If the suffix is neither ``.md`` nor ``.html``.
        """
        destination = Path(path)
        suffix = destination.suffix.lower()
        if suffix == ".md":
            content = self.to_markdown()
        elif suffix in {".html", ".htm"}:
            content = self.to_html()
        else:
            raise ValueError(
                f"unsupported report extension {destination.suffix!r}; use .md or .html"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination

    def _html_meta(self) -> dict[str, str]:
        """Return header metadata for the HTML rendering."""
        meta = {"Generated": self.created_at, "Tool": f"quant-falsifier {__version__}"}
        if self.subject:
            meta["Subject"] = self.subject
        meta.update(self.metadata)
        return meta


def deflated_sharpe_section(
    result: DeflatedSharpeResult,
    *,
    alpha: float = 0.95,
    warn_margin: float = 0.05,
) -> Section:
    """Build a section from a deflated Sharpe ratio result.

    Args:
        result: Output of :func:`quant_falsifier.deflated_sharpe_ratio`.
        alpha: Probability required to pass.
        warn_margin: Width of the warning band below ``alpha``.

    Returns
    -------
        A :class:`Section`.
    """
    if result.dsr >= alpha:
        verdict = Verdict.PASS
    elif result.dsr >= alpha - warn_margin:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.FAIL
    summary = (
        f"After correcting for {result.n_trials} trials, the chance that the selected "
        f"strategy has positive true skill is {result.dsr:.3f}; the bar is {alpha:.2f}."
    )
    metrics = (
        ("Deflated Sharpe ratio", f"{result.dsr:.4f}"),
        ("Uncorrected PSR", f"{result.psr_vs_zero:.4f}"),
        ("Observed Sharpe (annualised)", f"{result.observed_sr:.4f}"),
        ("Expected max Sharpe under the null", f"{result.expected_max_sr:.4f}"),
        ("Sharpe dispersion across trials", f"{result.trial_sr_std:.4f}"),
        ("Trials", str(result.n_trials)),
        ("Observations", str(result.n_obs)),
        ("Skewness / kurtosis", f"{result.skew:.3f} / {result.kurtosis:.3f}"),
    )
    notes = (
        "The deflated Sharpe ratio is only as honest as the trial count. Every variant "
        "that was evaluated and discarded belongs in the ledger.",
    )
    return Section(
        title="Deflated Sharpe ratio",
        verdict=verdict,
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def reality_check_section(result: RealityCheckResult, *, alpha: float = 0.05) -> Section:
    """Build a section from White's reality check result."""
    verdict = _pvalue_verdict(result.pvalue, alpha)
    summary = (
        f"White's reality check over {result.n_trials} strategies gives p = "
        f"{result.pvalue:.4f}; the best strategy is column {result.best_index}."
    )
    metrics = (
        ("Reality check statistic", f"{result.statistic:.4f}"),
        ("p-value", f"{result.pvalue:.4f}"),
        ("Best mean return per bar", f"{float(result.means[result.best_index]):.6f}"),
        ("Trials", str(result.n_trials)),
        ("Bootstrap replications", str(result.n_boot)),
        ("Mean block length", f"{result.mean_block:.2f}"),
    )
    notes = (
        "The null is that no strategy beats the benchmark. A large p-value does not "
        "prove the winner is worthless, only that the search has not demonstrated value.",
    )
    return Section(
        title="Reality check (data snooping)",
        verdict=verdict,
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def spa_section(result: SpaResult, *, alpha: float = 0.05) -> Section:
    """Build a section from Hansen's SPA test result."""
    verdict = _pvalue_verdict(result.pvalue, alpha)
    summary = (
        f"Hansen's SPA test over {result.n_trials} strategies gives p = "
        f"{result.pvalue:.4f}; the strongest studentised strategy is column {result.best_index}."
    )
    metrics = (
        ("SPA statistic", f"{result.statistic:.4f}"),
        ("p-value", f"{result.pvalue:.4f}"),
        ("Studentised mean (best)", f"{float(result.studentised_means[result.best_index]):.4f}"),
        ("Trials", str(result.n_trials)),
        ("Bootstrap replications", str(result.n_boot)),
    )
    notes = (
        "SPA studentises each strategy before taking the maximum, so a single "
        "high-variance fluke cannot dominate the statistic.",
    )
    return Section(
        title="Superior predictive ability",
        verdict=verdict,
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def romano_wolf_section(result: RomanoWolfResult, *, alpha: float = 0.05) -> Section:
    """Build a section from a Romano-Wolf stepdown result."""
    significant = result.significant_indices()
    best = float(np.min(result.adjusted_pvalues)) if result.adjusted_pvalues.size else 1.0
    if significant:
        verdict = Verdict.PASS
    elif best <= 2.0 * alpha:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.FAIL
    summary = (
        f"{len(significant)} of {result.n_trials} strategies survive Romano-Wolf "
        f"family-wise correction at alpha = {alpha:.2f}."
    )
    metrics = (
        ("Surviving strategies", str(len(significant))),
        ("Smallest adjusted p-value", f"{best:.4f}"),
        ("Trials", str(result.n_trials)),
        ("Alpha", f"{alpha:.2f}"),
    )
    notes = (
        "Stepdown p-values control the family-wise error rate, so the surviving set "
        "is a claim about the group, not about any single strategy in isolation.",
    )
    return Section(
        title="Romano-Wolf stepdown",
        verdict=verdict,
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def split_audit_section(audit: SplitAudit) -> Section:
    """Build a section from a cross-validation split audit."""
    verdict = Verdict.PASS if audit.is_clean else Verdict.FAIL
    summary = (
        "No label overlap, index collision or embargo violation detected."
        if audit.is_clean
        else (
            f"Leakage found: {audit.n_label_overlaps} overlapping training labels, "
            f"{audit.n_index_collisions} index collisions and "
            f"{audit.n_embargo_violations} embargo violations across {audit.n_splits} splits."
        )
    )
    metrics = (
        ("Splits inspected", str(audit.n_splits)),
        ("Overlapping training labels", str(audit.n_label_overlaps)),
        ("Train/test index collisions", str(audit.n_index_collisions)),
        ("Embargo violations", str(audit.n_embargo_violations)),
        ("Observations", str(audit.n_obs)),
    )
    notes: tuple[str, ...] = ()
    if audit.examples:
        rendered = ", ".join(f"(split {s}, train {t})" for s, t, _ in audit.examples)
        notes = (f"Example offending samples: {rendered}.",)
    return Section(
        title="Cross-validation leakage",
        verdict=verdict,
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def point_in_time_section(audit: PointInTimeAudit) -> Section:
    """Build a section from a point-in-time availability audit."""
    verdict = Verdict.PASS if audit.passed else Verdict.FAIL
    if audit.passed:
        summary = "Every decision used only inputs that were available at the decision time."
    else:
        summary = (
            f"{audit.n_violations} decisions used inputs that were not yet available; "
            f"the worst violation leads the decision by {audit.worst_violation:.4f} time units."
        )
    metrics = (
        ("Decisions inspected", str(audit.n_checks)),
        ("Violations", str(audit.n_violations)),
        ("Worst violation", f"{audit.worst_violation:.4f}"),
        ("Publication lag", f"{audit.publication_lag:.4f}"),
        ("Tolerance", f"{audit.tolerance:.4f}"),
    )
    return Section(
        title="Point-in-time availability",
        verdict=verdict,
        summary=summary,
        metrics=metrics,
    )


def label_overlap_section(overlap: LabelOverlap) -> Section:
    """Build a section from an overlapping-label analysis."""
    ratio = overlap.effective_n / overlap.n_obs if overlap.n_obs else 1.0
    verdict = Verdict.WARN if ratio < 0.8 else Verdict.PASS
    summary = (
        f"Overlapping labels reduce the effective sample from {overlap.n_obs} to about "
        f"{overlap.effective_n:.1f} independent observations."
    )
    metrics = (
        ("Mean cluster size", f"{overlap.mean_cluster_size:.3f}"),
        ("Overlap fraction", f"{overlap.overlap_fraction:.3f}"),
        ("Effective sample size", f"{overlap.effective_n:.1f}"),
        ("Observations", str(overlap.n_obs)),
    )
    notes = (
        "This is not leakage by itself, but t-statistics computed on overlapping "
        "labels are too optimistic unless the sample size is adjusted.",
    )
    return Section(
        title="Overlapping labels",
        verdict=verdict,
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def delay_section(curve: DelayCurve, *, min_retention: float = 0.5) -> Section:
    """Build a section from a signal-delay curve.

    Args:
        curve: Output of :func:`quant_falsifier.signal_delay_curve`.
        min_retention: Minimum share of the baseline Sharpe ratio that must
            survive the next-larger delay to avoid a warning. The baseline is
            delay zero when present and the smallest measured delay otherwise.

    Returns
    -------
        A :class:`Section`.
    """
    zero_positions = np.flatnonzero(curve.delays == 0)
    has_zero = zero_positions.size > 0
    base_index = int(zero_positions[0]) if has_zero else int(np.argmin(curve.delays))
    base_delay = int(curve.delays[base_index])
    base = float(curve.sharpes[base_index])
    metrics: list[tuple[str, str]] = [
        (f"Sharpe at delay {int(delay)}", f"{float(value):.4f}")
        for delay, value in zip(curve.delays.tolist(), curve.sharpes.tolist(), strict=True)
    ]
    if base <= 0.0:
        verdict = Verdict.INFO
        if has_zero:
            summary = (
                "The undelayed signal has no positive Sharpe ratio, so delay sensitivity "
                "carries no information."
            )
        else:
            summary = (
                f"The signal at delay {base_delay} has no positive Sharpe ratio, so delay "
                "sensitivity carries no information."
            )
        retention = float("nan")
    else:
        later = np.flatnonzero(curve.delays > curve.delays[base_index])
        if later.size == 0:
            verdict = Verdict.INFO
            summary = (
                f"Only one delay ({base_delay}) was measured, so retention cannot be computed."
            )
            retention = float("nan")
        else:
            next_index = int(later[0])
            next_delay = int(curve.delays[next_index])
            retention = float(curve.sharpes[next_index]) / base
            verdict = Verdict.PASS if retention >= min_retention else Verdict.WARN
            metrics.append((f"Retention at delay {next_delay}", f"{retention:.3f}"))
            baseline_label = "delay-zero" if has_zero else f"delay-{base_delay}"
            summary = (
                f"Acting on the signal {next_delay - base_delay} bar(s) later keeps "
                f"{retention:.0%} of the {baseline_label} Sharpe ratio."
            )
    notes = (
        "A large drop at the first measured delay is not proof of leakage, but it means "
        "the result depends on trading information from the bar it is evaluated on.",
    )
    return Section(
        title="Signal delay sensitivity",
        verdict=verdict,
        summary=summary,
        metrics=tuple(metrics),
        notes=notes,
    )


def permutation_section(result: PermutationResult, *, alpha: float = 0.05) -> Section:
    """Build a section from a block-permutation test."""
    verdict = _pvalue_verdict(result.pvalue, alpha)
    summary = (
        f"Block-permutation test ({result.metric}) gives p = {result.pvalue:.4f} "
        f"against a null with mean {result.null_mean:.4f}."
    )
    metrics = (
        ("Observed metric", f"{result.observed:.4f}"),
        ("p-value", f"{result.pvalue:.4f}"),
        ("Null mean / std", f"{result.null_mean:.4f} / {result.null_std:.4f}"),
        ("Permutations", str(result.n_perm)),
        ("Block length", str(result.mean_block)),
    )
    return Section(
        title="Permutation test",
        verdict=verdict,
        summary=summary,
        metrics=metrics,
    )


def ledger_section(
    verification: LedgerVerification,
    *,
    records: Sequence[TrialRecord] | None = None,
) -> Section:
    """Build a section from a ledger verification result.

    Args:
        verification: Output of :meth:`quant_falsifier.TrialLedger.verify`.
        records: Optional records, used to report the size of the search.

    Returns
    -------
        A :class:`Section`.
    """
    if not verification.valid:
        verdict = Verdict.FAIL
        summary = verification.describe()
    elif verification.n_records == 0:
        verdict = Verdict.INFO
        summary = "The ledger is empty, so no trial count is available."
    else:
        verdict = Verdict.PASS
        summary = verification.describe()
    metrics: list[tuple[str, str]] = [
        ("Records", str(verification.n_records)),
        ("Chain intact", "yes" if verification.valid else "no"),
    ]
    if records:
        unique = len({record.hypothesis_id for record in records})
        metrics.append(("Distinct hypotheses", str(unique)))
    return Section(
        title="Trial ledger",
        verdict=verdict,
        summary=summary,
        metrics=tuple(metrics),
        notes=(
            "The hash chain makes after-the-fact edits detectable. It does not prove "
            "that every attempted variant was recorded.",
        ),
    )


_LIMITATIONS = (
    "These checks cannot prove the absence of look-ahead bias; they detect specific, "
    "common failure modes and make the rest of the research protocol auditable.",
    "Statistical corrections assume the trials are known and reasonably distinct. "
    "Hidden variation (universe tweaks, date edits, silent data revisions) is invisible here.",
    "Nothing in this report accounts for transaction costs, capacity, borrow costs or "
    "market impact unless those are already inside the return series you supplied.",
    "Results are conditional on the sample used. A pass is not evidence of future "
    "profitability and is not investment advice.",
)


def _limitations_markdown() -> list[str]:
    """Render the standard limitations block as Markdown."""
    lines = ["## What this report does not prove", ""]
    lines.extend(f"- {item}" for item in _LIMITATIONS)
    return lines


def _pvalue_verdict(pvalue: float, alpha: float) -> Verdict:
    """Map a p-value to a verdict, with a one-alpha-width warning band."""
    if pvalue <= alpha:
        return Verdict.PASS
    if pvalue <= 2.0 * alpha:
        return Verdict.WARN
    return Verdict.FAIL


def _md_text(text: str) -> str:
    """Escape characters that Markdown renderers would treat as raw HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_cell(text: str) -> str:
    """Escape a Markdown table cell."""
    return _md_text(text).replace("|", "\\|").replace("\n", " ").strip()


_CSS = """
:root {
  --fg: #1f2328;
  --muted: #59636e;
  --border: #d1d9e0;
  --bg: #ffffff;
  --pass: #1a7f37;
  --warn: #9a6700;
  --fail: #cf222e;
  --info: #59636e;
  --surface: #f6f8fa;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font:
    15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
    Helvetica, Arial, sans-serif;
}
.page { max-width: 900px; margin: 0 auto; padding: 48px 24px 64px; }
h1 { font-size: 28px; letter-spacing: -0.01em; margin: 0 0 12px; }
h2 {
  font-size: 19px;
  margin: 36px 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
p { margin: 10px 0; }
.overall { font-size: 17px; }
.meta, .notes { color: var(--muted); padding-left: 20px; }
.meta li, .notes li { margin: 4px 0; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }
th, td {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
thead th { background: var(--surface); font-weight: 600; }
table.metrics th { width: 45%; font-weight: 500; color: var(--muted); }
.badge {
  display: inline-block; padding: 1px 8px; border-radius: 999px;
  font: 600 12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: 0.02em; border: 1px solid currentColor;
}
.badge.pass { color: var(--pass); }
.badge.warn { color: var(--warn); }
.badge.fail { color: var(--fail); }
.badge.info { color: var(--info); }
.limits {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 20px 20px;
  margin-top: 40px;
}
.limits h2 { border-bottom: none; margin-top: 16px; }
footer { margin-top: 32px; color: var(--muted); font-size: 12px; }
@media print {
  .page { max-width: none; padding: 0; }
  .limits { break-inside: avoid; }
}
"""
