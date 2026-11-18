"""Tests for Markdown and HTML report rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_falsifier import (
    AuditReport,
    Section,
    Verdict,
    audit_splits,
    deflated_sharpe_ratio,
    deflated_sharpe_section,
    delay_section,
    ledger_section,
    per_period_sharpes,
    signal_delay_curve,
    split_audit_section,
)


def _report() -> AuditReport:
    return AuditReport(
        title="Test audit",
        subject="unit test",
        metadata={"Seed": "0"},
    ).add(
        Section(
            title="Toy section",
            verdict=Verdict.WARN,
            summary="Something <script>alert(1)</script> looks off | here",
            metrics=(("Metric", "1.2345"),),
            notes=("A note.",),
        )
    )


def test_markdown_contains_verdicts_and_limitations() -> None:
    text = _report().to_markdown()
    assert "# Test audit" in text
    assert "`WARN`" in text
    assert "What this report does not prove" in text
    assert "Toy section" in text
    assert "A note." in text


def test_html_escapes_user_content() -> None:
    page = _report().to_html()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert page.startswith("<!DOCTYPE html>")
    assert "<style>" in page
    assert "quant-falsifier" in page


def test_markdown_escapes_raw_html_in_user_content() -> None:
    text = _report().to_markdown()
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_markdown_escapes_html_in_title_and_metadata() -> None:
    report = AuditReport(title="<b>Audit</b>", subject="a & b", metadata={"<key>": "<value>"})
    text = report.to_markdown()
    assert "<b>" not in text
    assert "&lt;b&gt;Audit&lt;/b&gt;" in text
    assert "a &amp; b" in text
    assert "&lt;key&gt;" in text


def test_overall_verdict_reflects_the_worst_section() -> None:
    report = AuditReport()
    report.add(Section("a", Verdict.PASS, "ok"))
    report.add(Section("b", Verdict.WARN, "warning"))
    assert report.overall_verdict is Verdict.WARN
    report.add(Section("c", Verdict.FAIL, "bad"))
    assert report.overall_verdict is Verdict.FAIL
    empty = AuditReport()
    assert empty.overall_verdict is Verdict.INFO


def test_write_dispatches_on_extension(tmp_path: Path) -> None:
    report = _report()
    markdown = report.write(tmp_path / "nested" / "report.md")
    page = report.write(tmp_path / "nested" / "report.html")
    assert markdown.read_text(encoding="utf-8").startswith("# Test audit")
    assert page.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    with pytest.raises(ValueError, match="unsupported"):
        report.write(tmp_path / "report.txt")


def test_deflated_sharpe_section_verdict_tracks_threshold() -> None:
    rng = np.random.default_rng(0)
    trials = rng.normal(0.0, 0.01, size=(400, 10))
    sharpes = per_period_sharpes(trials)
    section = deflated_sharpe_section(deflated_sharpe_ratio(trials[:, 0], sharpes))
    assert section.verdict in {Verdict.PASS, Verdict.WARN, Verdict.FAIL}
    assert "Deflated Sharpe ratio" in dict(section.metrics)


def test_split_audit_section_flags_leakage() -> None:
    times = np.arange(50, dtype=np.float64)
    ends = times + 4.0
    naive = [(np.setdiff1d(np.arange(50), test), test) for test in np.array_split(np.arange(50), 5)]
    section = split_audit_section(audit_splits(naive, times, label_end_times=ends))
    assert section.verdict is Verdict.FAIL
    assert any("split 1" in note for note in section.notes)


def test_ledger_section_reports_empty_ledger(tmp_path: Path) -> None:
    from quant_falsifier import TrialLedger

    ledger = TrialLedger(tmp_path / "trials.jsonl")
    section = ledger_section(ledger.verify())
    assert section.verdict is Verdict.INFO
    assert "empty" in section.summary


def test_ledger_section_fails_on_broken_chain(tmp_path: Path) -> None:
    from quant_falsifier import TrialLedger

    path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(path)
    ledger.append("a", metrics={"sharpe_per_period": 0.01})
    path.write_text(path.read_text(encoding="utf-8").replace("0.01", "9.99"), encoding="utf-8")
    section = ledger_section(ledger.verify())
    assert section.verdict is Verdict.FAIL


def test_delay_section_without_delay_zero_uses_first_delay_as_base() -> None:
    rng = np.random.default_rng(6)
    forward = rng.normal(0.0, 0.01, size=400)
    signal = forward + 0.3 * rng.normal(0.0, 0.01, size=400)
    curve = signal_delay_curve(signal, forward, delays=(2, 3))
    section = delay_section(curve)
    # No "undelayed" wording since delay zero is absent.
    assert "undelayed" not in section.summary
    assert "delay 2" in section.summary
