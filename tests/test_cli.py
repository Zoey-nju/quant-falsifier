"""End-to-end tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_falsifier import TrialLedger
from quant_falsifier.cli import main


def test_demo_writes_reports_and_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "out"
    exit_code = main(["demo", "--out", str(out_dir), "--seed", "3"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Honest audit: overall verdict" in captured
    assert "Negative controls: overall verdict FAIL" in captured
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "negative_controls.md").exists()
    assert (out_dir / "negative_controls.html").exists()
    assert (out_dir / "trials.jsonl").exists()
    assert TrialLedger(out_dir / "trials.jsonl").verify().valid
    assert "What this report does not prove" in (out_dir / "report.md").read_text(encoding="utf-8")
    controls = (out_dir / "negative_controls.md").read_text(encoding="utf-8")
    assert "naive K-fold" in controls
    assert "`FAIL`" in controls


def test_demo_is_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    main(["demo", "--out", str(first), "--seed", "11"])
    main(["demo", "--out", str(second), "--seed", "11"])

    def without_timestamp(path: Path) -> str:
        lines = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(line for line in lines if not line.startswith("- **Generated:**"))

    assert without_timestamp(first / "report.md") == without_timestamp(second / "report.md")


def test_ledger_verify_passes_and_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(path)
    ledger.append("h", metrics={"sharpe_per_period": 0.02})
    assert main(["ledger", "verify", str(path)]) == 0
    assert "verified" in capsys.readouterr().out

    path.write_text(path.read_text(encoding="utf-8").replace("0.02", "5.02"), encoding="utf-8")
    assert main(["ledger", "verify", str(path)]) == 1
    assert "failed" in capsys.readouterr().out


def test_ledger_verify_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "trials.jsonl"
    TrialLedger(path).append("h", metrics={"sharpe_per_period": 0.02})
    assert main(["ledger", "verify", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["n_records"] == 1


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "quant-falsifier" in capsys.readouterr().out


def test_demo_refuses_to_delete_a_foreign_ledger(tmp_path: Path) -> None:
    out_dir = tmp_path / "mine"
    out_dir.mkdir()
    TrialLedger(out_dir / "trials.jsonl").append("my_strategy", metrics={"sharpe_per_period": 0.1})

    with pytest.raises(SystemExit, match="does not look like demo output"):
        main(["demo", "--out", str(out_dir)])

    records = TrialLedger(out_dir / "trials.jsonl").records()
    assert [record.hypothesis_id for record in records] == ["my_strategy"]


def test_demo_force_overwrites_a_foreign_ledger(tmp_path: Path) -> None:
    out_dir = tmp_path / "mine"
    out_dir.mkdir()
    TrialLedger(out_dir / "trials.jsonl").append("my_strategy", metrics={"sharpe_per_period": 0.1})

    assert main(["demo", "--out", str(out_dir), "--force"]) == 0
    records = TrialLedger(out_dir / "trials.jsonl").records()
    assert records
    assert all(record.hypothesis_id.startswith("variant_") for record in records)
