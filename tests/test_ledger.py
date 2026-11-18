"""Tests for the hash-chained trial ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quant_falsifier import GENESIS_HASH, TrialLedger, fingerprint_data


def test_append_links_records(tmp_path: pytest.TempPathFactory) -> None:
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    first = ledger.append("mom_20d", params={"lookback": 20}, metrics={"sharpe_per_period": 0.05})
    second = ledger.append("mom_60d", metrics={"sharpe_per_period": 0.02})
    assert first.seq == 0
    assert first.prev_hash == GENESIS_HASH
    assert second.seq == 1
    assert second.prev_hash == first.hash
    assert len(ledger) == 2
    assert ledger.verify().valid
    assert ledger.sharpes().tolist() == pytest.approx([0.05, 0.02])


def test_verification_survives_reopening(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "trials.jsonl"
    TrialLedger(path).append("a", metrics={"sharpe_per_period": 0.01})
    reopened = TrialLedger(path)
    assert reopened.verify().valid
    assert len(reopened) == 1


def test_tampering_with_a_metric_is_detected(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(path)
    for index in range(3):
        ledger.append(f"trial_{index}", metrics={"sharpe_per_period": 0.01 * index})
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[1])
    payload["metrics"]["sharpe_per_period"] = 0.99
    lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verification = ledger.verify()
    assert not verification.valid
    assert verification.broken_at == 1
    assert "hash" in (verification.reason or "")


def test_removing_a_record_breaks_the_chain(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(path)
    for index in range(3):
        ledger.append(f"trial_{index}", metrics={"sharpe_per_period": 0.01})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    verification = ledger.verify()
    assert verification.valid
    assert verification.n_records == 2

    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    verification = ledger.verify()
    assert not verification.valid
    assert verification.broken_at == 1


def test_rejects_non_finite_metrics(tmp_path: pytest.TempPathFactory) -> None:
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError, match="finite"):
        ledger.append("bad", metrics={"sharpe_per_period": float("nan")})
    with pytest.raises(ValueError, match="non-empty"):
        ledger.append("   ")


def test_rejects_unserialisable_params(tmp_path: pytest.TempPathFactory) -> None:
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError, match="JSON-serialisable"):
        ledger.append("bad", params={"model": object()})


def test_empty_and_missing_ledger(tmp_path: pytest.TempPathFactory) -> None:
    ledger = TrialLedger(tmp_path / "missing.jsonl")
    assert len(ledger) == 0
    assert ledger.verify().valid
    assert ledger.verify().n_records == 0
    assert ledger.sharpes().size == 0


def test_missing_metric_raises_with_context(tmp_path: pytest.TempPathFactory) -> None:
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    ledger.append("only_params", params={"x": 1})
    with pytest.raises(KeyError, match="only_params"):
        ledger.sharpes()


def test_fingerprint_is_stable_and_sensitive() -> None:
    data = np.linspace(0.0, 1.0, 100)
    assert fingerprint_data(data) == fingerprint_data(data.copy())
    assert fingerprint_data(data) != fingerprint_data(data + 1e-9)
    assert fingerprint_data(data) != fingerprint_data(data.reshape(10, 10))
    assert fingerprint_data(data).startswith("sha256:")


def test_records_round_trip_params(tmp_path: pytest.TempPathFactory) -> None:
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    ledger.append("h", params={"b": 2, "a": [1, 2]}, notes="checked")
    record = ledger.records()[0]
    assert record.params == {"a": [1, 2], "b": 2}
    assert record.notes == "checked"
    assert record.compute_hash() == record.hash


def test_append_reports_corrupt_last_line_as_value_error(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    path.write_text('{"seq": 0', encoding="utf-8")
    with pytest.raises(ValueError, match="last ledger record"):
        TrialLedger(path).append("h")


def test_verify_marks_deeply_nested_json_invalid(tmp_path: Path) -> None:
    path = tmp_path / "deep.jsonl"
    path.write_text("[" * 100_000 + "]" * 100_000 + "\n", encoding="utf-8")
    verification = TrialLedger(path).verify()
    assert not verification.valid


def test_append_does_not_read_the_whole_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    ledger.append("a", metrics={"sharpe_per_period": 0.01})

    def boom(self: Path) -> bytes:
        raise AssertionError("append must not read the whole ledger")

    monkeypatch.setattr(Path, "read_bytes", boom)
    ledger.append("b", metrics={"sharpe_per_period": 0.02})
    assert len(ledger) == 2
    assert ledger.verify().valid


def test_append_to_empty_file_has_no_blank_first_line(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    TrialLedger(path).append("a", metrics={"sharpe_per_period": 0.01})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["hypothesis_id"] == "a"
