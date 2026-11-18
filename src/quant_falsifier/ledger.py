"""An append-only, hash-chained ledger of every hypothesis that was tested.

Backtest overfitting is usually not a modelling problem, it is a bookkeeping
problem: the number of trials that produced the winning Sharpe ratio is known to
whoever ran them until it becomes inconvenient. This module makes that number a
first-class artifact.

Each record is a JSON object whose ``hash`` covers its own payload and the hash of
the previous record, forming a chain from :data:`GENESIS_HASH`. Editing an old
line changes its hash and breaks every link after it, so accidental edits and
post-hoc insertions are detectable. The chain is not signed: anyone who rewrites
the whole file can rebuild a consistent chain, so anchor the final hash somewhere
external (a commit, an email, a timestamping service) if that threat matters. The
ledger is not a blockchain and does not need one: it is a local, auditable file
that a reviewer can verify in one command.

Example:
    >>> from quant_falsifier import TrialLedger
    >>> ledger = TrialLedger("trials.jsonl")            # doctest: +SKIP
    >>> ledger.append(                                  # doctest: +SKIP
    ...     hypothesis_id="mom_20d",
    ...     params={"lookback": 20, "universe": "top100"},
    ...     metrics={"sharpe_per_period": 0.071},
    ... )
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import numpy as np

from ._validation import FloatArray, as_1d

__all__ = [
    "GENESIS_HASH",
    "LedgerVerification",
    "TrialLedger",
    "TrialRecord",
    "fingerprint_data",
]

#: Hash of the (virtual) record preceding the first entry.
GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class TrialRecord:
    """One immutable entry in a :class:`TrialLedger`.

    Attributes
    ----------
        seq: Zero-based position in the ledger.
        created_at: UTC ISO-8601 timestamp with second precision.
        hypothesis_id: Short identifier of the hypothesis under test.
        params: JSON-serialisable parameters of the trial.
        metrics: Numeric results, ideally including ``sharpe_per_period``.
        data_fingerprint: Optional digest of the dataset used.
        notes: Optional free-text note.
        prev_hash: Hash of the preceding record, or :data:`GENESIS_HASH`.
        hash: SHA-256 digest of this record's payload.
    """

    seq: int
    created_at: str
    hypothesis_id: str
    params: dict[str, Any]
    metrics: dict[str, float]
    data_fingerprint: str | None
    notes: str | None
    prev_hash: str
    hash: str

    def payload(self) -> dict[str, Any]:
        """Return the hashed portion of the record as a dictionary."""
        return {
            "seq": self.seq,
            "created_at": self.created_at,
            "hypothesis_id": self.hypothesis_id,
            "params": self.params,
            "metrics": self.metrics,
            "data_fingerprint": self.data_fingerprint,
            "notes": self.notes,
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        """Recompute the record hash from its payload."""
        return _digest(self.payload())

    def to_json(self) -> str:
        """Serialise the record, including its hash, as a single JSON line."""
        payload = self.payload()
        payload["hash"] = self.hash
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class LedgerVerification:
    """Outcome of verifying a ledger's hash chain.

    Attributes
    ----------
        valid: Whether every record parses and the chain is intact.
        n_records: Number of records inspected.
        broken_at: Zero-based index of the first invalid record, if any.
        reason: Human-readable explanation when ``valid`` is false.
    """

    valid: bool
    n_records: int
    broken_at: int | None = None
    reason: str | None = None

    def describe(self) -> str:
        """Return a one-line human-readable summary."""
        if self.valid:
            return f"ledger verified: {self.n_records} record(s), hash chain intact"
        return f"ledger verification failed at record {self.broken_at}: {self.reason}"


class TrialLedger:
    """Append-only ledger of trials stored as newline-delimited JSON.

    Args:
        path: File path of the ledger. The parent directory is created on first
            append; reading a missing file yields an empty ledger.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __len__(self) -> int:
        """Return the number of records in the ledger."""
        return len(self.records())

    def __iter__(self) -> Iterator[TrialRecord]:
        """Iterate over records in append order."""
        return iter(self.records())

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return f"TrialLedger({str(self.path)!r}, n_records={len(self)})"

    def append(
        self,
        hypothesis_id: str,
        *,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        data_fingerprint: str | None = None,
        notes: str | None = None,
    ) -> TrialRecord:
        """Append a trial and return the resulting record.

        Args:
            hypothesis_id: Short identifier of the hypothesis.
            params: JSON-serialisable trial parameters.
            metrics: Numeric results. Non-finite values are rejected because they
                cannot be hashed meaningfully or compared across runs.
            data_fingerprint: Digest produced by :func:`fingerprint_data`.
            notes: Optional free-text note.

        Returns
        -------
            The appended :class:`TrialRecord`.

        Raises
        ------
            ValueError: If the identifier is empty, a metric is non-finite, or
                ``params`` is not JSON-serialisable.
        """
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise ValueError("hypothesis_id must be a non-empty string")
        clean_metrics = _clean_metrics(metrics)
        clean_params = _clean_params(params)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One open handle serves both the read of the previous record and the
        # write, and an exclusive lock keeps concurrent writers from reading the
        # same ``prev_hash`` and forking the chain.
        with self.path.open("a+b") as handle, _exclusive_lock(handle):
            previous = self._last_record_from(handle)
            seq = (previous.seq + 1) if previous is not None else 0
            record = TrialRecord(
                seq=seq,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                hypothesis_id=hypothesis_id.strip(),
                params=clean_params,
                metrics=clean_metrics,
                data_fingerprint=data_fingerprint,
                notes=notes,
                prev_hash=previous.hash if previous else GENESIS_HASH,
                hash="",
            )
            sealed = TrialRecord(
                seq=record.seq,
                created_at=record.created_at,
                hypothesis_id=record.hypothesis_id,
                params=record.params,
                metrics=record.metrics,
                data_fingerprint=record.data_fingerprint,
                notes=record.notes,
                prev_hash=record.prev_hash,
                hash=record.compute_hash(),
            )
            handle.seek(0, os.SEEK_END)
            if _needs_separator_newline(handle):
                handle.write(b"\n")
            handle.write((sealed.to_json() + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        return sealed

    def records(self) -> list[TrialRecord]:
        """Return all records, raising if the file cannot be parsed.

        Returns
        -------
            Records in append order, or an empty list when the file is missing.

        Raises
        ------
            ValueError: If a line is not a valid ledger record.
        """
        if not self.path.exists():
            return []
        out: list[TrialRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, RecursionError) as exc:
                    raise ValueError(f"line {line_number} is not valid JSON: {exc}") from exc
                out.append(_record_from_payload(payload, line_number))
        return out

    def _last_record(self) -> TrialRecord | None:
        """Return the last record, or ``None`` when the ledger is empty."""
        if not self.path.exists():
            return None
        with self.path.open("rb") as handle:
            return self._last_record_from(handle)

    def _last_record_from(self, handle: IO[bytes]) -> TrialRecord | None:
        """Return the last record read from an already-open binary handle.

        Reads only the final line of the file so that appending stays O(1)
        regardless of ledger size.
        """
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            return None
        chunk_size = min(size, 8192)
        tail = b""
        offset = size
        while offset > 0:
            read_size = min(chunk_size, offset)
            offset -= read_size
            handle.seek(offset)
            tail = handle.read(read_size) + tail
            # Drop trailing newlines so the last non-empty line is found.
            while tail.endswith(b"\n") or tail.endswith(b"\r"):
                tail = tail[:-1]
            newline = tail.rfind(b"\n")
            last_line = tail[newline + 1 :] if newline != -1 else tail
            if last_line:
                try:
                    payload = json.loads(last_line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
                    raise ValueError(f"the last ledger record is not valid JSON: {exc}") from exc
                return _record_from_payload(payload, 0)
            # No non-empty content yet (file is whitespace-only so far);
            # keep reading backwards.
            tail = tail[:newline] if newline != -1 else b""
        return None

    def verify(self) -> LedgerVerification:
        """Verify sequence numbers, previous-hash links and record hashes.

        Returns
        -------
            A :class:`LedgerVerification`. Structural problems that prevent
            parsing at all are reported as invalid rather than raised, so the
            method can be used directly by a report or a CI check.
        """
        try:
            records = self.records()
        except ValueError as exc:
            return LedgerVerification(valid=False, n_records=0, broken_at=0, reason=str(exc))

        expected_prev = GENESIS_HASH
        for position, record in enumerate(records):
            if record.seq != position:
                return LedgerVerification(
                    valid=False,
                    n_records=len(records),
                    broken_at=position,
                    reason=f"sequence number is {record.seq}, expected {position}",
                )
            if record.prev_hash != expected_prev:
                return LedgerVerification(
                    valid=False,
                    n_records=len(records),
                    broken_at=position,
                    reason="previous-hash link does not match the preceding record",
                )
            if record.compute_hash() != record.hash:
                return LedgerVerification(
                    valid=False,
                    n_records=len(records),
                    broken_at=position,
                    reason="record content does not match its stored hash",
                )
            expected_prev = record.hash
        return LedgerVerification(valid=True, n_records=len(records))

    def sharpes(self, key: str = "sharpe_per_period") -> FloatArray:
        """Return one per-period Sharpe ratio per record.

        Args:
            key: Metric name to read from each record.

        Returns
        -------
            Array of per-period Sharpe ratios, ready for
            :func:`quant_falsifier.deflated_sharpe_ratio`.

        Raises
        ------
            KeyError: If a record does not contain ``key``.
        """
        values: list[float] = []
        for record in self.records():
            if key not in record.metrics:
                raise KeyError(
                    f"record {record.seq} ({record.hypothesis_id}) has no metric {key!r}; "
                    f"available metrics: {sorted(record.metrics)}"
                )
            values.append(float(record.metrics[key]))
        if not values:
            return np.empty(0, dtype=np.float64)
        return as_1d(values, key)

    def parameter_sets(self) -> list[dict[str, Any]]:
        """Return the parameters of every record, useful for spotting near-duplicates."""
        return [dict(record.params) for record in self.records()]


@contextmanager
def _exclusive_lock(handle: IO[bytes]) -> Iterator[None]:
    """Hold an exclusive, cross-process lock on ``handle``.

    Windows uses ``msvcrt.locking`` and POSIX uses ``fcntl.flock``; both are
    advisory locks that only coordinate cooperating writers. On Windows the
    acquisition retries for about ten seconds before raising ``OSError``, so a
    stuck peer cannot block the caller indefinitely.
    """
    if os.name == "nt":
        module = importlib.import_module("msvcrt")
        handle.seek(0)
        module.locking(handle.fileno(), module.LK_LOCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            module.locking(handle.fileno(), module.LK_UNLCK, 1)
    else:
        module = importlib.import_module("fcntl")
        module.flock(handle.fileno(), module.LOCK_EX)
        try:
            yield
        finally:
            module.flock(handle.fileno(), module.LOCK_UN)


def _needs_separator_newline(handle: IO[bytes]) -> bool:
    """Return whether a new record needs a leading newline separator.

    True only when the file is non-empty and its last byte is not a newline, so
    appending to an empty ledger does not create a blank first line.
    """
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        return False
    handle.seek(-1, os.SEEK_END)
    return handle.read(1) != b"\n"


def fingerprint_data(values: object) -> str:
    """Return a stable SHA-256 digest of a numeric array.

    The digest covers the dtype, the shape and the float64 payload, so a modified
    or reordered dataset produces a different fingerprint without storing the data.

    Args:
        values: Array-like numeric data.

    Returns
    -------
        A string of the form ``sha256:<hex>``.
    """
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    # ``ndarray.data`` exposes the existing buffer as a memoryview instead of
    # copying it the way ``tobytes()`` would, which keeps memory flat.
    digest.update(array.data)
    return f"sha256:{digest.hexdigest()}"


def _clean_metrics(metrics: dict[str, float] | None) -> dict[str, float]:
    """Validate and coerce a metrics mapping."""
    if not metrics:
        return {}
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"metric {key!r} must be finite, got {value!r}")
        cleaned[str(key)] = number
    return cleaned


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Validate that parameters round-trip through JSON with sorted keys."""
    if not params:
        return {}
    try:
        json.dumps(params, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"params must be JSON-serialisable: {exc}") from exc
    decoded: dict[str, Any] = json.loads(json.dumps(params, sort_keys=True, allow_nan=False))
    return decoded


def _digest(payload: dict[str, Any]) -> str:
    """Return the SHA-256 digest of a canonical JSON payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _record_from_payload(payload: Any, line_number: int) -> TrialRecord:
    """Build a :class:`TrialRecord` from a decoded JSON object."""
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number} is not a JSON object")
    try:
        seq = int(payload["seq"])
        created_at = str(payload["created_at"])
        hypothesis_id = str(payload["hypothesis_id"])
        params = dict(payload.get("params") or {})
        metrics = {str(k): float(v) for k, v in dict(payload.get("metrics") or {}).items()}
        data_fingerprint = payload.get("data_fingerprint")
        notes = payload.get("notes")
        prev_hash = str(payload["prev_hash"])
        record_hash = str(payload["hash"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number} is missing ledger fields: {exc}") from exc
    return TrialRecord(
        seq=seq,
        created_at=created_at,
        hypothesis_id=hypothesis_id,
        params=params,
        metrics=metrics,
        data_fingerprint=str(data_fingerprint) if data_fingerprint is not None else None,
        notes=str(notes) if notes is not None else None,
        prev_hash=prev_hash,
        hash=record_hash,
    )
