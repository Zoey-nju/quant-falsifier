# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Bootstrap-based tests (`reality_check`, `spa_test`, `romano_wolf`) now draw index paths in
  bounded chunks. Peak memory on a 20,000-row series with 2,000 replicates fell from about
  2 GB to about 110 MB, so long series no longer exhaust memory.
- `TrialLedger.append` no longer reads the whole file on every call. Appends are O(1) in ledger
  size, and each write is flushed with `os.fsync` before returning.
- Concurrent writers no longer fork the hash chain: `append` holds an exclusive cross-process
  lock (``msvcrt`` on Windows, ``fcntl`` on POSIX) while reading the previous hash and writing.
- A corrupt or deeply nested last line now raises ``ValueError`` instead of leaking
  ``JSONDecodeError`` or ``RecursionError``; ``verify`` reports such files as invalid.
- `demo --out` refuses to delete a `trials.jsonl` that was not produced by the demo; pass
  `--force` to overwrite deliberately.
- Markdown reports escape raw HTML in user-supplied titles, subjects, metadata, section text
  and notes, matching the HTML renderer.
- `DelayCurve.retention` and `delay_section` no longer assume delay zero is present, and
  `deflated_sharpe_ratio_from_stats` accepts a `periods_per_year` argument.
- `audit_splits` now counts embargo violations in gaps between non-contiguous test blocks.
- `spa_test` uses the same bootstrap p-value convention (`>=`) as the other tests.

### Removed

- Unused internal helpers `as_float_array`, `describe_seconds` and `safe_log`.

### Security

- `.gitignore` now covers `.env*`, `secrets.json`, `credentials*.json`, `*.pem`, `*.key`,
  `*.p12`, `*.pfx`, SSH keys and local database files.

## [0.1.0] - 2026-11-16

First public release. The scope is deliberately narrow: record the search, correct the
statistics for it, screen the evaluation protocol, and write down what the checks cannot
prove.

### Added

- `TrialLedger`: append-only JSON Lines ledger with a SHA-256 hash chain, plus `fingerprint_data`
  for pinning the input series to a record.
- `metrics`: `sharpe_ratio`, `probabilistic_sharpe_ratio`, `expected_max_sharpe`,
  `deflated_sharpe_ratio`, `deflated_sharpe_ratio_from_stats`, `min_track_record_length` and
  `per_period_sharpes`.
- `multiple_testing`: stationary bootstrap, White's reality check, Hansen's SPA test and the
  Romano-Wolf stepdown procedure.
- `leakage`: `PurgedKFold` with embargo, `audit_splits` (works on splits from any library),
  `check_point_in_time`, `label_overlap`, `signal_delay_curve` and `permutation_pvalue`.
- `report`: `AuditReport` with Markdown and self-contained HTML rendering, per-check verdicts
  and a fixed "what this report does not prove" section.
- `synthetic`: seeded synthetic datasets for tests and the demo, so no market data is shipped.
- CLI: `quant-falsifier demo` and `quant-falsifier ledger verify`.
- Documentation, Apache-2.0 license, CI (lint, type check, tests on Python 3.10-3.13),
  PyPI trusted publishing and Dependabot.

[Unreleased]: https://github.com/Zoey-nju/quant-falsifier/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Zoey-nju/quant-falsifier/releases/tag/v0.1.0
