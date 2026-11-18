# API reference

Every public object is re-exported from the package root, so `from quant_falsifier import ...`
is always sufficient. Signatures below are the supported surface; anything with a leading
underscore is internal and may change without notice.

All functions accept any array-like input — lists, tuples, NumPy arrays and pandas objects —
and convert it with `numpy.asarray`. Inputs that cannot be interpreted as finite one- or
two-dimensional numeric arrays raise `ValueError`; nothing is repaired silently.

## Metrics

### `sharpe_ratio(returns, *, periods_per_year=252, annualize=True, risk_free=0.0) -> float`

Mean excess return divided by its sample standard deviation, computed on the native sampling
frequency of `returns`. `annualize=True` multiplies by `sqrt(periods_per_year)`.
`risk_free` is an annual rate converted to a per-period rate before subtraction.

Raises `ValueError` for fewer than two observations, non-finite values or zero variance.

### `per_period_sharpes(trial_returns, *, risk_free=0.0) -> ndarray`

Per-period Sharpe ratio for every column of a `(n_obs, n_trials)` matrix. This is the correct
input for every correction in the library; annualise only when printing.

### `probabilistic_sharpe_ratio(sr, n_obs, *, skew=0.0, kurtosis=3.0, benchmark_sr=0.0) -> float`

Bailey & López de Prado (2012). Probability that the true Sharpe ratio exceeds `benchmark_sr`,
given sampling error and the higher moments of the returns. All Sharpe ratios are per-period.

### `expected_max_sharpe(n_trials, sr_std) -> float`

Expected maximum per-period Sharpe ratio of `n_trials` independent worthless strategies, given
the standard deviation of trial Sharpe ratios. Requires `n_trials >= 2` and `sr_std > 0`.

### `deflated_sharpe_ratio(returns, trial_sharpes, *, periods_per_year=252, risk_free=0.0) -> DeflatedSharpeResult`

Convenience wrapper: computes the observed ratio from `returns`, then deflates it against
`trial_sharpes` (per-period values, including the selected trial). Returns a frozen dataclass
with `dsr`, `psr_vs_zero`, `observed_sr`, `expected_max_sr`, `trial_sr_std`, `n_obs`,
`n_trials`, `skew`, `kurtosis` and `periods_per_year`. The annualised fields assume
`periods_per_year`.

Reject when `dsr` is below `0.95`; treat `0.90`–`0.95` as a warning band.

### `deflated_sharpe_ratio_from_stats(sr, n_obs, *, skew, kurtosis, trial_sharpes) -> DeflatedSharpeResult`

Same result from summary statistics, for the case where only the moments were stored. The
annualised fields assume 252 periods per year.

### `min_track_record_length(sr, *, skew=0.0, kurtosis=3.0, benchmark_sr=0.0, confidence=0.95) -> float`

Minimum number of observations before `sr` is distinguishable from `benchmark_sr` at the given
confidence. Returns a float because the formula is not integer-valued.

## Multiple testing

### `stationary_bootstrap_indices(n_obs, n_boot, *, mean_block=None, seed=None) -> ndarray`

`(n_boot, n_obs)` integer array of resampled indices from the stationary bootstrap
(Politis & Romano, 1994). `mean_block` defaults to `default_block_length(n_obs)`, i.e.
`round(n_obs ** (1/3))`. Fixed `seed` makes the draw reproducible.

### `reality_check(returns, *, benchmark=0.0, n_boot=1000, mean_block=None, seed=None) -> RealityCheckResult`

White's (2000) data-snooping test over the columns of `returns`, where each column is a
strategy's per-period return in excess of `benchmark`. Fields: `statistic`, `pvalue`,
`best_index`, `best_mean`, `n_trials`, `n_boot`, `mean_block`.

### `spa_test(returns, *, benchmark=0.0, n_boot=1000, mean_block=None, seed=None) -> SpaResult`

Hansen's (2005) superior predictive ability test: studentised, with the
`sqrt(2 log log T)` recentring correction. Fields: `statistic`, `pvalue`, `best_index`,
`best_statistic`, `n_trials`, `n_boot`.

### `romano_wolf(returns, *, benchmark=0.0, alpha=0.05, n_boot=1000, mean_block=None, seed=None) -> RomanoWolfResult`

Romano-Wolf (2005) stepdown procedure. Fields include `adjusted_pvalues`, `n_trials` and
`significant_indices()`, which returns the strategies that survive family-wise correction.

## Leakage

### `PurgedKFold(n_splits=5, *, embargo_pct=0.0, label_end_times=None)`

Splitter implementing the scikit-learn protocol. `X` for `split` and `get_n_splits` is the
**timestamp vector**, sorted ascending. `label_end_times` gives the end of each observation's
label window; when omitted it is inferred as `times[i+1]` with the last observation extended
by the median spacing.

### `audit_splits(splits, times, *, label_end_times=None, embargo=0) -> SplitAudit`

Inspect *any* iterable of `(train_indices, test_indices)` pairs and count index collisions,
training labels overlapping the test window, and embargo-window violations.
`SplitAudit.is_clean` is true when all three counts are zero.

### `check_point_in_time(feature_times, decision_times, *, publication_lag=0.0, tolerance=0.0) -> PointInTimeAudit`

Flag decisions where the newest input timestamp plus `publication_lag` exceeds the decision
time (with `tolerance` slack). Fields: `n_checks`, `n_violations`, `worst_violation`,
`violation_indices`, `publication_lag`, `tolerance`, and `passed`.

### `label_overlap(times, label_end_times) -> LabelOverlap`

Average overlap between consecutive label windows and the implied number of independent
observations (`n_obs`, `mean_overlap`, `effective_sample_size`).

### `signal_delay_curve(signal, forward_returns, *, delays=(0, 1, 2, 3, 5), periods_per_year=252) -> DelayCurve`

Sharpe ratio of the zero-mean signal-weighted portfolio at each execution delay (annualised by
`periods_per_year`). `DelayCurve.retention()` returns an array: each delay's Sharpe ratio
divided by the delay-0 value.

### `permutation_pvalue(signal, forward_returns, *, n_perm=1000, mean_block=None, metric="sharpe", seed=None) -> PermutationResult`

Block-permutation null for the signal-return relationship. Fields: `pvalue`, `observed`,
`null_mean`, `null_std`, `metric`, `n_perm`.

### `block_shuffle_indices(n_obs, block, rng) -> ndarray`

The circular block permutation used by `permutation_pvalue`, exposed for custom statistics.
Pass a `numpy.random.Generator` as `rng`.

## Ledger

### `TrialLedger(path)`

Append-only JSON Lines ledger at `path`.

- `append(hypothesis, *, params=None, metrics=None, data_fingerprint=None) -> TrialRecord`
- `records() -> tuple[TrialRecord, ...]`
- `verify() -> LedgerVerification` with `valid`, `n_records`, `first_invalid_line`, `reason`, `describe()`
- `sharpes() -> ndarray` — per-period Sharpe ratios stored in `metrics["sharpe_per_period"]`
- `parameter_sets() -> tuple[dict, ...]`

### `fingerprint_data(array) -> str`

Stable SHA-256 fingerprint of an array's dtype, shape and bytes, for pinning a record to the
exact data used.

## Reporting

### `AuditReport(*, title="Backtest audit", subject=None, metadata=None)`

- `add(section)` / `extend(sections)`
- `overall_verdict -> Verdict` — the worst section verdict
- `to_markdown() -> str`
- `to_html() -> str` — self-contained, escaped, inline CSS
- `write(path) -> Path` — `.md`, `.html` or `.htm`

### `Section(title, verdict, summary, metrics=(), notes=())`

One check. `verdict` is a `Verdict` enum member: `PASS`, `WARN`, `FAIL` or `INFO`. `metrics` is
a sequence of `(label, value)` string pairs; `notes` is a sequence of caveats.

### Section builders

`deflated_sharpe_section`, `reality_check_section`, `spa_section`, `romano_wolf_section`,
`split_audit_section`, `point_in_time_section`, `label_overlap_section`, `delay_section`,
`permutation_section`, `ledger_section` — each takes the corresponding result object and
returns a `Section` with a summary, metrics table and caveats.

## Synthetic data

### `demo_dataset(n_obs=900, n_trials=30, skill=0.0013, seed=7) -> DemoData`

Seeded synthetic dataset used by the tests and the CLI demo. Fields: `trial_returns`
(`n_obs × n_trials`), `forward_returns`, `clean_signal`, `leaky_signal`. The generator is
deterministic for a fixed `seed`; nothing is read from disk. No market data is included anywhere
in this project.

### `make_strategy_returns` / `make_skill_signal`

Lower-level generators for building your own scenarios, including deliberately leaky signals.
