# Limitations

This page is part of the product, not a disclaimer bolted on at the end. A validation tool
whose boundary is unclear is worse than no tool, because it invites false confidence.

## What each check does not prove

### Deflated Sharpe ratio

- It corrects for the **number** of trials, not for their **correlation**. $N$ near-duplicate
  variants of one strategy are treated as $N$ independent searches, which overstates the
  penalty; genuinely independent but unreported searches understate it.
- It assumes the trial Sharpe ratios are roughly exchangeable under the null. A search in which
  some variants are obviously better motivated than others violates that.
- It says nothing about whether the return series is real. A backtest built on a vendor's
  restated prices can produce a perfectly deflated, perfectly meaningless Sharpe ratio.
- The variance across trials must be positive and meaningful. With fewer than two trials, or
  with identical trial Sharpe ratios, the function raises instead of guessing.

### Reality check, SPA and Romano-Wolf

- They test "no strategy beats the benchmark", not "the best strategy will keep working".
  A significant result is evidence against the null, not evidence of a durable edge.
- They depend on the block length. Too short and serial dependence is destroyed; too long and
  the bootstrap has too few effective blocks. The default ($n^{1/3}$) is a rule of thumb, not a
  result.
- The benchmark must be chosen before seeing the data. Choosing it afterwards reintroduces the
  selection bias the test is meant to remove.
- SPA's recentring makes it conservative against strategies that are much worse than the
  benchmark; that is deliberate, but it can hide a weak signal in a large universe.

### Purged K-fold and `audit_splits`

- Purging removes training samples whose **declared** label windows overlap the test window.
  If the declared windows are wrong — because labels use a longer horizon than documented, or
  because features are computed on a rolling window that reaches further back — the audit
  cannot see it.
- `audit_splits` checks structure, not statistics. A leaking splitter that happens to produce
  non-overlapping indices passes.
- Embargo length is a heuristic. It should reflect how long the features' information persists,
  which is a modelling assumption, not a property of the data.

### `check_point_in_time`

- It verifies that you passed timestamps consistent with the decision time. It cannot verify
  that the timestamps describe reality. If a feature is stamped with the period it refers to
  rather than the period it was published, the check passes and the leak remains.
- Publication lag is a single scalar. Real releases have irregular delays, revisions and
  restatements; a single lag is a coarse approximation.

### Overlapping labels and delay curves

- The effective sample size is an approximation based on average overlap. It is a useful
  warning, not a corrected test statistic.
- A flat delay curve is consistent with robustness, but also with a signal that simply does not
  depend on the most recent bar for reasons unrelated to leakage. The converse — a steep curve
  — is stronger evidence of a problem than a flat curve is evidence of correctness.

### Permutation test

- It tests whether the signal and the forward returns are related. A placeholder signal that
  encodes the future through an untested channel (for example, a survivorship-filtered universe)
  can still pass.
- Block shuffling preserves the signal's own autocorrelation but not its cross-dependence with
  the returns, which is the thing being tested. That is the intended null, and it is narrow.

### Trial ledger

- It proves the ledger has not been edited. It does not prove the ledger is complete. Omitting
  failed variants before the first append is invisible forever.
- Hashes are not signed. Someone who controls the whole file can rebuild a consistent chain.
  If you need third-party verifiability, publish the chain head somewhere you do not control.
- Records contain parameters verbatim and are hashed, not redacted. Do not put credentials or
  licensed data into a ledger.

## What this project does not attempt

- **Executing strategies.** No order routing, no portfolio construction, no position sizing.
- **Cost modelling.** Transaction costs, borrow, financing, capacity and market impact are out
  of scope. A gross Sharpe ratio is not a strategy.
- **Data validation.** Survivorship bias, corporate actions, point-in-time vendor quality and
  tick-level data issues are not inspected.
- **High-frequency data.** The bootstrap and label-overlap assumptions are calibrated for
  bar-level series with moderate dependence. Applicability to intraday data with strong
  microstructure effects is unverified.
- **Deep-learning or LLM-generated strategies.** The library audits the evaluation protocol, so
  it applies regardless of how a hypothesis was produced. It does not check the hypothesis
  itself, and it cannot tell whether a generated strategy was reviewed by a human.
- **Certification.** There is no combination of green verdicts that makes a strategy safe. The
  report records which screens ran; it does not issue a licence.

## Known failure modes of the library itself

- Passing annualised Sharpe ratios into `deflated_sharpe_ratio` or `probabilistic_sharpe_ratio`
  produces meaningless output. Use `per_period_sharpes` or the `deflated_sharpe_ratio` wrapper,
  which computes the ratio internally. This is the single most common way to misuse the library.
- Very short series (< 30 observations) produce unstable higher-moment estimates; the DSR will
  move substantially if a single observation changes.
- Memory grows with `n_boot × n_obs` for the index matrix. Thousands of strategies with
  hundreds of thousands of observations will be slow; the implementation loops over columns to
  keep the peak bounded, but it does not parallelise.
- Numerical edge cases (near-zero variance, exactly-tied statistics) are handled by raising
  rather than by smoothing. A user who wants a number in those conditions is asking for a
  fiction.
