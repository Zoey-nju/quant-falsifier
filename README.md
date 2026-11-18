# quant-falsifier

[![CI](https://github.com/Zoey-nju/quant-falsifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Zoey-nju/quant-falsifier/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/quant-falsifier.svg)](https://pypi.org/project/quant-falsifier/)
[![Python](https://img.shields.io/pypi/pyversions/quant-falsifier.svg)](https://pypi.org/project/quant-falsifier/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Falsification tooling for quantitative research.** A small, dependency-light library that
asks one uncomfortable question about a backtest: *how many things did you try before this
one worked?*

---

## Motivation

A Sharpe ratio is not a result, it is a maximum over an undisclosed search. Try 200
variants of a momentum rule and the best one will look impressive even if every variant is
worthless, because the expected maximum of 200 zero-skill Sharpe ratios is strictly positive.
The same mechanism shows up in cross-validation: overlapping labels leak test information
into the training set, and the reported out-of-sample score quietly stops being out of
sample.

`quant-falsifier` is the small layer that sits between "my notebook says 2.1 Sharpe" and
"here is what I can actually defend". It does not generate strategies and it does not place
orders. It records the search, corrects the statistics for it, screens the evaluation
protocol for leakage, and writes a report that states its own limits.

The design borrows one idea from LLM-driven research agents: hypotheses can be proposed by
anything, but their truth value has to be decided by code a reviewer can re-run. Here that
code is a few hundred lines of auditable Python.

## What it does

| Module | Question it answers |
| --- | --- |
| `TrialLedger` | What exactly was tried, in what order, and has the record been edited? (append-only JSONL, SHA-256 hash chain) |
| `deflated_sharpe_ratio` | Does the winner survive the number of trials that produced it? (Bailey & López de Prado) |
| `reality_check`, `spa_test`, `romano_wolf` | Is the best strategy better than luck, and which strategies survive a family-wise correction? (stationary bootstrap) |
| `PurgedKFold`, `audit_splits` | Does the cross-validation protocol leak? Works on splits produced by *any* library |
| `check_point_in_time` | Did any decision use data that had not been published yet? |
| `label_overlap`, `signal_delay_curve`, `permutation_pvalue` | How much of the sample is really independent, does the edge survive a one-bar delay, and is the signal-return pairing meaningful? |
| `AuditReport` | A Markdown or self-contained HTML report with per-check verdicts and a limitations section |

## Install

```bash
pip install quant-falsifier
```

Python 3.10+, NumPy is the only runtime dependency. Pandas objects are accepted anywhere an
array is expected, and never required.

## Quick start

### 1. Record the search, then deflate the winner

```python
import numpy as np
from quant_falsifier import (
    TrialLedger,
    deflated_sharpe_ratio,
    per_period_sharpes,
    sharpe_ratio,
)

rng = np.random.default_rng(0)
trial_returns = rng.normal(0.0, 0.01, size=(900, 30))  # 30 variants, 900 bars
sharpes = per_period_sharpes(trial_returns)  # per-period, not annualised

ledger = TrialLedger("trials.jsonl")
for i, sr in enumerate(sharpes):
    ledger.append(
        f"variant_{i:02d}", params={"variant": i}, metrics={"sharpe_per_period": float(sr)}
    )

best = int(np.argmax(sharpes))
result = deflated_sharpe_ratio(trial_returns[:, best], ledger.sharpes())
print(f"observed {result.observed_sr:.2f}, bar {result.expected_max_sr:.2f}, DSR {result.dsr:.3f}")
```

`DSR < 0.95` means the result is indistinguishable from the best of 30 worthless strategies.
That is the whole point: it is a number you can lose to.

### 2. Audit a cross-validation protocol you already have

```python
import numpy as np
from quant_falsifier import PurgedKFold, audit_splits

times = np.arange(1200, dtype=float)
label_ends = times + 5  # five-bar forward returns: labels overlap

naive = audit_splits(
    [(np.setdiff1d(np.arange(1200), test), test) for test in np.array_split(np.arange(1200), 5)],
    times,
    label_end_times=label_ends,
)
print(naive.is_clean, naive.n_label_overlaps)  # False 20

purged = audit_splits(
    PurgedKFold(5, label_end_times=label_ends).split(times), times, label_end_times=label_ends
)
print(purged.is_clean)  # True
```

### 3. Get a report you can attach to a pull request

```bash
quant-falsifier demo --out demo_out
```

```
Honest audit: overall verdict WARN
  PASS  Trial ledger
  WARN  Deflated Sharpe ratio
  PASS  Reality check (data snooping)
  PASS  Superior predictive ability
  PASS  Romano-Wolf stepdown
  PASS  Cross-validation leakage (purged K-fold)
  PASS  Point-in-time availability (clean signal)
  WARN  Overlapping labels
  PASS  Signal delay sensitivity
  PASS  Permutation test
Negative controls: overall verdict FAIL
  FAIL  Cross-validation leakage (naive K-fold)
  FAIL  Point-in-time availability (leaky signal)
```

The demo uses synthetic data generated from a seed, never market data. It writes two
reports: `demo_out/report.{md,html}` audits the sweep honestly, and
`demo_out/negative_controls.{md,html}` re-runs the same data through two deliberately broken
protocols so you can see each screen fire. The honest report lands on `WARN` because the
winner survives a single-strategy significance test but not the deflation for 30 trials —
that gap is the reason the library exists. The ledger is written to
`demo_out/trials.jsonl`.

### 4. Verify a ledger in CI

```bash
quant-falsifier ledger verify trials.jsonl   # exit code 1 if the chain is broken
```

## Design rules

1. **No silent cleaning.** `NaN`, infinities, zero-variance series and empty input raise.
   A validator that repairs its input cannot be trusted to judge it.
2. **Units are explicit.** All correction formulas work in per-period units, and
   annualisation happens only at the reporting boundary.
3. **Null hypotheses are explicit.** Every p-value names the null it refers to, and uses a
   `(1 + exceedances) / (B + 1)` convention so it is never exactly zero.
4. **Scope is small on purpose.** Four modules, one dependency, no plugin system. The
   interesting part of this project is what it refuses to claim.

## What it does not do

- It cannot detect every kind of look-ahead bias. The screens catch structural leakage,
  point-in-time violations, overlapping labels and delay sensitivity. A signal built from a
  subtly revised dataset will still pass.
- It does not verify that the ledger is complete. It verifies that the ledger has not been
  edited. Only disciplined logging makes the trial count true.
- It does not model transaction costs, borrow costs, capacity or market impact.
- It is not financial advice, and a passing report is not a forecast.

## Roadmap

- [ ] `v0.2`: block-bootstrap confidence intervals for Sharpe ratios and drawdowns
- [ ] `v0.2`: combinatorial purged cross-validation (CPCV) with a path-level overfitting probability
- [ ] `v0.3`: optional pandas/polars adapters that keep index metadata alongside the arrays
- [ ] `v0.3`: JSON schema for audit reports so CI can gate on machine-readable output

Roadmap items live in [Issues](https://github.com/Zoey-nju/quant-falsifier/issues) and
are not commitments.

## Development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
ruff check . && ruff format --check .
mypy
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[docs/limitations.md](docs/limitations.md) for the fine print on every check.

## References

- Bailey, D. H. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier*. Journal of Risk 15(2).
- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio*. Journal of Portfolio Management 40(5).
- White, H. (2000). *A Reality Check for Data Snooping*. Econometrica 68(5).
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. JBES 23(4).
- Romano, J. P. & Wolf, M. (2005). *Stepwise Multiple Testing as Formalized Data Snooping*. Econometrica 73(4).
- Politis, D. N. & Romano, J. P. (1994). *The Stationary Bootstrap*. JASA 89(428).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7 (purged cross-validation).

## License

Apache-2.0. See [LICENSE](LICENSE).
