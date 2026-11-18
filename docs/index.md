# quant-falsifier

Falsification tooling for quantitative research.

`quant-falsifier` sits between "my notebook says 2.1 Sharpe" and "here is what I can actually
defend". It records the search, corrects the statistics for it, screens the evaluation
protocol for leakage, and writes a report that states its own limits.

It does **not** generate strategies, place orders, or certify that a result is real.

## The problem in one paragraph

A reported Sharpe ratio is a maximum over an undisclosed search. Evaluate 200 variants of a
momentum rule and the best one looks impressive even when every variant is worthless, because
the expected maximum of 200 zero-skill Sharpe ratios is strictly positive. Cross-validation
has a second version of the same problem: when labels overlap, a naive K-fold split puts test
information into the training set and the "out-of-sample" score stops being out of sample.
Both failures are invisible in the notebook that produced them.

## The four modules

| Module | Question it answers |
| --- | --- |
| `TrialLedger` | What was tried, in what order, and has the record been edited? |
| `metrics` | Does the winner survive the number of trials that produced it? |
| `multiple_testing` | Is the best strategy better than luck, and which ones survive a family-wise correction? |
| `leakage` | Does the evaluation protocol leak, and how much of the sample is really independent? |

`report` turns any subset of those results into a Markdown or self-contained HTML document.

## Install

```bash
pip install quant-falsifier
```

Python 3.10+. NumPy is the only runtime dependency; pandas objects are accepted anywhere an
array is expected and never required.

## Thirty-second tour

```python
import numpy as np
from quant_falsifier import TrialLedger, deflated_sharpe_ratio, per_period_sharpes

rng = np.random.default_rng(0)
trial_returns = rng.normal(0.0, 0.01, size=(900, 30))  # 30 variants, 900 bars
sharpes = per_period_sharpes(trial_returns)  # per-period, not annualised

ledger = TrialLedger("trials.jsonl")
for i, sr in enumerate(sharpes):
    ledger.append(f"variant_{i:02d}", metrics={"sharpe_per_period": float(sr)})

best = int(np.argmax(sharpes))
result = deflated_sharpe_ratio(trial_returns[:, best], ledger.sharpes())
print(result.as_dict())
```

Run the full demo, which writes two reports and a hash-chained ledger:

```bash
quant-falsifier demo --out demo_out
```

## Design rules

1. **No silent cleaning.** `NaN`, infinities, empty input and degenerate series raise.
2. **Units are explicit.** Correction formulas work in per-period units; annualisation happens
   at the reporting boundary.
3. **Null hypotheses are explicit.** Every p-value names its null and uses
   `(1 + exceedances) / (B + 1)` so it is never exactly zero.
4. **Scope is small on purpose.** A documented negative result beats a broad, untested
   heuristic.

Read [Concepts](concepts.md) for the statistics and [Limitations](limitations.md) for the fine
print.
