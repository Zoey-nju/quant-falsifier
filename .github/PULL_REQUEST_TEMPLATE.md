## What does this change?

<!-- One paragraph. Say why the change exists, not just which files moved. -->

## Type

- [ ] Bug fix (wrong number, crash, false positive/negative)
- [ ] New check or feature
- [ ] Documentation
- [ ] Refactor with no behaviour change
- [ ] Packaging, CI or tooling

## Checklist

- [ ] One idea in this pull request.
- [ ] Behaviour changes come with tests that fail without them.
- [ ] `pytest`, `ruff check .`, `ruff format --check .` and `mypy` pass locally.
- [ ] No `NaN`/infinity repair was added; validation still raises on bad input.
- [ ] Statistical routines work in per-period units; annualisation stays in the report layer.
- [ ] Any new check documents what it **cannot** detect.
- [ ] No real market data, broker exports, credentials or licensed content in the diff.
- [ ] Public API changes are reflected in `docs/api.md` and `CHANGELOG.md`.

## How was this verified?

<!-- Command lines, notebook cell, or a pointer to the test that covers it. -->

```bash
pytest
ruff check . && ruff format --check .
mypy
```

## Related issues

<!-- Fixes #123 -->
