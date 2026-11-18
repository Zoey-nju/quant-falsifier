# Contributing to quant-falsifier

Thanks for taking the time to look at this project. It is deliberately small, and the most
useful contribution is usually a sharper argument, not more code.

## Ways to contribute

- **Report a wrong answer.** If a statistic disagrees with the reference paper, that is the
  highest-value bug report this project can receive. Please include the paper, the equation
  number, the input you used and the number you got.
- **Report a false negative.** A leakage pattern the screens miss is equally important. A
  small synthetic example is worth more than a description.
- **Improve documentation.** Every check has a stated scope; if a limitation is unclear, that
  is a documentation bug.
- **Add a test.** Tests are the specification here, so a failing test that describes correct
  behaviour is a contribution on its own.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
pre-commit install
```

Run the same checks CI runs:

```bash
pytest                             # tests
ruff check .                       # lint
ruff format --check .              # formatting
mypy                               # static types
mkdocs build --strict              # documentation, optional
```

## Pull request expectations

1. **One idea per pull request.** Small diffs get reviewed faster and are easier to revert.
2. **Tests come with behaviour.** New public functions need at least one test that would fail
   without them, and validation tests for the inputs they reject.
3. **No silent cleaning.** Functions must raise on `NaN`, infinities, empty input and
   degenerate series rather than repairing them. This is the core design rule; a patch that
   relaxes it will be declined.
4. **Units are explicit.** Statistical routines work in per-period units. Annualisation
   belongs in the reporting layer only.
5. **Document the limitation.** Every new check must say what it cannot detect. A section
   without a "does not prove" note is incomplete.
6. **No market data, no credentials.** Do not commit real price series, broker exports,
   API keys or anything derived from a licensed dataset. Use `quant_falsifier.synthetic` or
   a stub.
7. **Keep the dependency footprint at NumPy.** The library installs in environments where
   nothing else is available; optional integrations belong behind an extra.

## Commit messages

Conventional prefixes are used, but the important part is the body: say *why* the change
exists, not just what moved.

```
fix(metrics): reject zero dispersion across trials

np.std([0.1, 0.1, 0.1], ddof=1) returns ~1e-17 instead of 0, so the guard
never fired and the deflated Sharpe ratio was computed against a zero
threshold. Compare against a scale-relative tolerance instead.
```

## Reporting a bug

Open an issue with the reproduction script and the exact output. If the bug is a wrong
statistic, please state the expected value and where it comes from.

## Code of conduct

Participation is covered by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Technical disagreement
is welcome; contempt is not.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contribution is licensed under Apache-2.0, as described
in [LICENSE](LICENSE).
