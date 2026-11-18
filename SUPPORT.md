# Support

## I have a question about using the library

Open a [GitHub Discussion](https://github.com/Zoey-nju/quant-falsifier/discussions) or an
issue labelled `question`. Include the version (`quant-falsifier --version`), the function you
called and the exact error or output.

## I think I found a bug

Open an issue using the bug report template. A reproduction matters more than a description:
five lines of `numpy` that produce the wrong number is enough.

## I think a statistic is wrong

That is the most useful report this project can get. Please include:

- the paper and equation the implementation claims to follow,
- a small input where the result differs from hand calculation or the reference implementation,
- the value you expected and the value you got.

## I want to report a security issue

Do not open an issue. See [SECURITY.md](SECURITY.md).

## I want a new check or module

Open a feature request describing the leakage or statistical failure mode you want covered,
and — if possible — a synthetic example where the existing checks stay silent. Scope is kept
small on purpose: the project prefers a well-documented negative result over a broad,
untested heuristic.

## What this project will not do

- Give investment advice, market forecasts or strategy recommendations.
- Accept real trading data, broker exports or API credentials.
- Host or run your backtest. It audits arrays you already produced.
- Certify a strategy. A passing report means the listed screens did not fire; it does not mean
  the strategy is sound.
