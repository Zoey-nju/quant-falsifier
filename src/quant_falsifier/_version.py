"""Package version, resolved from installed metadata when available."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quant-falsifier")
except PackageNotFoundError:  # pragma: no cover - running from the source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
