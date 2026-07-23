"""Backward-compatible CLI entry point for the backtesting engine."""

from rhein.backtest import *  # noqa: F403
from rhein.backtest import main


if __name__ == "__main__":
    main()

