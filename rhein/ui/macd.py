"""Display-only MACD calculations used by the trade chart.

This module deliberately does not participate in strategy execution.  It
keeps the chart overlay explicit and stable even when a strategy uses a
different MACD configuration in its entry or exit rules.
"""
from __future__ import annotations

import pandas as pd


MACD_FAST_WINDOW = 12
MACD_SLOW_WINDOW = 24
MACD_SIGNAL_WINDOW = 8


def add_display_macd(frame: pd.DataFrame, *, close_column: str = "Close") -> pd.DataFrame:
    """Return a copy of *frame* with DIF, DEA and MACD histogram columns.

    The convention is MACD(12, 24, 8): DIF = EMA12 - EMA24; DEA is the
    eight-period EMA of DIF; the displayed histogram is 2 × (DIF - DEA), the
    common Chinese charting convention.  ``adjust=False`` matches recursive
    trading-platform EMA calculations.
    """
    result = frame.copy()
    close = pd.to_numeric(result[close_column], errors="coerce")
    dif = close.ewm(span=MACD_FAST_WINDOW, adjust=False, min_periods=1).mean() - close.ewm(
        span=MACD_SLOW_WINDOW,
        adjust=False,
        min_periods=1,
    ).mean()
    dea = dif.ewm(span=MACD_SIGNAL_WINDOW, adjust=False, min_periods=1).mean()
    result["MACD_DIF"] = dif
    result["MACD_DEA"] = dea
    result["MACD_HIST"] = 2 * (dif - dea)
    return result
