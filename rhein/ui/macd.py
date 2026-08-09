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
MMACD_FAST_WINDOW = 10
MMACD_SLOW_WINDOW = 24
MMACD_SIGNAL_WINDOW = 8


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


def add_display_mmacd(frame: pd.DataFrame, *, close_column: str = "Close") -> pd.DataFrame:
    """Add the formula-language normalised MMACD display panel.

    ``DD = (EMA(C, 10) - EMA(C, 24)) / C * 100`` puts the fast/slow EMA
    spread on a percentage-of-price scale. ``DED = EMA(DD, 8)`` and the bar
    chart uses the Chinese-formula convention ``(DD - DED) * 2``. This is a
    chart-only calculation and does not change the strategy executor.
    """
    result = frame.copy()
    close = pd.to_numeric(result[close_column], errors="coerce")
    dd = (
        close.ewm(span=MMACD_FAST_WINDOW, adjust=False, min_periods=1).mean()
        - close.ewm(span=MMACD_SLOW_WINDOW, adjust=False, min_periods=1).mean()
    ) / close.replace(0, float("nan")) * 100
    ded = dd.ewm(span=MMACD_SIGNAL_WINDOW, adjust=False, min_periods=1).mean()
    result["MMACD_DD"] = dd
    result["MMACD_DED"] = ded
    result["MMACD_HIST"] = 2 * (dd - ded)
    return result
