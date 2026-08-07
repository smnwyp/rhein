"""Display-only DMI calculations used by the trade chart.

The DMI panel is deliberately separate from strategy execution.  It gives a
consistent, inspectable DMI(10, 6) view for a selected trade without changing
signals, stored results, or any strategy-specific indicator configuration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DMI_DIRECTIONAL_WINDOW = 10
DMI_ADX_WINDOW = 6


def add_display_dmi(
    frame: pd.DataFrame,
    *,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
) -> pd.DataFrame:
    """Return a copy of *frame* with Chinese-style DMI(10, 6) columns.

    ``PDI`` and ``MDI`` use ten-day rolling sums of positive/negative
    directional movement over true range.  ``ADX`` is the six-day simple
    average of DX; ``ADXR`` averages ADX with its value six days earlier.
    This is the common formula-chart convention for DMI(N, M), rather than a
    new backtest indicator or a substitute for a strategy's own DMI rules.
    """
    result = frame.copy()
    high = pd.to_numeric(result[high_column], errors="coerce")
    low = pd.to_numeric(result[low_column], errors="coerce")
    close = pd.to_numeric(result[close_column], errors="coerce")

    previous_high = high.shift(1)
    previous_low = low.shift(1)
    previous_close = close.shift(1)
    upward_move = high - previous_high
    downward_move = previous_low - low
    positive_dm = pd.Series(
        np.where((upward_move > downward_move) & (upward_move > 0), upward_move, 0.0),
        index=result.index,
        dtype="float64",
    )
    negative_dm = pd.Series(
        np.where((downward_move > upward_move) & (downward_move > 0), downward_move, 0.0),
        index=result.index,
        dtype="float64",
    )
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1, skipna=True)
    true_range = true_range.fillna(high - low)

    rolling_true_range = true_range.rolling(DMI_DIRECTIONAL_WINDOW, min_periods=1).sum()
    positive_di = (100 * positive_dm.rolling(DMI_DIRECTIONAL_WINDOW, min_periods=1).sum() / rolling_true_range).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    negative_di = (100 * negative_dm.rolling(DMI_DIRECTIONAL_WINDOW, min_periods=1).sum() / rolling_true_range).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    dx = (100 * (positive_di - negative_di).abs() / (positive_di + negative_di)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    adx = dx.rolling(DMI_ADX_WINDOW, min_periods=1).mean()

    result["DMI_PDI"] = positive_di
    result["DMI_MDI"] = negative_di
    result["DMI_ADX"] = adx
    result["DMI_ADXR"] = ((adx + adx.shift(DMI_ADX_WINDOW)) / 2).fillna(adx)
    return result
