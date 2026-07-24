"""Indicator preparation for the momentum-breakout engine."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(
    df: pd.DataFrame, *, sma_n: int, entry_trend_fast_sma: int,
    entry_trend_slow_sma: int, entry_volume_fast_window: int,
    entry_volume_slow_window: int, baseline_rsi_period: int, vol_window: int,
) -> dict[str, np.ndarray]:
    """Return the exact vectorized series historically used by ``run_backtest``."""
    close = df["Close"].to_numpy()
    volume = df["Volume"].to_numpy()
    ret1 = np.full(len(df), np.nan)
    ret1[1:] = close[1:] / close[:-1] - 1.0
    delta = pd.Series(close).diff()
    avg_gain = delta.clip(lower=0).ewm(
        alpha=1 / baseline_rsi_period, adjust=False, min_periods=baseline_rsi_period,
    ).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / baseline_rsi_period, adjust=False, min_periods=baseline_rsi_period,
    ).mean()
    return {
        "ret1": ret1,
        "sma": pd.Series(close).rolling(sma_n).mean().to_numpy(),
        "entry_fast_sma": pd.Series(close).rolling(entry_trend_fast_sma).mean().to_numpy(),
        "entry_slow_sma": pd.Series(close).rolling(entry_trend_slow_sma).mean().to_numpy(),
        "baseline_sma20": pd.Series(close).rolling(20).mean().to_numpy(),
        "vol_fast_sma": pd.Series(volume).rolling(entry_volume_fast_window).mean().to_numpy(),
        "vol_slow_sma": pd.Series(volume).rolling(entry_volume_slow_window).mean().to_numpy(),
        "rsi": (100 - 100 / (1 + avg_gain / avg_loss)).to_numpy(),
        "sig": pd.Series(ret1).rolling(vol_window).std().shift(1).to_numpy(),
    }
