"""Pure metrics shared by the web UI and the detached backtest worker."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_sharpe(ohlc: pd.DataFrame, trades: pd.DataFrame) -> float | None:
    """Return the same deterministic daily strategy Sharpe used in the UI."""
    if trades.empty or len(ohlc) < 3:
        return None
    closes = pd.to_numeric(ohlc["Close"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(ohlc["Date"], errors="coerce")
    positions = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates) if not pd.isna(date)}
    daily_returns = np.zeros(len(ohlc), dtype=float)
    for trade in trades.itertuples(index=False):
        entry_index, exit_index = positions.get(str(trade.entry)), positions.get(str(trade.exit))
        if entry_index is None or exit_index is None or exit_index <= entry_index:
            continue
        if not np.isfinite(closes[entry_index]) or closes[entry_index] == 0:
            continue
        for index in range(entry_index + 1, exit_index):
            previous, current = closes[index - 1], closes[index]
            if np.isfinite(previous) and previous != 0 and np.isfinite(current):
                daily_returns[index] = current / previous - 1
        gross_trade_return = 1 + float(trade.ret_pct) / 100
        prior_multiplier = float(np.prod(1 + daily_returns[entry_index + 1:exit_index]))
        if prior_multiplier > 0:
            daily_returns[exit_index] = gross_trade_return / prior_multiplier - 1
    volatility = float(np.std(daily_returns, ddof=1))
    return float(np.mean(daily_returns) / volatility * np.sqrt(252)) if np.isfinite(volatility) and volatility else None
