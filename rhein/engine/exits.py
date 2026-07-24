"""Exit execution rules, including early stop and forced liquidation priority."""
from __future__ import annotations

import numpy as np


def find_exit(*, close, open_, low, sma, sig, t0: int, entry_idx: int,
              entry_px: float, stop_level: float, params: dict) -> tuple[float, int, str]:
    """Return the first valid exit, preserving historical rule priority exactly."""
    for j in range(entry_idx + 1, len(close)):
        day_from_signal = j - t0
        if params["use_early_stop"] and day_from_signal in params["hard_stop_days"]:
            if params["stop_intraday"] and low[j] <= stop_level:
                return (min(open_[j], stop_level) if open_[j] < stop_level else stop_level,
                        j, f"t{day_from_signal} 日内止损")
            if not params["stop_intraday"] and close[j] <= stop_level:
                return close[j], j, f"t{day_from_signal} 收盘止损"
        # Forced exit is a holding cap, but early stop above has priority on the same day.
        if params["use_forced_exit"] and day_from_signal == params["forced_exit_day"]:
            protection = entry_px * (1 - params["forced_exit_intraday_stop_pct"])
            if params["use_forced_exit_intraday_protection"] and low[j] <= protection:
                return (min(open_[j], protection) if open_[j] < protection else protection,
                        j, f"t{params['forced_exit_day']} 强制日日内保护")
            return close[j], j, f"t{params['forced_exit_day']} 强制平仓"
        after_day = max(params["hard_stop_days"]) + 1 if params["use_early_stop"] else params["entry_lag"] + 1
        if day_from_signal >= after_day:
            if params["use_exit_below_entry"] and close[j] < entry_px:
                return close[j], j, "收盘价低于入场价"
            if params["use_exit_below_sma"] and not np.isnan(sma[j]) and close[j] < sma[j]:
                return close[j], j, f"收盘价低于 SMA{params['sma_n']}"
    return close[-1], len(close) - 1, "数据结束强制平仓"
