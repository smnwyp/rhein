"""Progress-aware execution of a parameter set across a collection of symbols."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def annualized_sharpe(ohlc: pd.DataFrame, trades: pd.DataFrame) -> float | None:
    """Return a zero-risk-rate, annualized Sharpe ratio from daily strategy returns.

    The strategy enters at the close, so the entry-day return is zero.  While a
    position is open we mark it to the daily close.  On the exit day the return
    is calibrated to the recorded net trade return, which preserves fees and
    intraday/gap exit prices.  Days without a position therefore contribute a
    zero return rather than being omitted.
    """
    if trades.empty or len(ohlc) < 3:
        return None

    closes = pd.to_numeric(ohlc["Close"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(ohlc["Date"], errors="coerce")
    positions = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)
                 if not pd.isna(date)}
    daily_returns = np.zeros(len(ohlc), dtype=float)

    for trade in trades.itertuples(index=False):
        entry_index = positions.get(str(trade.entry))
        exit_index = positions.get(str(trade.exit))
        if entry_index is None or exit_index is None or exit_index <= entry_index:
            continue
        if not np.isfinite(closes[entry_index]) or closes[entry_index] == 0:
            continue

        # Mark open positions to close until the final exit day.
        for index in range(entry_index + 1, exit_index):
            previous, current = closes[index - 1], closes[index]
            if np.isfinite(previous) and previous != 0 and np.isfinite(current):
                daily_returns[index] = current / previous - 1

        gross_trade_return = 1 + float(trade.ret_pct) / 100
        prior_multiplier = float(np.prod(1 + daily_returns[entry_index + 1:exit_index]))
        if prior_multiplier > 0:
            daily_returns[exit_index] = gross_trade_return / prior_multiplier - 1

    volatility = float(np.std(daily_returns, ddof=1))
    if not np.isfinite(volatility) or volatility == 0:
        return None
    return float(np.mean(daily_returns) / volatility * np.sqrt(252))


def collect_results(paths: list[Path], params: dict, capital: float, compound: bool,
                    *, load_ohlc, run_backtest, progress_label: str = "回测中") -> tuple[pd.DataFrame, pd.DataFrame]:
    progress = st.progress(0, text=progress_label)
    status = st.empty()
    kpis, trades, errors = [], [], []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        try:
            ohlc = load_ohlc(path)
            trade_df, stats = run_backtest(ohlc, capital=capital, compound=compound, **params)
            calendar_days = max(int((ohlc["Date"].iloc[-1] - ohlc["Date"].iloc[0]).days), 1)
            equity_multiple = stats["final_equity"] / stats["initial_capital"]
            annualized_return_pct = (equity_multiple ** (365.25 / calendar_days) - 1) * 100
            sharpe_ratio = annualized_sharpe(ohlc, trade_df)
            kpis.append({
                "标的": path.stem.upper(), "源文件": str(path),
                "数据覆盖天数": calendar_days, "annualized_return_pct": annualized_return_pct,
                "sharpe_ratio": sharpe_ratio,
                **stats,
            })
            if not trade_df.empty:
                item = trade_df.copy()
                item.insert(0, "symbol", path.stem.upper())
                trades.append(item)
        except Exception as exc:
            errors.append({"source": str(path), "error": str(exc)})
        if index == total or index % max(1, total // 100) == 0:
            progress.progress(index / total, text=f"{progress_label}：{index}/{total}")
            status.caption(f"已处理 {index}/{total} 个标的")
    progress.empty()
    status.empty()
    if errors:
        st.warning(f"{len(errors)} 个文件未能运行。", icon="⚠️")
        st.dataframe(pd.DataFrame(errors), hide_index=True)
    kpi_frame = pd.DataFrame(kpis)
    if not kpi_frame.empty:
        kpi_frame["cumulative_return_pct"] = (kpi_frame["final_equity"] / kpi_frame["initial_capital"] - 1) * 100
    return kpi_frame, (pd.concat(trades, ignore_index=True) if trades else pd.DataFrame())
