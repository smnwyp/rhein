"""Aggregation and dataframe presentation helpers for UI result views."""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_scan(kpis: pd.DataFrame, params: dict) -> dict:
    active = kpis[kpis["n_trades"] > 0]
    total_trades = int(kpis["n_trades"].sum())
    weighted_win_rate = ((kpis["win_rate_pct"].fillna(0) * kpis["n_trades"]).sum() / total_trades
                         if total_trades else np.nan)
    gross_profit = kpis["gross_profit"].fillna(0).sum()
    gross_loss = kpis["gross_loss"].fillna(0).sum()
    return {
        "信号下限": params["band_lo"], "信号上限": params["band_hi"],
        "入场确认": f"t{params['entry_lag']}",
        "早期止损日": ",".join(f"t{day}" for day in params["hard_stop_days"]),
        "止损": params["stop_pct"], "MA": params["sma_n"],
        "交易数": total_trades, "有交易标的": len(active), "胜率_%": weighted_win_rate,
        "平均标的胜率_%": active["win_rate_pct"].mean() if len(active) else np.nan,
        "中位标的胜率_%": active["win_rate_pct"].median() if len(active) else np.nan,
        "中位标的累计收益_%": active["cumulative_return_pct"].median() if len(active) else np.nan,
        "平均标的累计收益_%": active["cumulative_return_pct"].mean() if len(active) else np.nan,
        "盈利因子": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "中位最大回撤_%": active["max_drawdown_pct"].median() if len(active) else np.nan,
        "回撤25分位数_%": active["max_drawdown_pct"].quantile(.25) if len(active) else np.nan,
        "平均每标的交易数": kpis["n_trades"].mean(),
    }


def style_by_drawdown(frame: pd.DataFrame, drawdown_column: str,
                      threshold_pct: float, integer_columns: tuple[str, ...] = ()) -> pd.io.formats.style.Styler:
    def style_row(row: pd.Series) -> list[str]:
        drawdown = row[drawdown_column]
        color = "" if pd.isna(drawdown) else ("color: #d62728" if drawdown < threshold_pct else "color: #198754")
        return [color for _ in row]

    formatters = {column: "{:,.0f}" if column in integer_columns else "{:,.2f}"
                  for column in frame.select_dtypes(include="number").columns}
    return frame.style.apply(style_row, axis=1).format(formatters, na_rep="—")
