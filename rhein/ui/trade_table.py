"""Display-only helpers for ranking individual closed trades."""
from __future__ import annotations

import pandas as pd


TRADE_RANK_OPTIONS: dict[str, str] = {
    "单笔收益率（高→低）": "ret_pct",
    "交易盈亏（高→低）": "pnl_eur",
    "持仓天数（高→低）": "days_held",
    "入场日期（新→旧）": "entry",
    "出场日期（新→旧）": "exit",
}

_REQUIRED_TRADE_COLUMNS = (
    "symbol", "signal", "entry", "exit", "days_held", "entry_px", "exit_px", "ret_pct", "pnl_eur", "reason",
)


def build_trade_ranking(trades: pd.DataFrame, *, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a non-mutating all-trades ranking and its Chinese display view.

    ``trades`` remains the authoritative deterministic backtest result.  This
    function only gives every closed trade a stable row ID for the current
    result set and supplies graceful placeholders for older saved runs that
    lack a newer display field.
    """
    if metric not in TRADE_RANK_OPTIONS.values():
        raise ValueError(f"unsupported trade ranking metric: {metric}")

    ranked = trades.copy(deep=True).reset_index(drop=True)
    for column in _REQUIRED_TRADE_COLUMNS:
        if column not in ranked:
            ranked[column] = pd.NA
    ranked.insert(0, "_trade_row_id", range(len(ranked)))
    ranked = ranked.sort_values(metric, ascending=False, na_position="last", kind="stable").reset_index(drop=True)
    ranked.insert(1, "交易排名", range(1, len(ranked) + 1))

    display = ranked[[
        "交易排名", "symbol", "signal", "entry", "exit", "days_held", "entry_px", "exit_px", "ret_pct", "pnl_eur", "reason",
    ]].rename(columns={
        "symbol": "标的",
        "signal": "信号日期",
        "entry": "入场日期",
        "exit": "出场日期",
        "days_held": "持仓天数",
        "entry_px": "入场价",
        "exit_px": "出场价",
        "ret_pct": "单笔收益率 (%)",
        "pnl_eur": "交易盈亏",
        "reason": "卖出原因",
    })
    return ranked, display
