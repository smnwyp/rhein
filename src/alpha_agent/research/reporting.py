"""Defensive, deterministic helpers for presenting persisted backtest output."""
from __future__ import annotations

import pandas as pd


def aggregate_gross_pnl(kpis: pd.DataFrame, trades: pd.DataFrame) -> tuple[float, float]:
    """Return aggregate winning and losing P&L across traded symbols.

    New runs expose ``gross_profit``/``gross_loss`` in their KPI rows.  Older
    runs, or a run with zero closed trades, may not contain those optional
    columns.  The immutable trade ledger is preferred when present; otherwise
    the available KPI fields are used.  Reporting must never make a completed
    backtest page fail solely because a non-applicable metric is absent.
    """
    active = kpis[kpis["n_trades"] > 0] if "n_trades" in kpis else kpis.iloc[0:0]
    if not active.empty and {"symbol", "pnl_eur"}.issubset(trades.columns):
        relevant = trades[trades["symbol"].isin(active.get("标的", pd.Series(dtype=str)))]
        pnl = pd.to_numeric(relevant["pnl_eur"], errors="coerce").dropna()
        return float(pnl[pnl > 0].sum()), float(-pnl[pnl <= 0].sum())

    profit = pd.to_numeric(active.get("gross_profit", pd.Series(dtype=float)), errors="coerce").fillna(0)
    loss = pd.to_numeric(active.get("gross_loss", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return float(profit.sum()), float(loss.sum())
