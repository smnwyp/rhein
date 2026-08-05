"""Defensive, deterministic helpers for presenting persisted backtest output."""
from __future__ import annotations

import pandas as pd


# The KPI payload is persisted with each historical run.  It is therefore an
# external, versioned input to the presentation layer: older records can
# predate a metric which later becomes available in the table.  Keep the
# defaults in one place so a new presentation column is deliberately
# backwards-compatible instead of making an old run white-screen.
_PRESENTATION_KPI_DEFAULTS: dict[str, object] = {
    "标的": "",
    "n_trades": 0,
    "sharpe_ratio": float("nan"),
    "annualized_return_pct": float("nan"),
    "max_drawdown_pct": float("nan"),
    "win_rate_pct": float("nan"),
    "cumulative_return_pct": float("nan"),
    "payoff_ratio": float("nan"),
    "profit_factor": float("nan"),
    "avg_return_pct": float("nan"),
    "avg_days_held": float("nan"),
}


def presentation_kpis(kpis: pd.DataFrame) -> pd.DataFrame:
    """Return a backwards-compatible KPI view for the results UI.

    This never alters the immutable saved run.  A metric absent from an old
    record means "not available", rather than zero or a silently invented
    value.  When old rows contain gross profit/loss, their profit factor is
    still deterministically recoverable and is filled from those fields.
    """
    view = kpis.copy()
    for column, default in _PRESENTATION_KPI_DEFAULTS.items():
        if column not in view:
            view[column] = default

    # Some historical records have gross P&L but predate the row-level
    # profit-factor field.  It is safe to recover only where a positive loss
    # denominator exists; an all-winning series has no finite profit factor.
    missing_profit_factor = view["profit_factor"].isna()
    if {"gross_profit", "gross_loss"}.issubset(view.columns) and missing_profit_factor.any():
        gross_profit = pd.to_numeric(view["gross_profit"], errors="coerce")
        gross_loss = pd.to_numeric(view["gross_loss"], errors="coerce")
        recoverable = missing_profit_factor & gross_loss.gt(0)
        view.loc[recoverable, "profit_factor"] = gross_profit[recoverable] / gross_loss[recoverable]

    # JSON histories sometimes restore a numeric column as object dtype due
    # to null values.  Normalising avoids mixed-type sort failures in pandas.
    numeric_columns = [column for column in _PRESENTATION_KPI_DEFAULTS if column not in {"标的"}]
    for column in numeric_columns:
        view[column] = pd.to_numeric(view[column], errors="coerce")
    view["n_trades"] = view["n_trades"].fillna(0).astype(int)
    return view


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
