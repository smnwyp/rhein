"""Synthetic visual aid for reviewing an interpreted strategy.

This is intentionally not market data and must never be used as backtest input.
Its only job is to place every source-clause ID on an illustrative t0 timeline.
"""
from __future__ import annotations

import re

import pandas as pd

from alpha_agent.domain.interpretation import ClauseCoverage, SourceClause


def _offset(text: str) -> int:
    offsets = [int(value) for value in re.findall(r"\bt\s*(\d+)\b", text, flags=re.IGNORECASE)]
    if offsets:
        return offsets[0]
    if any(word in text for word in ("基准", "单日涨幅", "RSI", "MA", "均线", "阳线")):
        return 0
    if any(word in text for word in ("入场", "买入")):
        return 1
    if any(word in text for word in ("出场", "止损", "平仓", "卖出")):
        return 4
    return 0


def build_mock_candle_timeline(source_clauses: list[SourceClause], coverage: list[ClauseCoverage]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return candles and numbered labels that map one-to-one to source clauses."""
    markers: list[dict[str, object]] = []
    marker_offsets = [_offset(clause.text) for clause in source_clauses]
    latest = max(marker_offsets, default=4)
    days = list(range(-5, max(7, latest + 4)))
    closes = [100.0 + index * 0.35 + (1.4 if day == 0 else 0) - (1.0 if day >= 4 else 0) for index, day in enumerate(days)]
    candles = pd.DataFrame({"day": days, "close": closes})
    candles["open"] = candles["close"].shift(1).fillna(candles["close"] - 0.3)
    candles["high"] = candles[["open", "close"]].max(axis=1) + 0.75
    candles["low"] = candles[["open", "close"]].min(axis=1) - 0.75
    candles["ma5"] = candles["close"].rolling(5, min_periods=1).mean()
    candles["ma10"] = candles["close"].rolling(10, min_periods=1).mean() - 0.25
    by_id = {entry.clause_id: entry for entry in coverage}
    for stack, (clause, day) in enumerate(zip(source_clauses, marker_offsets)):
        candle = candles.loc[candles["day"] == day].iloc[0]
        entry = by_id.get(clause.clause_id)
        markers.append({
            "day": day,
            "price": float(candle.high + 0.55 + (stack % 5) * 0.55),
            "label": clause.clause_id,
            "disposition": entry.disposition if entry else "missing",
        })
    return candles, pd.DataFrame(markers)
