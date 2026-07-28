"""Stable identities for the per-trade chart selector."""
from __future__ import annotations

import pandas as pd


def trade_selector_options(trades: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Return immutable selector IDs and their human-readable labels.

    A row number alone is not a valid UI identity: its meaning changes when the
    selected symbol (or the cached result set) changes.  Including the three
    trade dates makes the chart and its selector refer to the exact same trade;
    the row number keeps otherwise-identical rows distinct.
    """
    option_ids: list[str] = []
    labels: dict[str, str] = {}
    for index, row in trades.iterrows():
        option_id = f"{index}|{row.signal}|{row.entry}|{row.exit}|{row.reason}"
        option_ids.append(option_id)
        labels[option_id] = (
            f"第 {index + 1} 笔：t0 {row.signal}｜入场 {row.entry}｜"
            f"出场 {row.exit}｜{row.ret_pct:+.2f}%"
        )
    return option_ids, labels
