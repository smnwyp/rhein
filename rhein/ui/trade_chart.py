"""Stable identities for the per-trade chart selector."""
from __future__ import annotations

from collections.abc import Mapping

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


def selected_event_id(chart_state: object, *, selection_name: str = "trade_event") -> str | None:
    """Extract one event ID from Streamlit's Vega-Lite selection payload.

    Streamlit/Vega-Lite versions represent point selections either as a list
    of records or as a field-to-values mapping, so this normalizes both forms.
    Invalid or non-entry/non-exit selections intentionally return ``None``.
    """
    selection = chart_state.get("selection", {}) if isinstance(chart_state, Mapping) else getattr(chart_state, "selection", {})
    selected = selection.get(selection_name) if isinstance(selection, Mapping) else None
    candidate: object | None = None
    if isinstance(selected, list) and selected and isinstance(selected[0], Mapping):
        candidate = selected[0].get("EventId")
    elif isinstance(selected, Mapping):
        candidate = selected.get("EventId")
    if isinstance(candidate, list):
        candidate = candidate[0] if candidate else None
    return str(candidate) if candidate in {"entry", "exit"} else None
