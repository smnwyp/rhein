"""Stable identities for the per-trade chart selector."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import re

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


def event_text_layer(layers: list[dict[str, object]]) -> dict[str, object]:
    """Find the annotation text layer without relying on layer order."""
    for layer in layers:
        mark = layer.get("mark")
        if isinstance(mark, dict) and mark.get("type") == "text":
            return layer
    raise ValueError("trade chart does not contain an event annotation text layer")


def compact_audit_overlay_lines(rows: Iterable[Mapping[str, object]], *, maximum: int = 7) -> list[str]:
    """Turn deterministic audit rows into concise, chart-safe evidence lines.

    The full tabular evidence remains available in the artifact expander.  The
    chart overlay is intentionally compact so it can sit after an exit marker
    without hiding the entry/exit candles it is meant to explain.
    """
    lines: list[str] = []
    for row in rows:
        if len(lines) >= maximum:
            break
        left, operator, right = str(row.get("左侧", "")), str(row.get("比较", "")), str(row.get("右侧", ""))
        # Group-result rows are useful in the full audit tree, but duplicate
        # their leaves inside the constrained in-chart card and can make an
        # unselected OR branch look like an entry requirement.
        if operator == "group_result" or left.startswith("组合条件"):
            continue
        left_value, right_value = row.get("左侧数值"), row.get("右侧数值")
        passed = row.get("结果") == "通过"
        prefix = "✓" if passed else "×"
        values = ""
        if left_value is not None or right_value is not None:
            values = f"  [{left_value if left_value is not None else '—'} / {right_value if right_value is not None else '—'}]"
        lines.append(f"{prefix} {left} {operator} {right}{values}".strip())
    return lines


def decisive_entry_audit_rows(rows: Iterable[Mapping[str, object]]) -> tuple[str | None, list[Mapping[str, object]]]:
    """Return only the OR entry branches that actually admitted this trade.

    ``build_trade_event_audit`` intentionally records every leaf in an OR tree
    for forensic completeness. A compact chart card must not lead with a
    rejected branch, though: it would visually suggest a false condition was
    accepted. This helper finds the passed top-level anchor branches and keeps
    their descendants. It returns ``None`` for non-OR/simple anchors.
    """
    snapshots = list(rows)
    root_pattern = re.compile(r"^anchor\.condition\.conditions\[(\d+)]$")
    passed_indices: list[int] = []
    for row in snapshots:
        match = root_pattern.match(str(row.get("DSL 路径", "")))
        if match and row.get("比较") == "group_result" and row.get("结果") == "通过":
            passed_indices.append(int(match.group(1)))
    if not passed_indices:
        return None, snapshots
    prefixes = tuple(f"anchor.condition.conditions[{index}]" for index in passed_indices)
    selected = [row for row in snapshots if str(row.get("DSL 路径", "")).startswith(prefixes)]
    branch_label = "命中买入分支：" + "、".join(f"第 {index + 1} 组" for index in passed_indices)
    return branch_label, selected
